from collections import defaultdict, namedtuple
import numpy as np
import torch
from rhyme.utility.file_utils import get_subdirs, get_files
import random
import collections
import os
import os.path as osp
import json
from PIL import Image
from tqdm import tqdm
from pathlib import Path
from tqdm import tqdm
import cv2
from rhyme.utility.eval_utils import gif_of_clip
from omegaconf import DictConfig

normalize_threshold = 5e-2


def create_sample_indices(
    episode_ends: np.ndarray,
    sequence_length: int,
    pad_before: int = 0,
    pad_after: int = 0,
):
    indices = list()
    for i in range(len(episode_ends)):
        start_idx = 0
        if i > 0:
            start_idx = episode_ends[i - 1]
        end_idx = episode_ends[i]
        episode_length = end_idx - start_idx

        min_start = -pad_before
        max_start = episode_length - sequence_length + pad_after

        # range stops one idx before end
        for idx in range(min_start, max_start + 1):
            buffer_start_idx = max(idx, 0) + start_idx
            buffer_end_idx = min(idx + sequence_length, episode_length) + start_idx
            start_offset = buffer_start_idx - (idx + start_idx)
            end_offset = (idx + sequence_length + start_idx) - buffer_end_idx
            sample_start_idx = 0 + start_offset
            sample_end_idx = sequence_length - end_offset
            indices.append(
                [buffer_start_idx, buffer_end_idx, sample_start_idx, sample_end_idx]
            )
    indices = np.array(indices)
    return indices


def sample_sequence(
    train_data,
    sequence_length,
    buffer_start_idx,
    buffer_end_idx,
    sample_start_idx,
    sample_end_idx,
):
    result = dict()
    for key, input_arr in train_data.items():
        sample = input_arr[buffer_start_idx:buffer_end_idx]
        data = sample
        if (sample_start_idx > 0) or (sample_end_idx < sequence_length):
            data = np.zeros(
                shape=(sequence_length,) + input_arr.shape[1:], dtype=input_arr.dtype
            )
            if sample_start_idx > 0:
                data[:sample_start_idx] = sample[0]
            if sample_end_idx < sequence_length:
                data[sample_end_idx:] = sample[-1]
            data[sample_start_idx:sample_end_idx] = sample
        result[key] = data
    return result


# normalize data
def get_data_stats(data):
    data = data.reshape(-1, data.shape[-1])
    stats = {"min": np.min(data, axis=0), "max": np.max(data, axis=0)}
    return stats


def normalize_data(data, stats):
    # nomalize to [0,1]
    ndata = data.copy()
    for i in range(ndata.shape[1]):
        if stats["max"][i] - stats["min"][i] > normalize_threshold:
            ndata[:, i] = (data[:, i] - stats["min"][i]) / (
                stats["max"][i] - stats["min"][i]
            )
            # normalize to [-1, 1]
            ndata[:, i] = ndata[:, i] * 2 - 1
    return ndata


def unnormalize_data(ndata, stats):
    data = ndata.copy()
    for i in range(ndata.shape[1]):
        if stats["max"][i] - stats["min"][i] > normalize_threshold:
            ndata[:, i] = (ndata[:, i] + 1) / 2
            data[:, i] = (
                ndata[:, i] * (stats["max"][i] - stats["min"][i]) + stats["min"][i]
            )
    return data


class KitchenBCDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data_dirs,
        proto_dirs,
        pred_horizon,
        obs_horizon,
        action_horizon,
        resize_shape=None,
        proto_horizon=None,
        raw_representation=False,
        softmax_prototype=False,
        prototype=False,
        one_hot_prototype=False,
        prototype_snap=False,
        snap_frames=100,
        mask=None,
        obs_image_based=False,
        unnormal_list=[],
        pipeline=None,
        verbose=False,
        seed=0,
        human_type='human',
        paired_data=False,
        paired_percent=0,
        paired_proto_dirs=None,
        paired_demo_img_path=None,
        nearest_neighbor_replacement=False,
        replace_percent=0,
        nearest_neighbor_data_dirs=None,
        save_lookups=False,
    ):
        """
        Support 1) raw representation 2) softmax prototype 3) prototype 4) one-hot prototype
        """
        self.verbose = verbose
        self.resize_shape = resize_shape
        if mask is not None:
            with open(mask, "r") as f:
                self.mask = json.load(f)
        else:
            self.mask = None

        self.seed = seed
        self.set_seed(self.seed)
        self.raw_representation = raw_representation
        self.softmax_prototype = softmax_prototype
        self.prototype = prototype
        self.one_hot_prototype = one_hot_prototype
        self.obs_image_based = obs_image_based
        self.prototype_snap = prototype_snap
        self.snap_frames = snap_frames
        self.pipeline = pipeline
        self.unnormal_list = unnormal_list
        self.human_type = human_type
        self.paired_data = paired_data
        self.paired_percent = paired_percent
        self.paired_proto_dirs = paired_proto_dirs
        self.paired_demo_img_path = paired_demo_img_path
        self.nearest_neighbor_replacement = nearest_neighbor_replacement
        self.replace_percent = replace_percent
        self.nearest_neighbor_data_dirs = nearest_neighbor_data_dirs
        self.save_lookups = save_lookups

        self.data_dirs = data_dirs
        self.proto_dirs = proto_dirs
        self._build_dir_tree()

        train_data = defaultdict(list)
        self.load_data(train_data)

        episode_ends = []
        for eps_action_data in train_data["actions"]:
            episode_ends.append(len(eps_action_data))
        
        i = 0
        for k, v in tqdm(train_data.items()):
            try:
                train_data[k] = np.concatenate(v)
            except MemoryError:
                print(f"MemoryError: Could not concatenate arrays for key {k}. Consider processing data in smaller chunks.")
            i += 1
        
        print(f"training data len {len(train_data['actions'])}")

        # Marks one-past the last index for each episode
        episode_ends = np.cumsum(episode_ends)
        self.episode_ends = episode_ends

        # compute start and end of each state-action sequence
        # also handles padding
        indices = create_sample_indices(
            episode_ends=episode_ends,
            sequence_length=pred_horizon,
            # add padding such that each timestep in the dataset are seen
            pad_before=obs_horizon - 1,
            pad_after=action_horizon - 1,
        )

        # compute statistics and normalized data to [-1,1]
        stats = dict()
        # normalized_train_data = dict()
        for key, data in train_data.items():
            if key == "images" or key in self.unnormal_list:
                pass
            else:
                stats[key] = get_data_stats(data)

            if key == "images" or key in self.unnormal_list:
                pass
            else:
                train_data[key] = normalize_data(data, stats[key])

        self.indices = indices
        self.stats = stats
        # self.normalized_train_data = normalized_train_data
        self.normalized_train_data = train_data
        self.pred_horizon = pred_horizon
        self.action_horizon = action_horizon
        self.obs_horizon = obs_horizon
        if proto_horizon is None:
            self.proto_horizon = obs_horizon
        else:
            self.proto_horizon = proto_horizon

    def set_seed(self, seed):
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)

    def _build_dir_tree(self):
        """Build a dict of indices for iterating over the dataset."""
        self._dir_tree = collections.OrderedDict()
        for i, path in enumerate(self.data_dirs):
            vids = get_subdirs(
                path,
                nonempty=False,
                sort_numerical=True,
            )
            if vids:
                vids = np.array(vids)
                if self.mask is not None:
                    vids = vids[self.mask]
                self._dir_tree[path] = vids

    def load_action_and_to_tensor(self, vid):
        action_path = os.path.join(vid, "actions.json")
        with open(action_path, "r") as f:
            action_data = json.load(f)
        action_data = np.array(action_data)
        action_data = np.array(action_data, dtype=np.float32)
        return action_data

    def load_state_and_to_tensor(self, vid):
        state_path = os.path.join(vid, "states.json")
        with open(state_path, "r") as f:
            state_data = json.load(f)
        state_data = np.array(state_data, dtype=np.float32)
        return state_data

    def find_episode_and_frame(self, human_z_idx, idx_dict):
        """
        Given a list of human z indices from the data bank and a dictionary that maps episode numbers to human z indices,
        extracts the true episode number and the frame from within that episode.


        Parameters
        ----------
        human_z_idx : numpy.ndarray
            List of indices (ranging between 0 and the total number of z's in the bank)
        idx_dict : dictionary
            Key: episode num, Value: 2-element list with lower (inclusive) and upper (exclusive) index bound

        Ex) human_z_idx = [5, 260]
            idx_dict = {0: [0, 250], 250: [250, 500]}
            (human z indices 0-249 come from episode 0, indices 250-499 come from episode 250)
        

        Returns 
        -------
        episode_numbers : numpy.ndarray
            episode_number[i] = episode number for which human_z_idx[i] comes from
        frames : numpy.ndarray
            frame[i] = frame number within episode_number[i]
        """
        episode_numbers = []
        frames = []
        for idx in human_z_idx:
            for ep_num, (lower, upper) in idx_dict.items():
                if lower <= idx < upper:
                    episode_numbers.append(ep_num)
                    frames.append(idx-lower)
                    break

        return np.array(episode_numbers).astype(np.int32), np.array(frames).astype(np.int32)

    def load_proto_and_to_tensor(self, vid, is_paired=False, is_lookup=False):
        proto_path = osp.join(self.proto_dirs, os.path.basename(os.path.normpath(vid)))

        def add_representation_suffix(path):
            if self.raw_representation:
                return os.path.join(path, "traj_representation.json")
            elif self.softmax_prototype or self.one_hot_prototype:
                return os.path.join(path , "softmax_encode_protos.json")
            elif self.prototype:
                return os.path.join(path, "encode_protos.json")

        proto_path = add_representation_suffix(proto_path)
        
        with open(proto_path, "r") as f:
            proto_data = json.load(f)
        proto_data = np.array(proto_data, dtype=np.float32)  # (T,D)
        if self.one_hot_prototype:
            one_hot_proto = np.zeros_like(proto_data)
            max_proto = np.argmax(proto_data, axis=1)
            one_hot_proto[np.arange(len(proto_data)), max_proto] = 1
            proto_data = one_hot_proto

        if self.prototype_snap:
            cur_proto_data = proto_data
            if is_paired: # loads a human sequence of z's as well as the robot z_t
                human_proto_path = osp.join(self.paired_proto_dirs, os.path.basename(os.path.normpath(vid)))
                human_proto_path = add_representation_suffix(human_proto_path)
                with open(human_proto_path, "r") as f:
                    human_proto_data = json.load(f)
                human_proto_data = np.array(human_proto_data, dtype=np.float32) # (T,D)
                cur_proto_data = human_proto_data                

            if is_lookup: # does nearest neighbor replacement on the robot sequence of z's
                l2_dist_path = osp.join(self.nearest_neighbor_data_dirs, os.path.basename(os.path.normpath(vid)))
                l2_dist_path = os.path.join(l2_dist_path, 'l2_dists.json')
                with open(l2_dist_path, "r") as f:
                    l2_dist_data = json.load(f)
                l2_dist_data = np.array(l2_dist_data, dtype=np.float32)
                eps_len = len(l2_dist_data)
                snap_idx = random.sample(list(range(eps_len)), k=self.snap_frames)
                snap_idx.sort()

                l2_dist_data = l2_dist_data[snap_idx]
                human_z_idx = np.argmin(l2_dist_data, axis=1)
                with open(os.path.join(self.nearest_neighbor_data_dirs, 'vid_to_idx_range.json'), 'r') as f:
                    idx_dict = json.load(f)

                z_tilde = []
                episode_nums, frame_nums = self.find_episode_and_frame(human_z_idx, idx_dict)
                cfg = DictConfig({'data_path': self.paired_demo_img_path, 'resize_shape': [124,124]})
                reconstructed_video = []
                orig_video = []
                robot_vid_num = int(os.path.basename(os.path.normpath(vid)))
                for k, (ep_num, frame_num) in enumerate(zip(episode_nums, frame_nums)):
                    human_proto_path = osp.join(self.paired_proto_dirs, os.path.basename(os.path.normpath(str(ep_num))))
                    human_proto_path = add_representation_suffix(human_proto_path)
                    with open(human_proto_path, "r") as f:
                        human_proto_data = json.load(f)

                    # NN Replacement
                    z_tilde.append(human_proto_data[int(frame_num)])
                    
                    if self.save_lookups and k % 9 == 0:
                        human_imgs = gif_of_clip(cfg, self.human_type, ep_num, frame_num, 8, None, save=False)
                        robot_imgs = gif_of_clip(cfg, 'robot', robot_vid_num, snap_idx[k], 8, None, save=False)
                        reconstructed_video.extend(human_imgs)
                        orig_video.extend(robot_imgs)

                if self.save_lookups:
                    reconstructed_video[0].save(os.path.join(self.nearest_neighbor_data_dirs, str(robot_vid_num), f'constructed_human.gif'), save_all=True, append_images=reconstructed_video[1:], duration=200, loop=0)
                    orig_video[0].save(os.path.join(self.nearest_neighbor_data_dirs, str(robot_vid_num), f'orig_robot.gif'), save_all=True, append_images=orig_video[1:], duration=200, loop=0)
                
                snap = np.array(z_tilde, dtype=np.float32)
                snap = snap.flatten()
                # TODO: fix the number of times its tiled
                snap = np.tile(snap, (len(cur_proto_data), 1))  # (T,snap_frams*model_dim)
                return proto_data, snap
            
            human_eps_len = len(cur_proto_data)
            robot_eps_len = len(proto_data)
            snap_idx = random.sample(list(range(human_eps_len)), k=self.snap_frames)
            snap_idx.sort()
            snap = cur_proto_data[snap_idx]
            snap = snap.flatten()
            snap = np.tile(snap, (robot_eps_len, 1))  # (T,snap_frams*model_dim)
            return proto_data, snap

        return proto_data

    def load_images(self, vid):
        images = []  # initialize an empty list to store the images

        # get a sorted list of filenames in the folder
        filenames = sorted(
            [f for f in os.listdir(Path(vid)) if f.endswith(".png")],
            key=lambda x: int(os.path.splitext(x)[0]),
        )

        # loop through all PNG files in the sorted list
        for filename in filenames:
            # open the image file using PIL library
            img = Image.open(os.path.join(vid, filename))
            # convert the image to a NumPy array
            img_arr = np.array(img)
            if self.resize_shape is not None:
                img_arr = cv2.resize(img_arr, self.resize_shape)
            images.append(img_arr)  # add the image array to the list

        # convert the list of image arrays to a NumPy array
        images_arr = np.array(images)
        assert images_arr.dtype == np.uint8
        return images_arr

    def transform_images(self, images_arr):
        images_arr = images_arr.astype(np.float32)
        images_tensor = np.transpose(images_arr, (0, 3, 1, 2)) / 255.0  # (T,dim,h,w)
        return images_tensor

    def load_data(self, train_data):
        assert not (self.paired_data and self.nearest_neighbor_replacement)
        # HACK. Fix later
        vid = list(self._dir_tree.values())[0]
        print("loading data")

        paired_set = set()
        if self.paired_data:
            vid_nums = np.array(vid)
            np.random.shuffle(vid_nums)
            paired_set.update(vid_nums[:int(len(vid) * self.paired_percent)])

        lookup_set = set()
        if self.nearest_neighbor_replacement:
            vid_nums = np.array(vid)
            np.random.shuffle(vid_nums)
            lookup_set.update(vid_nums[:int(len(vid) * self.replace_percent)])
        
        for j, v in tqdm(enumerate(vid), desc="Loading data", disable=not self.verbose):
            if self.obs_image_based:
                images = self.load_images(v)
                train_data["images"].append(images)

            train_data["obs"].append(self.load_state_and_to_tensor(v))
            if self.prototype_snap:
                proto_data, proto_snap = self.load_proto_and_to_tensor(v, is_paired=(v in paired_set), is_lookup=(v in lookup_set))
                train_data["proto_snap"].append(proto_snap)
            else:
                proto_data = self.load_proto_and_to_tensor(v)

            train_data["protos"].append(proto_data)
            train_data["actions"].append(self.load_action_and_to_tensor(v))
        

    def __len__(self):
        # all possible segments of the dataset
        return len(self.indices)

    def __getitem__(self, idx):
        # get the start/end indices for this datapoint
        (
            buffer_start_idx,
            buffer_end_idx,
            sample_start_idx,
            sample_end_idx,
        ) = self.indices[idx]

        # get nomralized data using these indices
        nsample = sample_sequence(
            train_data=self.normalized_train_data,
            sequence_length=self.pred_horizon,
            buffer_start_idx=buffer_start_idx,
            buffer_end_idx=buffer_end_idx,
            sample_start_idx=sample_start_idx,
            sample_end_idx=sample_end_idx,
        )
        # discard unused observations
        nsample["obs"] = nsample["obs"][: self.obs_horizon, :]
        if self.prototype_snap:
            # set as prediction target
            nsample["protos"] = nsample["protos"][: self.obs_horizon, :]
            # most recent prototype
            nsample["protos"] = nsample["protos"][-1:, :]
            # duplicate. only take one
            nsample["proto_snap"] = nsample["proto_snap"][-1:, :]
        else:
            nsample["protos"] = nsample["protos"][: self.obs_horizon, :]
            nsample["protos"] = nsample["protos"][-self.proto_horizon :, :]

        if self.obs_image_based:
            nsample["images"] = self.transform_images(nsample["images"])
            nsample["images"] = nsample["images"][: self.obs_horizon, :]
            nsample["obs"] = nsample["obs"][: self.obs_horizon, :9]
        
        return nsample
