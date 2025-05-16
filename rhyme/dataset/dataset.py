# from collections import namedtuple
# import numpy as np
# import torch
# from rhyme.utility.file_utils import get_subdirs
# from rhyme.utility.file_utils import load_image
# import random
# import collections
# import torchvision.transforms as T
# import pathlib
# import json
# import concurrent.futures
# import cv2
# IndexBatch = namedtuple("IndexBatch", "im_q index info")

# class EpisodeTrajDataset(torch.utils.data.Dataset):
#     def __init__(
#         self,
#         frame_sampler,
#         _allowed_dirs=[],
#         slide=None,
#         seed=None,
#         sort_numerical=True,
#         vid_mask = None,
#         max_get_threads = 4,
#         resize_shape=[135, 135],
#     ) -> None:
#         super().__init__()
#         self._frame_sampler = frame_sampler
#         self.max_get_threads=max_get_threads
#         self.resize_shape=resize_shape
#         self._seed = seed
#         self.slide = slide
#         self.sort_numerical = sort_numerical
#         if vid_mask is not None:
#             with open(vid_mask, 'r') as f:
#                 self.vid_mask = json.load(f)
#         else:
#             self.vid_mask = None

#         self._allowed_dirs = _allowed_dirs
#         print(self._allowed_dirs)

#         self.seed_rng()
#         self._indexfile = {}
#         self._build_dir_tree()

#     def seed_rng(self):
#         if self._seed:
#             random.seed(self._seed)

#     # @profile
#     def _build_dir_tree(self):
#         """Build a dict of indices for iterating over the dataset."""
#         self._dir_tree = collections.OrderedDict()
#         num_vids = 0
#         for i, path in enumerate(self._allowed_dirs):
#             vids = get_subdirs(
#                 path,
#                 nonempty=False,
#                 sort_numerical=True if self.sort_numerical else False,
#             )
#             if vids:
#                 vids = np.array(vids)
#                 if self.vid_mask is not None:
#                     vids = vids[self.vid_mask]
#                 self._dir_tree[path] = vids
#                 for j, v in enumerate(vids):
#                     self._indexfile[num_vids] = (i, j)
#                     num_vids += 1


#     @property
#     def class_names(self):
#         """The stems of the allowed video class subdirs."""
#         return [str(pathlib.Path(f).stem) for f in self._allowed_dirs]

#     def _get_video_path(self, class_idx, vid_idx):
#         """Return video paths given class and video indices.

#         Args:
#         class_idx: The index of the action class folder in the dataset directory
#             tree.
#         vid_idx: The index of the video in the action class folder to retrieve.

#         Returns:
#         A path to a video to sample in the dataset.
#         """
#         action_class = list(self._dir_tree)[class_idx]
#         return self._dir_tree[action_class][vid_idx]

#     def _get_sequence_data(self, sample,resize_shape=None):
#         frame_paths = np.array([str(f) for f in sample["frames"]])
#         frame_paths = np.take(frame_paths, sample["ctx_idxs"], axis=0)
#         frame_paths = frame_paths.flatten()

#         frames = [None for _ in range(len(frame_paths))]

#         def get_image(image_index, image_path,frames,resize_shape):
#             try:
#                 if resize_shape is not None:
#                     frame = load_image(image_path)
#                     resized_frames = cv2.resize(frame, resize_shape)
#                     frames[image_index] = resized_frames
#                 else:
#                     frames[image_index] = load_image(image_path)
#                 return True
#             except Exception as e:
#                 print(image_index,image_path)
#                 return False

#         with concurrent.futures.ThreadPoolExecutor(
#                 max_workers=self.max_get_threads) as executor:
#             futures = set()
#             for i, idx in enumerate(frame_paths):
#                 futures.add(
#                     executor.submit(get_image, i, idx, frames,resize_shape))

#             completed, futures = concurrent.futures.wait(futures)
#             for f in completed:
#                 if not f.result():
#                     raise RuntimeError('Failed to get image!')
                
#         sequence_data = np.stack(frames)  # Shape: (S * X, H, W, C)

#         return sequence_data


#     def __len__(self):
#         return len(self._indexfile)

#     # @profile
#     def __getitem__(self, idx):
#         info = {}
#         class_idx, vid_idx = self._indexfile[idx]
#         info['class_idx'] = class_idx
#         info['vid_idx'] = vid_idx
#         vid_paths = self._get_video_path(class_idx, vid_idx)
#         sample = self._frame_sampler.sample(vid_paths)
#         sequence_data = self._get_sequence_data(sample,self.resize_shape)  # (T,h,w,dim)

#         im_q = self.transform(sequence_data)
#         return IndexBatch(im_q, idx, info)


#     def transform(self, sequence_data):
#         # Horig, Worig = sequence_data.shape[1:3]
#         sequence_data = np.transpose(sequence_data, (0, 3, 1, 2)).astype(
#             np.float32)  # (T,dim,h,w)
#         sequence_data = sequence_data / 255

#         return sequence_data

# class PairedRepDataset(torch.utils.data.Dataset):
#     def __init__(
#         self,
#         frame_sampler,
#         _allowed_dirs=[],
#         slide=None,
#         seed=None,
#         sort_numerical=True,
#         vid_mask = None,
#         max_get_threads = 4,
#         resize_shape=[135, 135],
#         percentage_pairing=1,
#     ) -> None:
#         super().__init__()
#         self._frame_sampler = frame_sampler
#         self.max_get_threads=max_get_threads
#         self.resize_shape=resize_shape
#         self._seed = seed
#         self.slide = slide
#         self.sort_numerical = sort_numerical
#         self.percentage_pairing = percentage_pairing
#         if vid_mask is not None:
#             with open(vid_mask, 'r') as f:
#                 self.vid_mask = json.load(f)
#         else:
#             self.vid_mask = None

#         self._allowed_dirs = _allowed_dirs
#         print(self._allowed_dirs)

#         self.seed_rng()
#         self._indexfile = {}
#         self._build_dir_tree()

#     def seed_rng(self):
#         if self._seed:
#             random.seed(self._seed)

#     # @profile
#     def _build_dir_tree(self):
#         """Build a dict of indices for iterating over the dataset."""
#         self._dir_tree = collections.OrderedDict()
#         assert len(self._allowed_dirs) == 2
#         num_vids = 0
#         path1, path2 = self._allowed_dirs[0], self._allowed_dirs[1]
#         vids = get_subdirs(
#             path1,
#             nonempty=False,
#             sort_numerical=True if self.sort_numerical else False,
#         )
#         vids_paired = get_subdirs(
#             path2,
#             nonempty=False,
#             sort_numerical=True if self.sort_numerical else False,
#         )
#         vids = np.array(vids)
#         vids_paired = np.array(vids_paired)

#         if self.vid_mask is not None:
#             vids = vids[self.vid_mask]
#             vids_paired = vids_paired[self.vid_mask]
            
#         vid_nums = np.arange(len(vids))
#         np.random.shuffle(vid_nums)
#         vid_nums = vid_nums[:int(self.percentage_pairing * len(vids))]
#         vids, vids_paired = vids[vid_nums], vids_paired[vid_nums]
        
#         self._dir_tree[path1] = vids
#         self._dir_tree[path2] = vids_paired
#         for j, v in enumerate(vids):
#             self._indexfile[num_vids] = (0, j)
#             num_vids += 1


#     @property
#     def class_names(self):
#         """The stems of the allowed video class subdirs."""
#         return [str(pathlib.Path(f).stem) for f in self._allowed_dirs]

#     def _get_video_path(self, class_idx, vid_idx):
#         """Return video paths given class and video indices.

#         Args:
#         class_idx: The index of the action class folder in the dataset directory
#             tree.
#         vid_idx: The index of the video in the action class folder to retrieve.

#         Returns:
#         A path to a video to sample in the dataset.
#         """
#         action_class = list(self._dir_tree)[class_idx]
#         return self._dir_tree[action_class][vid_idx]

#     def _get_sequence_data(self, sample,resize_shape=None):
#         frame_paths = np.array([str(f) for f in sample["frames"]])
#         frame_paths = np.take(frame_paths, sample["ctx_idxs"], axis=0)
#         frame_paths = frame_paths.flatten()

#         frames = [None for _ in range(len(frame_paths))]

#         def get_image(image_index, image_path,frames,resize_shape):
#             try:
#                 if resize_shape is not None:
#                     frame = load_image(image_path)
#                     resized_frames = cv2.resize(frame, resize_shape)
#                     frames[image_index] = resized_frames
#                 else:
#                     frames[image_index] = load_image(image_path)
#                 return True
#             except Exception as e:
#                 print(image_index,image_path)
#                 return False

#         with concurrent.futures.ThreadPoolExecutor(
#                 max_workers=self.max_get_threads) as executor:
#             futures = set()
#             for i, idx in enumerate(frame_paths):
#                 futures.add(
#                     executor.submit(get_image, i, idx, frames,resize_shape))

#             completed, futures = concurrent.futures.wait(futures)
#             for f in completed:
#                 if not f.result():
#                     raise RuntimeError('Failed to get image!')
                
#         sequence_data = np.stack(frames)  # Shape: (S * X, H, W, C)

#         return sequence_data


#     def __len__(self):
#         return len(self._indexfile)

#     # @profile
#     def __getitem__(self, idx):
#         info = {}
#         class_idx, vid_idx = self._indexfile[idx]
#         info['class_idx'] = 0
#         info['vid_idx'] = vid_idx

#         info2 = {}
#         info2['class_idx'] = 1
#         info2['vid_idx'] = vid_idx
        
#         vid_paths = self._get_video_path(0, vid_idx)
#         vid_paired_paths = self._get_video_path(1, vid_idx)
        
#         sample = self._frame_sampler.sample(vid_paths)
#         sequence_data = self._get_sequence_data(sample,self.resize_shape)  # (T,h,w,dim)

#         sample_paired = self._frame_sampler.sample(vid_paired_paths)
#         sequence_data_paired = self._get_sequence_data(sample_paired,self.resize_shape)  # (T,h,w,dim)

#         im_q = self.transform(sequence_data)
#         im_q_paired = self.transform(sequence_data_paired)
#         return torch.Tensor(im_q), torch.Tensor(im_q_paired)


#     def transform(self, sequence_data):
#         # Horig, Worig = sequence_data.shape[1:3]
#         sequence_data = np.transpose(sequence_data, (0, 3, 1, 2)).astype(
#             np.float32)  # (T,dim,h,w)
#         sequence_data = sequence_data / 255

#         return sequence_data


# class ConcatDataset(torch.utils.data.Dataset):
#     def __init__(self, *datasets):
#         self.datasets = datasets

#     def __getitem__(self, i):
#         return tuple(d[i] for d in self.datasets)

#     def __len__(self):
#         return min(len(d) for d in self.datasets)

# class ConcatDatasetMax(torch.utils.data.Dataset):
#     def __init__(self, *datasets):
#         self.datasets = datasets
#         self.max_len = max(len(d) for d in self.datasets)

#     def __getitem__(self, i):
#         return tuple(d[i % len(d)] for d in self.datasets)

#     def __len__(self):
#         return self.max_len


# # consider changing to this
# # class ConcatDataset(torch.utils.data.Dataset):
# #     def __init__(self, *datasets):
# #         self.datasets = datasets

# #     def __getitem__(self, i):
# #         return tuple(d[i%len(d)] for d in self.datasets)

# #     def __len__(self):
# #         return max(len(d) for d in self.datasets)



from datasets import load_dataset, Image
import numpy as np
import torch
from collections import namedtuple
import random
import collections
import torchvision.transforms as T
import pathlib
import concurrent.futures
import cv2
import json

IndexBatch = namedtuple("IndexBatch", "im_q index info")

class EpisodeTrajDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        frame_sampler,
        dataset_name='prithwishdan/RHyME',
        config_name='robot',
        split='train',
        slide=None,
        seed=None,
        max_get_threads=4,
        resize_shape=[135, 135],
    ) -> None:
        super().__init__()
        self._frame_sampler = frame_sampler
        self.max_get_threads = max_get_threads
        self.resize_shape = resize_shape
        self._seed = seed
        self.slide = slide
        
        # Load dataset from huggingface
        self.dataset = load_dataset(dataset_name, config_name, split=split)
        print(f"Loaded dataset {dataset_name}/{config_name} split {split} with {len(self.dataset)} samples")
        # Create Image decoder for handling encoded frames
        self.image_decoder = Image()
        
        self.seed_rng()
        self._build_index()

    def seed_rng(self):
        if self._seed:
            random.seed(self._seed)

    def _build_index(self):
        """Build indices for the dataset."""
        self._indexfile = {i: i for i in range(len(self.dataset))}

    def _get_sequence_data(self, sample, ctx_idxs):
        """Process a sample to get sequence data.
        
        Args:
            sample: A sample from the huggingface dataset
            ctx_idxs: Indices of frames to extract
            
        Returns:
            sequence_data: Numpy array of shape (T, H, W, C)
        """
        frames_encoded = sample["frames"]
        frame_indices = [idx.item() for idx in ctx_idxs]
        
        processed_frames = [None for _ in range(len(frame_indices))]
        
        def process_image(image_index, frame_idx, frames_encoded, processed_frames, resize_shape):
            try:
                # Decode the encoded frame
                frame_pil = self.image_decoder.decode_example(frames_encoded[frame_idx])
                
                # Convert PIL image to numpy array
                frame_np = np.array(frame_pil)
                
                # Resize if needed
                if resize_shape is not None:
                    frame_np = cv2.resize(frame_np, tuple(resize_shape))
                
                processed_frames[image_index] = frame_np
                return True
            except Exception as e:
                print(f"Error processing image {image_index}, frame {frame_idx}: {e}")
                return False
        
        with concurrent.futures.ThreadPoolExecutor(
                max_workers=self.max_get_threads) as executor:
            futures = set()
            for i, frame_idx in enumerate(frame_indices):
                futures.add(
                    executor.submit(process_image, i, frame_idx, frames_encoded, processed_frames, self.resize_shape))

            completed, futures = concurrent.futures.wait(futures)
            for f in completed:
                if not f.result():
                    raise RuntimeError('Failed to process image!')
                
        sequence_data = np.stack(processed_frames)  # Shape: (S * X, H, W, C)
        return sequence_data

    def __len__(self):
        return len(self._indexfile)

    def __getitem__(self, idx):
        info = {}
        dataset_idx = self._indexfile[idx]
        info['dataset_idx'] = dataset_idx
        
        # Get the sample directly from the huggingface dataset
        sample = self.dataset[dataset_idx]
        info['video_id'] = sample["video_id"]
        
        # Use the frame sampler to get frame indices
        frames = sample['frames']
        frame_idxs = self._frame_sampler._sample(frames)
        sampled_data = {
            "frames": frames,
            "frame_idxs": frame_idxs,
            "vid_len": len(frames),
            "ctx_idxs": self._frame_sampler._get_context_steps(frame_idxs, len(frames)),
        }
        
        # Get the frames based on the sampled indices
        sequence_data = self._get_sequence_data(sampled_data, sampled_data["ctx_idxs"])  # (T,h,w,dim)

        im_q = self.transform(sequence_data)
        return IndexBatch(im_q, idx, info)

    def transform(self, sequence_data):
        # Horig, Worig = sequence_data.shape[1:3]
        sequence_data = np.transpose(sequence_data, (0, 3, 1, 2)).astype(
            np.float32)  # (T,dim,h,w)
        sequence_data = sequence_data / 255
        return sequence_data


class PairedRepDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        frame_sampler,
        dataset_name='prithwishdan/RHyME',
        config_name='sphere-easy+robot',  # Parse embodiment names from this
        split='train',
        slide=None,
        seed=None,
        max_get_threads=4,
        resize_shape=[135, 135],
        percentage_pairing=1,
    ) -> None:
        super().__init__()
        self._frame_sampler = frame_sampler
        self.max_get_threads = max_get_threads
        self.resize_shape = resize_shape
        self._seed = seed
        self.percentage_pairing = percentage_pairing
        self.slide = slide
        
        # Parse embodiment names from config_name
        if '+' in config_name:
            self.embodiment_names = tuple(config_name.split('+'))
            print(f"Parsed embodiment names: {self.embodiment_names}")
        else:
            raise ValueError(f"Config name '{config_name}' does not contain '+' separator for embodiment names")
        
        config_name = f"{config_name}_paired"
        # Load combined dataset from huggingface
        self.dataset = load_dataset(dataset_name, config_name, split=split)
        print(f"Loaded paired dataset {dataset_name}/{config_name} split {split} with {len(self.dataset)} samples")
        
        # Create Image decoder for handling encoded frames
        self.image_decoder = Image()
        
        self.seed_rng()
        self._build_index()

    def seed_rng(self):
        if self._seed:
            random.seed(self._seed)

    def _build_index(self):
        """Build indices for paired data."""
        # Create indices for all samples
        all_indices = np.arange(len(self.dataset))
        
        # Shuffle and select percentage_pairing if needed
        if self.percentage_pairing < 1:
            np.random.shuffle(all_indices)
            all_indices = all_indices[:int(self.percentage_pairing * len(all_indices))]
        
        # Create index mapping
        self._indexfile = {i: idx.item() for i, idx in enumerate(all_indices)}

    def _get_sequence_data(self, sample, ctx_idxs):
        """Process a sample to get sequence data.
        
        Args:
            sample: A sample from the huggingface dataset
            ctx_idxs: Indices of frames to extract
            
        Returns:
            sequence_data: Numpy array of shape (T, H, W, C)
        """
        frames_encoded = sample["frames"]
        frame_indices = [idx.item() for idx in ctx_idxs]
        
        processed_frames = [None for _ in range(len(frame_indices))]
        
        def process_image(image_index, frame_idx, frames_encoded, processed_frames, resize_shape):
            try:
                # Decode the encoded frame
                frame_pil = self.image_decoder.decode_example(frames_encoded[frame_idx])
                
                # Convert PIL image to numpy array
                frame_np = np.array(frame_pil)
                
                # Resize if needed
                if resize_shape is not None:
                    frame_np = cv2.resize(frame_np, tuple(resize_shape))
                
                processed_frames[image_index] = frame_np
                return True
            except Exception as e:
                print(f"Error processing image {image_index}, frame {frame_idx}: {e}")
                return False
        
        with concurrent.futures.ThreadPoolExecutor(
                max_workers=self.max_get_threads) as executor:
            futures = set()
            for i, frame_idx in enumerate(frame_indices):
                futures.add(
                    executor.submit(process_image, i, frame_idx, frames_encoded, processed_frames, self.resize_shape))

            completed, futures = concurrent.futures.wait(futures)
            for f in completed:
                if not f.result():
                    raise RuntimeError('Failed to process image!')
                
        sequence_data = np.stack(processed_frames)  # Shape: (S * X, H, W, C)
        return sequence_data

    def __len__(self):
        return len(self._indexfile)

    def __getitem__(self, idx):
        dataset_idx = self._indexfile[idx]
        
        # Get the sample from the dataset
        sample = self.dataset[dataset_idx]
        
        # Get the embodiment names
        embodiment1_name, embodiment2_name = self.embodiment_names
        
        # Use the original frame sampler with the adapted samples
        frames1 = sample[f"frames_{embodiment1_name}"]
        frame_idxs1 = self._frame_sampler._sample(frames1)
        sampled_data1 = {
            "frames": frames1,
            "frame_idxs": frame_idxs1,
            "vid_len": len(frames1),
            "ctx_idxs": self._frame_sampler._get_context_steps(frame_idxs1, len(frames1)),
        }

        frames2 = sample[f"frames_{embodiment2_name}"]
        frame_idxs2 = self._frame_sampler._sample(frames2)
        sampled_data2 = {
            "frames": frames2,
            "frame_idxs": frame_idxs2,
            "vid_len": len(frames2),
            "ctx_idxs": self._frame_sampler._get_context_steps(frame_idxs2, len(frames2)),
        }
        
        # Get sequence data for both embodiments
        sequence_data1 = self._get_sequence_data(sampled_data1, sampled_data1["ctx_idxs"])
        sequence_data2 = self._get_sequence_data(sampled_data2, sampled_data2["ctx_idxs"])

        # Transform the sequence data
        im_q1 = self.transform(sequence_data1)
        im_q2 = self.transform(sequence_data2)
        
        return torch.Tensor(im_q1), torch.Tensor(im_q2)

    def transform(self, sequence_data):
        sequence_data = np.transpose(sequence_data, (0, 3, 1, 2)).astype(
            np.float32)  # (T,dim,h,w)
        sequence_data = sequence_data / 255
        return sequence_data


# The ConcatDataset classes remain unchanged
class ConcatDataset(torch.utils.data.Dataset):
    def __init__(self, *datasets):
        self.datasets = datasets

    def __getitem__(self, i):
        return tuple(d[i] for d in self.datasets)

    def __len__(self):
        return min(len(d) for d in self.datasets)

class ConcatDatasetMax(torch.utils.data.Dataset):
    def __init__(self, *datasets):
        self.datasets = datasets
        self.max_len = max(len(d) for d in self.datasets)

    def __getitem__(self, i):
        return tuple(d[i % len(d)] for d in self.datasets)

    def __len__(self):
        return self.max_len