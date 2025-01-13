import numpy as np
from PIL import Image
import os
from skimage.transform import resize
from tqdm import tqdm
from omegaconf import DictConfig
import hydra
import json
import torch
import torchvision.transforms as Tr
from torchvision import transforms, datasets
import torch.nn.functional as F
from torch import nn, einsum
import omegaconf
from tqdm import tqdm
import pandas as pd
import seaborn as sns
import cv2

OBS_ELEMENT_INDICES = {
    "bottom burner": np.array([11, 12]),
    "top burner": np.array([15, 16]),
    "light switch": np.array([17, 18]),
    "slide cabinet": np.array([19]),
    "hinge cabinet": np.array([20, 21]),
    "microwave": np.array([22]),
    "kettle": np.array([23, 24, 25, 26, 27, 28, 29]),
}


def detect_moving_objects_array(arr, obs_indices, threshold=0.005):
    """Returns a numpy array where each row corresponds to a time step and each column corresponds to an object.
    The value at each cell in the array is a boolean indicating whether the corresponding object has moved or not.

    Parameters:
    arr (np.ndarray): The input array of shape (T, 30).
    obs_indices (dict): A dictionary containing indices of objects in the array.
    threshold (float): The threshold for detecting a change in the object's value.

    Returns:
    np.ndarray: A boolean array of shape (T, len(obs_indices)) indicating which objects have moved for each time step.
    """
    moving_objects_array = np.zeros((arr.shape[0], len(obs_indices)), dtype=bool)

    for t in range(arr.shape[0]):
        for i, (obj, indices) in enumerate(obs_indices.items()):
            # Get the value of the object at the current time step
            obj_value = arr[t, indices]

            # Get the value of the object at the previous time step
            if t > 0:
                prev_obj_value = arr[t - 1, indices]
            else:
                prev_obj_value = obj_value
            # print(np.abs(obj_value - prev_obj_value)> threshold)
            # Check if the difference between the current and previous values is greater than threshold
            if (np.abs(obj_value - prev_obj_value) > threshold).any():
                moving_objects_array[t, i] = True

    return moving_objects_array


def repeat_last_proto(encode_protos, eps_len):
    rep_proto = encode_protos[-1].unsqueeze(0).repeat(eps_len - len(encode_protos), 1)
    return torch.cat([encode_protos, rep_proto])


def load_state_and_to_tensor(vid):
    state_path = os.path.join(vid, "states.json")
    with open(state_path, "r") as f:
        state_data = json.load(f)
    state_data = np.array(state_data, dtype=np.float32)
    return state_data


def load_images(folder_path, resize_shape=None):
    images = []  # initialize an empty list to store the images

    # get a sorted list of filenames in the folder
    filenames = sorted(
        [f for f in os.listdir(folder_path) if f.endswith(".png")],
        key=lambda x: int(os.path.splitext(x)[0]),
    )

    # loop through all PNG files in the sorted list
    for filename in filenames:
        # open the image file using PIL library
        img = Image.open(os.path.join(folder_path, filename))
        # convert the image to a NumPy array
        img_arr = np.array(img)
        if resize_shape is not None:
            img_arr = cv2.resize(img_arr, resize_shape)
        images.append(img_arr)  # add the image array to the list

    # convert the list of image arrays to a NumPy array
    images_arr = np.array(images)
    return images_arr


def load_model(cfg):
    exp_cfg = omegaconf.OmegaConf.load(os.path.join(cfg.exp_path, ".hydra/config.yaml"))
    model = hydra.utils.instantiate(exp_cfg.Model).to(cfg.device)

    loadpath = os.path.join(cfg.exp_path, f"epoch={cfg.ckpt}.ckpt")
    checkpoint = torch.load(loadpath, map_location=cfg.device)

    model.load_state_dict(checkpoint["state_dict"])
    model.to(cfg.device)
    model.eval()
    print("model loaded")
    return model


def convert_images_to_tensors(images_arr, pipeline):
    images_tensor = np.transpose(images_arr, (0, 3, 1, 2))  # (T,dim,h,w)
    images_tensor = torch.tensor(images_tensor, dtype=torch.float32) / 255
    images_tensor = pipeline(images_tensor)

    return images_tensor


@hydra.main(
    version_base=None,
    config_path="../config/simulation",
    config_name="label_sim_kitchen_dataset",
)
def label_dataset(cfg: DictConfig):
    model = load_model(cfg)

    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )
    pipeline = nn.Sequential(Tr.CenterCrop((112, 112)), normalize)

    datasets = ["robot", cfg.cross_embodiment] if cfg.include_robot else [cfg.cross_embodiment]

    for demo_type in datasets:
        data_path = os.path.join(cfg.data_path, demo_type)
        all_folders = os.listdir(data_path)
        all_folders = sorted(all_folders, key=lambda x: int(x))
        if cfg.plot_top_k is not None:
            all_folders = all_folders[: cfg.plot_top_k]
        for folder_path in tqdm(all_folders, disable=not cfg.verbose):
            # store the proto in proto pretrain folder
            if demo_type != 'robot':
                save_folder = os.path.join(
                    cfg.exp_path, f"{demo_type}_encode_protos", f"ckpt_{cfg.ckpt}", folder_path
                )
            else:
                save_folder = os.path.join(
                    cfg.exp_path, "encode_protos", f"ckpt_{cfg.ckpt}", folder_path
                )
            os.makedirs(save_folder, exist_ok=True)

            data_folder = os.path.join(data_path, folder_path)

            images_arr = load_images(data_folder, resize_shape=cfg.resize_shape)

            images_tensor = convert_images_to_tensors(images_arr, pipeline).cuda()

            eps_len = images_tensor.shape[0]
            im_q = torch.stack(
                [
                    images_tensor[j : j + model.slide + 1]
                    for j in range(eps_len - model.slide)
                ]
            )  # (b,slide+1,c,h,w)
            state_representation = model.encoder_q.get_state_representation(im_q, None)
            traj_representation = model.encoder_q.get_traj_representation(
                state_representation
            )
            traj_representation = repeat_last_proto(traj_representation, eps_len)
            traj_representation = traj_representation.detach().cpu().numpy()
            traj_representation = np.array(traj_representation).tolist()

            with open(os.path.join(save_folder, "traj_representation.json"), "w") as f:
                json.dump(traj_representation, f)


if __name__ == "__main__":
    label_dataset()
