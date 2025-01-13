import numpy as np
from PIL import Image
import random
import os
import os.path as osp
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
import json
import matplotlib.pyplot as plt
from xskill.utility.eval_utils import gif_of_clip, traj_representations, load_model

def find_episode_and_frame(human_z_idx, idx_dict):
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

def add_representation_suffix(path):
    return os.path.join(path, "traj_representation.json")

def list_digit_folders(directory):
    # List all items in the directory
    items = os.listdir(directory)
    
    # Filter out only the folders whose names are composed of digits
    digit_folders = [item for item in items if os.path.isdir(os.path.join(directory, item)) and item.isdigit()]
    
    return digit_folders

def copy_images(source_folder, new_folder):
    import shutil
    # Create the new folder if it doesn't exist
    if not os.path.exists(new_folder):
        os.makedirs(new_folder)

    # Get the number of existing files in the new folder
    num_existing_files = len([name for name in os.listdir(new_folder) if os.path.isfile(os.path.join(new_folder, name))])

    # Get a sorted list of files in the source folder based on the numerical part of the filename
    files = sorted(os.listdir(source_folder), key=lambda x: int(os.path.splitext(x)[0]))

    # Iterate through the sorted files in the source folder
    for i, file_name in enumerate(files):
        # Generate the new file name
        new_file_name = str(num_existing_files + i) + '.png'
        
        # Copy the image from the source folder to the new folder with the new name
        shutil.copy(os.path.join(source_folder, file_name), os.path.join(new_folder, new_file_name))


@hydra.main(
    version_base=None,
    config_path="../config/simulation",
    config_name="segment_wise_dists",
)
def main(cfg: DictConfig):
    """
    Computes temporal cyclic consistency metrics between human and robot videos.


    Parameters
    ----------
    cfg : DictConfig
        Specifies configuration details associated with the visual encoder being tested.
        - Note that exp_path must be the path to a trained model
        - correct_thresholds specifies the range of frames to consider for TCC accuracy

    Side Effects
    ------------
    - Prints out TCC accuracy over the correct_thresholds
    - save_clips == True will save gifs from the videos and a plot of TCC accuracy

    Returns 
    -------
    None
    """
    data_path = os.path.join(cfg.data_path, 'robot')
    all_folders = list_digit_folders(data_path)
    all_folders = sorted(all_folders, key=lambda x: int(x))
    for folder_path in tqdm(all_folders, disable=not cfg.verbose):
        new_episode_folder = os.path.join(
            cfg.data_path, f"{cfg.pretrain_model_name}_{cfg.cross_embodiment_segments}_generated_ot_{cfg.num_chops}_ckpt{cfg.ckpt}", folder_path
        )
        if cfg.ot_lookup:
            os.makedirs(new_episode_folder, exist_ok=True)
            ot_dist_path = osp.join(cfg.nearest_neighbor_data_dirs, f'{folder_path}')
            for j in range(cfg.num_chops):
                ot_dist_subpath = os.path.join(ot_dist_path, str(j), 'ot_dists.json')
                with open(ot_dist_subpath, "r") as f:
                    ot_dist_data = json.load(f)
                ot_dist_data = np.array(ot_dist_data, dtype=np.float32)
                human_segment_idx = np.argmin(ot_dist_data)

                source_folder = os.path.join(cfg.data_path, cfg.cross_embodiment_segments, str(human_segment_idx))
                copy_images(source_folder, new_episode_folder)

        new_episode_folder = os.path.join(
            cfg.data_path, f"{cfg.pretrain_model_name}_{cfg.cross_embodiment_segments}_generated_tcc_{cfg.num_chops}_ckpt{cfg.ckpt}", folder_path
        )
        if cfg.tcc_lookup:
            os.makedirs(new_episode_folder, exist_ok=True)
            tcc_dist_path = osp.join(cfg.nearest_neighbor_data_dirs, f'{folder_path}')
            for j in range(cfg.num_chops):
                tcc_dist_subpath = os.path.join(tcc_dist_path, str(j), 'tcc_dists.json')
                with open(tcc_dist_subpath, "r") as f:
                    tcc_dist_data = json.load(f)
                tcc_dist_data = np.array(tcc_dist_data, dtype=np.float32)
                human_segment_idx = np.argmin(tcc_dist_data)

                source_folder = os.path.join(cfg.data_path, cfg.human_type, str(human_segment_idx))
                copy_images(source_folder, new_episode_folder)

    



if __name__ == "__main__":
    main()