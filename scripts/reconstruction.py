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
from rhyme.utility.eval_utils import gif_of_clip, traj_representations, load_model
from datasets import load_dataset, Image as HFImage
import concurrent.futures


# Reuse the image processing functions from your first script
def decode_and_process_image(frame_encoded, image_decoder, resize_shape=None):
    """Decode and process a single encoded frame."""
    try:
        # Decode the encoded frame
        frame_pil = image_decoder.decode_example(frame_encoded)
        
        # Convert PIL image to numpy array
        frame_np = np.array(frame_pil)
        
        # Resize if needed
        if resize_shape is not None:
            frame_np = cv2.resize(frame_np, tuple(resize_shape))
        
        return frame_np
    except Exception as e:
        print(f"Error processing image: {e}")
        return None


def process_frames_from_dataset(dataset, idx, image_decoder, target_embodiment_name=None, resize_shape=None):
    """Extract frames from a dataset sample and save them to a directory."""
    sample = dataset[idx]
    
    # Determine which frames to get
    if target_embodiment_name:
        frames = sample[f"frames_{target_embodiment_name}"]
    else:
        frames = sample["frames"]
    
    # Process each frame
    processed_frames = []
    for frame_encoded in frames:
        frame_np = decode_and_process_image(frame_encoded, image_decoder, resize_shape)
        if frame_np is not None:
            processed_frames.append(frame_np)
    
    return processed_frames


def save_frames_to_folder(frames, output_folder):
    """Save a list of numpy frames to a folder as PNG files."""
    os.makedirs(output_folder, exist_ok=True)
    
    for i, frame in enumerate(frames):
        frame_pil = Image.fromarray(frame)
        filename = os.path.join(output_folder, f"{i}.png")
        frame_pil.save(filename)


def list_digit_folders(directory):
    # List all items in the directory
    items = os.listdir(directory)
    
    # Filter out only the folders whose names are composed of digits
    digit_folders = [item for item in items if os.path.isdir(os.path.join(directory, item)) and item.isdigit()]
    
    return digit_folders


@hydra.main(
    version_base=None,
    config_path="../config/simulation",
    config_name="segment_wise_dists_hf",  # Changed to match your new config
)
def main(cfg: DictConfig):
    """
    Reconstructs videos using nearest neighbors based on TCC or OT distances.
    Updated to work with HuggingFace datasets.


    Parameters
    ----------
    cfg : DictConfig
        Specifies configuration details associated with the visual encoder being tested.
        - Note that exp_path must be the path where distance matrices were computed
        - correct_thresholds specifies the range of frames to consider for TCC accuracy

    Side Effects
    ------------
    - Creates reconstructed videos using nearest neighbor lookup
    - Saves frames from HuggingFace dataset to local folders

    Returns 
    -------
    None
    """
    # Create image decoder for processing HuggingFace data
    image_decoder = HFImage()
    
    # Load the lookup indices from the first script
    target_embodiment_name = cfg.cross_embodiment
    lookup_indices_path = os.path.join(
        cfg.exp_path, f"{target_embodiment_name}_distdata_{cfg.num_chops}",
        "lookup_indices.json"
    )
    
    if not os.path.exists(lookup_indices_path):
        raise FileNotFoundError(f"Lookup indices file not found at {lookup_indices_path}")
    
    with open(lookup_indices_path, 'r') as f:
        lookup_indices = np.array(json.load(f))
        lookup_indices = [int(idx) for idx in lookup_indices]
    
    print(f"Loaded {len(lookup_indices)} lookup indices")
    
    # Load the target embodiment dataset
    paired_config_name = f"{target_embodiment_name}+robot_paired"
    target_dataset = load_dataset('prithwishdan/RHyME', paired_config_name, split='train')
    print(f"Loaded target dataset with {len(target_dataset)} samples")
    
    # Get the path to distance matrices (from your first script)
    dist_data_path = os.path.join(
        cfg.exp_path, f"{target_embodiment_name}_distdata_{cfg.num_chops}",
        f"ckpt_{cfg.ckpt}"
    )
    
    if not os.path.exists(dist_data_path):
        raise FileNotFoundError(f"Distance data not found at {dist_data_path}")
    
    # Get all robot folders (assuming they're named with digit folders)
    all_folders = list_digit_folders(dist_data_path)
    all_folders = sorted(all_folders, key=lambda x: int(x))
    
    for folder_path in tqdm(all_folders, disable=not cfg.verbose, desc="Processing robot episodes"):
        # Create output folders for reconstructed videos
        if cfg.ot_lookup:
            new_episode_folder = os.path.join(
                cfg.data_path, f"{cfg.pretrain_model_name}_{target_embodiment_name}_generated_ot_{cfg.num_chops}_ckpt{cfg.ckpt}", folder_path
            )
            os.makedirs(new_episode_folder, exist_ok=True)
            
            # Collect frames from all chunks
            all_frames = []
            
            # Process each chunk
            for j in range(cfg.num_chops):
                ot_dist_subpath = os.path.join(dist_data_path, folder_path, str(j), 'ot_dists.json')
                
                if not os.path.exists(ot_dist_subpath):
                    print(f"Warning: OT distances not found at {ot_dist_subpath}")
                    continue
                
                with open(ot_dist_subpath, "r") as f:
                    ot_dist_data = json.load(f)
                ot_dist_data = np.array(ot_dist_data, dtype=np.float32)
                
                # Find the closest target segment
                closest_idx_in_subset = np.argmin(ot_dist_data)
                # Map back to actual dataset index
                actual_target_idx = lookup_indices[closest_idx_in_subset]
                
                # Get frames from the target dataset
                frames = process_frames_from_dataset(
                    target_dataset, 
                    actual_target_idx, 
                    image_decoder, 
                    target_embodiment_name,
                    resize_shape=cfg.resize_shape if hasattr(cfg, 'resize_shape') else None
                )
                
                # Add frames to the collection
                all_frames.extend(frames)
            
            # Save all frames to folder
            save_frames_to_folder(all_frames, new_episode_folder)
        
        if cfg.tcc_lookup:
            new_episode_folder = os.path.join(
                cfg.data_path, f"{cfg.pretrain_model_name}_{target_embodiment_name}_generated_tcc_{cfg.num_chops}_ckpt{cfg.ckpt}", folder_path
            )
            os.makedirs(new_episode_folder, exist_ok=True)
            
            # Collect frames from all chunks
            all_frames = []
            
            # Process each chunk
            for j in range(cfg.num_chops):
                tcc_dist_subpath = os.path.join(dist_data_path, folder_path, str(j), 'tcc_dists.json')
                
                if not os.path.exists(tcc_dist_subpath):
                    print(f"Warning: TCC distances not found at {tcc_dist_subpath}")
                    continue
                
                with open(tcc_dist_subpath, "r") as f:
                    tcc_dist_data = json.load(f)
                tcc_dist_data = np.array(tcc_dist_data, dtype=np.float32)
                
                # Find the closest target segment
                closest_idx_in_subset = np.argmin(tcc_dist_data)
                # Map back to actual dataset index
                actual_target_idx = lookup_indices[closest_idx_in_subset]
                
                # Get frames from the target dataset
                frames = process_frames_from_dataset(
                    target_dataset, 
                    actual_target_idx, 
                    image_decoder, 
                    target_embodiment_name,
                    resize_shape=cfg.resize_shape if hasattr(cfg, 'resize_shape') else None
                )
                
                # Add frames to the collection
                all_frames.extend(frames)
            
            # Save all frames to folder
            save_frames_to_folder(all_frames, new_episode_folder)


if __name__ == "__main__":
    main()