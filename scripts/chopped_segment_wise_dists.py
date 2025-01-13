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
from xskill.utility.eval_utils import traj_representations, load_model, compute_tcc_loss, compute_optimal_transport_loss

def list_digit_folders(directory):
    # List all items in the directory
    items = os.listdir(directory)
    
    # Filter out only the folders whose names are composed of digits
    digit_folders = [item for item in items if os.path.isdir(os.path.join(directory, item)) and item.isdigit()]
    
    return digit_folders


@hydra.main(
    version_base=None,
    config_path="../config/simulation",
    config_name="segment_wise_dists",
)
def main(cfg: DictConfig):
    """
    Generates l2 distance matrix between each robot video and all human z's from the dataset.

    Parameters
    ----------
    cfg : DictConfig
        Specifies vision encoder to use.

    Side Effects
    ------------
    - Saves TCC loss matrices to the folders corresponding to each robot episode.
    - Saves Optimal Transport distance matrices to the folders corresponding to each robot episode. 

    Returns 
    -------
    None
    """
    model = load_model(cfg)
    model.eval()
    frame_sampler = hydra.utils.instantiate(cfg.frame_sampler)

    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )
    pipeline = nn.Sequential(Tr.CenterCrop((112, 112)), normalize)

    data_path = os.path.join(cfg.data_path, cfg.cross_embodiment_segments)
    all_folders = list_digit_folders(data_path)
    all_folders = sorted(all_folders, key=lambda x: int(x))
    human_vid_to_traj = []
    for folder_path in tqdm(all_folders, disable=not cfg.verbose):
        with torch.no_grad():
            traj_representation, _ = traj_representations(cfg, model, pipeline, cfg.cross_embodiment_segments, int(folder_path))
        eps_len = traj_representation.shape[0]
        snap_idx = frame_sampler._sample(list(range(eps_len)))
        snap_idx.sort()
        snap_idx = [snap_idx[i].item() for i in range(len(snap_idx))]
        traj_representation = traj_representation[snap_idx]
        
        human_vid_to_traj.append(traj_representation)
    

    data_path = os.path.join(cfg.data_path, 'robot')
    all_folders = list_digit_folders(data_path)
    all_folders = sorted(all_folders, key=lambda x: int(x))
    for folder_path in tqdm(all_folders, disable=not cfg.verbose):
        save_folder = os.path.join(
            cfg.exp_path, f"{cfg.cross_embodiment_segments}_l2_{cfg.num_chops}", f"ckpt_{cfg.ckpt}", folder_path
        )
        os.makedirs(save_folder, exist_ok=True)
        from collections import defaultdict
        tcc_dist_dict = defaultdict(list)
        ot_dist_dict = defaultdict(list)
        with torch.no_grad():
            traj_representation, _ = traj_representations(cfg, model, pipeline, 'robot', int(folder_path))

            # Calculate the length of each subarray
            subarray_length = len(traj_representation) // cfg.num_chops

            # Calculate the split points
            split_points = [subarray_length * i for i in range(1, cfg.num_chops)]

            # Split the array into subarrays
            subarrays = np.split(traj_representation, split_points)

            for j, sub_clip_rep in enumerate(subarrays):
                eps_len = len(sub_clip_rep)
                snap_idx = frame_sampler._sample(list(range(eps_len)))
                snap_idx.sort()
                snap_idx = [snap_idx[i].item() for i in range(len(snap_idx))]
                sub_clip_rep = sub_clip_rep[snap_idx]
                for human_traj_representation in human_vid_to_traj:
                    tcc_dist_dict[str(j)].append(compute_tcc_loss(sub_clip_rep.unsqueeze(0), human_traj_representation.unsqueeze(0)).item())
                    ot_dists = compute_optimal_transport_loss(sub_clip_rep.unsqueeze(0), human_traj_representation.unsqueeze(0))
                    ot_dist_dict[str(j)].append(ot_dists[0][0].item())
        
        for j in range(len(subarrays)):
            os.makedirs(os.path.join(save_folder, str(j)), exist_ok=True)
            with open(os.path.join(save_folder, str(j), 'tcc_dists.json'), 'w') as f:
                json.dump(tcc_dist_dict[str(j)], f)
            with open(os.path.join(save_folder, str(j), 'ot_dists.json'), 'w') as f:
                json.dump(ot_dist_dict[str(j)], f)

        
    


if __name__ == "__main__":
    main()
