import numpy as np
import os
import cv2
import random
import concurrent.futures
from tqdm import tqdm
from omegaconf import DictConfig
import hydra
import json
import torch
import torchvision.transforms as Tr
from torchvision import transforms
from torch import nn
from datasets import load_dataset, Image
from rhyme.utility.eval_utils import load_model, compute_tcc_loss, compute_optimal_transport_loss, repeat_last_proto

# Shared image processing functions
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

def process_frames(frames_encoded, ctx_idxs, image_decoder, resize_shape=None, max_threads=4):
    """Process multiple frames in parallel."""
    frame_indices = [idx.item() for idx in ctx_idxs]
    processed_frames = [None for _ in range(len(frame_indices))]
    
    def process_single_frame(i, frame_idx):
        processed_frames[i] = decode_and_process_image(
            frames_encoded[frame_idx], image_decoder, resize_shape)
        return processed_frames[i] is not None
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_threads) as executor:
        futures = set()
        for i, frame_idx in enumerate(frame_indices):
            futures.add(executor.submit(process_single_frame, i, frame_idx))

        completed, _ = concurrent.futures.wait(futures)
        for f in completed:
            if not f.result():
                raise RuntimeError('Failed to process image!')
    
    sequence_data = np.stack(processed_frames)  # Shape: (T, H, W, C)
    return sequence_data

def transform_sequence(sequence_data):
    """Transform numpy sequence data to normalized tensor format."""
    sequence_data = np.transpose(sequence_data, (0, 3, 1, 2)).astype(np.float32)  # (T,C,H,W)
    sequence_data = sequence_data / 255.0
    return sequence_data

def get_frames_from_dataset(dataset, idx, image_decoder, resize_shape=None):
    """Extract frames from a dataset sample."""
    sample = dataset[idx]
    frames = sample["frames"]
    ctx_idxs = np.array(list(range(len(frames))))
    
    # Process frames
    sequence_data = process_frames(frames, ctx_idxs, image_decoder, resize_shape)
    
    # Transform the frames
    transformed_data = transform_sequence(sequence_data)
    
    return torch.Tensor(transformed_data)

def get_embodiment_frames_from_paired_dataset(dataset, idx, frame_sampler, image_decoder, embodiment_name, resize_shape=None):
    """Extract frames for a specific embodiment from a paired dataset sample."""
    sample = dataset[idx]
    frames = sample[f"frames_{embodiment_name}"]
    frame_idxs = frame_sampler._sample(frames)
    ctx_idxs = frame_sampler._get_context_steps(frame_idxs, len(frames))
    
    # Process frames
    sequence_data = process_frames(frames, ctx_idxs, image_decoder, resize_shape)
    
    # Transform the frames
    transformed_data = transform_sequence(sequence_data)
    
    return torch.Tensor(transformed_data)

@hydra.main(
    version_base=None,
    config_path="../config/simulation",
    config_name="segment_wise_dists_hf",
)
def main(cfg: DictConfig):
    """
    Generates distance matrices between robot videos and embodiment videos from the dataset.
    """
    random.seed(20)
    np.random.seed(20)
    model = load_model(cfg)
    model.eval()
    frame_sampler = hydra.utils.instantiate(cfg.frame_sampler)
    
    # Create image decoder once for all processing
    image_decoder = Image()

    # Setup normalization pipeline for model input
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )
    pipeline = nn.Sequential(Tr.CenterCrop((112, 112)), normalize)
    
    target_embodiment_name = cfg.cross_embodiment

    # Load paired dataset
    paired_config_name = f"{target_embodiment_name}+robot_paired"
    paired_dataset = load_dataset('prithwishdan/RHyME', paired_config_name, split='train')
    print(f"Loaded paired dataset with {len(paired_dataset)} samples")
    
    # Get representations for all target embodiment samples
    target_embodiment_reps = []
    for i in tqdm(range(len(paired_dataset)), desc=f"Processing {target_embodiment_name} samples"):
        with torch.no_grad():
            # Get frames for this embodiment
            frames = get_embodiment_frames_from_paired_dataset(
                paired_dataset, i, frame_sampler, image_decoder, 
                target_embodiment_name, cfg.resize_shape if hasattr(cfg, 'resize_shape') else None
            )
            
            # Get representation through model
            frames = frames.to(cfg.device)
            images_tensor = pipeline(frames)
            eps_len = images_tensor.shape[0]
            im_q = torch.stack(
                [
                    images_tensor[j : j + model.slide + 1]
                    for j in range(eps_len - model.slide)
                ]
            )  # (B,slide+1,C,H,W)
            state_rep = model.encoder_q.get_state_representation(im_q, None) # (T, 2, D)
            traj_rep = model.encoder_q.get_traj_representation(state_rep) # (T, D)
            traj_rep = repeat_last_proto(traj_rep, eps_len)
            
            target_embodiment_reps.append(traj_rep)
    
    random_indices = np.random.choice(len(target_embodiment_reps), size=500, replace=False)
    target_embodiment_reps = [target_embodiment_reps[i] for i in random_indices]
    
    # Save random indices to JSON file
    indices_save_path = os.path.join(
        cfg.exp_path, f"{target_embodiment_name}_distdata_{cfg.num_chops}",
        "lookup_indices.json"
    )
    os.makedirs(os.path.dirname(indices_save_path), exist_ok=True)
    with open(indices_save_path, 'w') as f:
        json.dump(random_indices.tolist(), f)
    
    # Load robot dataset
    robot_dataset = load_dataset('prithwishdan/RHyME', 'robot', split='train')
    print(f"Loaded robot dataset with {len(robot_dataset)} samples")
    
    # Process each robot sample
    for robot_idx in tqdm(range(len(robot_dataset)), desc="Processing robot samples"):
        # Create save folder
        save_folder = os.path.join(
            cfg.exp_path, f"{target_embodiment_name}_distdata_{cfg.num_chops}", 
            f"ckpt_{cfg.ckpt}", str(robot_idx)
        )
        os.makedirs(save_folder, exist_ok=True)
        
        # Initialize dictionaries for storing distances
        tcc_dist_dict = {str(j): [] for j in range(cfg.num_chops)}
        ot_dist_dict = {str(j): [] for j in range(cfg.num_chops)}
        
        with torch.no_grad():
            # Get robot frames
            frames = get_frames_from_dataset(
                robot_dataset, robot_idx, image_decoder, 
                cfg.resize_shape if hasattr(cfg, 'resize_shape') else None
            )
            
            # Get representation through model
            frames = frames.to(cfg.device)
            images_tensor = pipeline(frames)
            eps_len = images_tensor.shape[0]
            im_q = torch.stack(
                [
                    images_tensor[j : j + model.slide + 1]
                    for j in range(eps_len - model.slide)
                ]
            )  # (B,slide+1,C,H,W)
            state_rep = model.encoder_q.get_state_representation(im_q, None) # (T, 2, D)
            traj_rep = model.encoder_q.get_traj_representation(state_rep) # (T, D)
            robot_rep = repeat_last_proto(traj_rep, eps_len)
            
            # Split into chunks
            subarray_length = len(robot_rep) // cfg.num_chops
            split_points = [subarray_length * i for i in range(1, cfg.num_chops)]
            subarrays = np.split(robot_rep, split_points)
            
            # Process each chunk
            for j, sub_clip_rep in enumerate(subarrays):
                eps_len = len(sub_clip_rep)
                snap_idx = frame_sampler._sample(list(range(eps_len)))
                snap_idx.sort()
                snap_idx = [snap_idx[i].item() for i in range(len(snap_idx))]
                sub_clip_rep = sub_clip_rep[snap_idx]
                
                # Compare with each target embodiment
                for target_rep in target_embodiment_reps:
                    # TCC loss
                    tcc_loss = compute_tcc_loss(
                        sub_clip_rep.unsqueeze(0), target_rep.unsqueeze(0)
                    ).item()
                    tcc_dist_dict[str(j)].append(tcc_loss)
                    
                    # Optimal transport loss
                    ot_dists = compute_optimal_transport_loss(
                        sub_clip_rep.unsqueeze(0), target_rep.unsqueeze(0)
                    )
                    ot_dist_dict[str(j)].append(ot_dists[0][0].item())
        
        # Save results for each chunk
        for j in range(cfg.num_chops):
            chunk_dir = os.path.join(save_folder, str(j))
            os.makedirs(chunk_dir, exist_ok=True)
            
            # Save TCC distances
            with open(os.path.join(chunk_dir, 'tcc_dists.json'), 'w') as f:
                json.dump(tcc_dist_dict[str(j)], f)
            
            # Save OT distances
            with open(os.path.join(chunk_dir, 'ot_dists.json'), 'w') as f:
                json.dump(ot_dist_dict[str(j)], f)

if __name__ == "__main__":
    main()