from datasets import Dataset, Features, Sequence, Image, Value, load_from_disk, load_dataset
import numpy as np
import os
import glob
from tqdm.auto import tqdm
from PIL import Image as PILImage
import re
import json

def create_paired_hf_dataset(
    paired_dirs, 
    embodiment_names,  # New parameter for embodiment names
    max_frames=None, 
    batch_size=100, 
    output_dir=None, 
    train_mask=None, 
    eval_mask=None,
    percentage_pairing=1.0,
):
    """
    Process paired video directories into a HuggingFace dataset format in batches.
    
    Args:
        paired_dirs: Tuple of (dir1, dir2) where dir1 and dir2 contain corresponding videos
        embodiment_names: Tuple of (name1, name2) for the two embodiments (e.g., 'sphere-easy', 'robot')
        max_frames: Maximum number of frames per video (None for all frames)
        batch_size: Number of videos to process in each batch
        output_dir: Directory to save the processed dataset (None to skip saving)
        train_mask: Dictionary mapping video ids to boolean indicating if they're in training set
        eval_mask: Dictionary mapping video ids to boolean indicating if they're in evaluation set
        percentage_pairing: Percentage of videos to include in the paired dataset
    
    Returns:
        Dictionary containing 'train' and 'eval' HuggingFace Dataset objects
    """
    assert len(paired_dirs) == 2, "paired_dirs must contain exactly 2 directories"
    assert len(embodiment_names) == 2, "embodiment_names must contain exactly 2 names"
    
    # Extract embodiment names for readability
    embodiment1_name, embodiment2_name = embodiment_names
    im = Image()
    
    # Create features for the dataset with dynamic field names based on embodiment names
    features = Features({
        "video_id": Value("string"),
        f"frames_{embodiment1_name}": Sequence(Image()),
        f"frames_{embodiment2_name}": Sequence(Image()),
        "metadata": Value("string")
    })
    
    # Collect all paired video paths
    dir1, dir2 = paired_dirs
    videos_dir1 = sorted([p for p in glob.glob(os.path.join(dir1, "*")) if os.path.isdir(p)], 
                        key=lambda x: int(os.path.basename(x)))
    videos_dir2 = sorted([p for p in glob.glob(os.path.join(dir2, "*")) if os.path.isdir(p)], 
                        key=lambda x: int(os.path.basename(x)))
    
    # Ensure both directories have the same number of videos
    assert len(videos_dir1) == len(videos_dir2), f"Directories must have same number of videos: {len(videos_dir1)} vs {len(videos_dir2)}"
    
    # Apply percentage pairing if less than 100%
    if percentage_pairing < 1.0:
        total_videos = len(videos_dir1)
        num_to_keep = int(total_videos * percentage_pairing)
        indices = np.random.choice(total_videos, num_to_keep, replace=False)
        videos_dir1 = [videos_dir1[i] for i in indices]
        videos_dir2 = [videos_dir2[i] for i in indices]
    
    total_videos = len(videos_dir1)
    print(f"Found {total_videos} paired videos")
    
    # Prepare video pairs with IDs
    all_video_pairs = []
    for i in range(total_videos):
        video_path1 = videos_dir1[i]
        video_path2 = videos_dir2[i]
        video_id = os.path.basename(video_path1)
        all_video_pairs.append((video_path1, video_path2, int(video_id)))
    
    # Apply train and eval masks
    train_videos = []
    eval_videos = []
    if train_mask is not None and eval_mask is not None:
        for video_path1, video_path2, video_id in all_video_pairs:
            if train_mask[video_id]:
                train_videos.append((video_path1, video_path2, video_id))
            elif eval_mask[video_id]:
                eval_videos.append((video_path1, video_path2, video_id))
    else:
        # If no masks provided, use all videos for training
        train_videos = all_video_pairs
        
    print(f"Selected {len(train_videos)} video pairs for training")
    print(f"Selected {len(eval_videos)} video pairs for evaluation")
    
    # Process training videos
    train_dataset = process_paired_video_batch(
        train_videos, 
        embodiment_names,  # Pass embodiment names
        batch_size, 
        max_frames, 
    )
    
    # Process evaluation videos
    eval_dataset = process_paired_video_batch(
        eval_videos, 
        embodiment_names,  # Pass embodiment names
        batch_size, 
        max_frames, 
    )
    
    # Save the datasets to disk if output_dir is provided
    if output_dir:
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        train_output_dir = os.path.join(output_dir, "train")
        eval_output_dir = os.path.join(output_dir, "eval")
        
        train_dataset.save_to_disk(train_output_dir)
        print(f"Training dataset saved to {train_output_dir}")
        
        eval_dataset.save_to_disk(eval_output_dir)
        print(f"Evaluation dataset saved to {eval_output_dir}")
    
    return {"train": train_dataset, "eval": eval_dataset}


def process_paired_video_batch(video_pairs, embodiment_names, batch_size, max_frames):
    """
    Process a batch of paired videos into a HuggingFace dataset.
    
    Args:
        video_pairs: List of (video_path1, video_path2, video_id) tuples
        embodiment_names: Tuple of (name1, name2) for the two embodiments
        batch_size: Number of videos to process in each batch
        max_frames: Maximum number of frames per video
        
    Returns:
        HuggingFace Dataset object
    """
    im = Image()
    all_batch_datasets = []
    total_videos = len(video_pairs)
    
    # Extract embodiment names
    embodiment1_name, embodiment2_name = embodiment_names
    
    for batch_start in range(0, total_videos, batch_size):
        batch_end = min(batch_start + batch_size, total_videos)
        print(f"Processing batch {batch_start//batch_size + 1}: videos {batch_start} to {batch_end-1}")
        
        # Create batch dictionary with dynamic keys based on embodiment names
        batch_dict = {
            "video_id": [],
            f"frames_{embodiment1_name}": [],
            f"frames_{embodiment2_name}": [],
            "metadata": []
        }
        
        # Process each video pair in the current batch
        for i in tqdm(range(batch_start, batch_end), desc="Processing video pairs"):
            video_path1, video_path2, video_id = video_pairs[i]
            
            # Process frames from first directory
            frames1 = sorted(glob.glob(os.path.join(video_path1, "*.png")), 
                key=lambda x: int(re.search(r'(\d+)', os.path.basename(x)).group()))
            
            # Process frames from second directory
            frames2 = sorted(glob.glob(os.path.join(video_path2, "*.png")), 
                key=lambda x: int(re.search(r'(\d+)', os.path.basename(x)).group()))
            
            # Sample frames if max_frames is specified
            if max_frames is not None:
                if len(frames1) > max_frames:
                    indices = np.linspace(0, len(frames1) - 1, max_frames, dtype=int)
                    frames1 = [frames1[i] for i in indices]
                
                if len(frames2) > max_frames:
                    indices = np.linspace(0, len(frames2) - 1, max_frames, dtype=int)
                    frames2 = [frames2[i] for i in indices]
            
            # Convert frames to PIL.Image objects and possibly resize
            frame_arrays1 = []
            frame_arrays2 = []
            
            # Process frames from first directory
            for frame_path in frames1:
                try:
                    img = PILImage.open(frame_path)
                    frame_arrays1.append(im.encode_example(value=np.array(img)))
                except Exception as e:
                    print(f"Error loading {frame_path}: {e}")
            
            # Process frames from second directory
            for frame_path in frames2:
                try:
                    img = PILImage.open(frame_path)
                    frame_arrays2.append(im.encode_example(value=np.array(img)))
                except Exception as e:
                    print(f"Error loading {frame_path}: {e}")
            
            # Only add to batch if we have frames from both directories
            if frame_arrays1 and frame_arrays2:
                path_components1 = video_path1.split(os.sep)
                path_components2 = video_path2.split(os.sep)
                relative_path1 = os.path.join(path_components1[-2], path_components1[-1])
                relative_path2 = os.path.join(path_components2[-2], path_components2[-1])
                
                metadata = {
                    f"original_path_{embodiment1_name}": relative_path1,
                    f"original_path_{embodiment2_name}": relative_path2,
                    f"num_frames_{embodiment1_name}": len(frame_arrays1),
                    f"num_frames_{embodiment2_name}": len(frame_arrays2)
                }
                
                batch_dict["video_id"].append(str(video_id))
                batch_dict[f"frames_{embodiment1_name}"].append(frame_arrays1)
                batch_dict[f"frames_{embodiment2_name}"].append(frame_arrays2)
                batch_dict["metadata"].append(metadata)
        
        # Convert metadata to JSON strings
        batch_dict["metadata"] = [json.dumps(m) for m in batch_dict["metadata"]]
        
        # Create HuggingFace dataset for this batch
        batch_dataset = Dataset.from_dict(batch_dict)
        all_batch_datasets.append(batch_dataset)
        print(f"Batch {batch_start//batch_size + 1} appended")

    # Concatenate all batch datasets
    if all_batch_datasets:
        from datasets import concatenate_datasets
        full_dataset = concatenate_datasets(all_batch_datasets)
        print(f"Dataset created with {len(full_dataset)} samples")
        return full_dataset
    else:
        # Create empty dataset schema with proper column names
        empty_dict = {
            "video_id": [], 
            f"frames_{embodiment1_name}": [], 
            f"frames_{embodiment2_name}": [], 
            "metadata": []
        }
        return Dataset.from_dict(empty_dict)

# Update the load function to pass embodiment names
def load_hf_dataset(dataset_dir, split=None):
    """
    Load a HuggingFace dataset from disk.
    
    Args:
        dataset_dir: Directory where the dataset was saved
        split: Which split to load ('train', 'eval', or None for both)
    
    Returns:
        HuggingFace Dataset object or dict of Dataset objects
    """
    try:
        if split is None:
            # Load both splits
            train_path = os.path.join(dataset_dir, "train")
            eval_path = os.path.join(dataset_dir, "eval")
            
            train_dataset = load_from_disk(train_path)
            eval_dataset = load_from_disk(eval_path)
            
            print(f"Successfully loaded training dataset with {len(train_dataset)} samples")
            print(f"Successfully loaded evaluation dataset with {len(eval_dataset)} samples")
            
            return {"train": train_dataset, "eval": eval_dataset}
        else:
            # Load specific split
            split_path = os.path.join(dataset_dir, split)
            dataset = load_from_disk(split_path)
            print(f"Successfully loaded {split} dataset from {split_path} with {len(dataset)} samples")
            return dataset
    except Exception as e:
        print(f"Error loading dataset from {dataset_dir}: {e}")
        return None
    
def view_paired_sample_frames(dataset, embodiment_names, sample_idx=0, output_dir=None, max_frames_to_save=None):
    """
    Extract and optionally save individual frames from a paired dataset sample.
    
    Args:
        dataset: HuggingFace Dataset object (paired dataset)
        embodiment_names: Tuple of (name1, name2) for the two embodiments
        sample_idx: Index of the sample to view (default: 0)
        output_dir: Directory to save extracted frames (default: None, doesn't save)
        max_frames_to_save: Maximum number of frames to save (default: None, saves all)
    
    Returns:
        Tuple of (List of frames from first embodiment, List of frames from second embodiment)
    """
    if sample_idx >= len(dataset):
        print(f"Error: Sample index {sample_idx} out of range. Dataset has {len(dataset)} samples.")
        return [], []
    
    # Extract embodiment names
    embodiment1_name, embodiment2_name = embodiment_names
    
    sample = dataset[sample_idx]
    video_id = sample["video_id"]
    frames1_encoded = sample[f"frames_{embodiment1_name}"]
    frames2_encoded = sample[f"frames_{embodiment2_name}"]
    metadata = json.loads(sample["metadata"])
    
    print(f"Sample {sample_idx} details:")
    print(f"  Video ID: {video_id}")
    print(f"  Number of frames in {embodiment1_name}: {len(frames1_encoded)}")
    print(f"  Number of frames in {embodiment2_name}: {len(frames2_encoded)}")
    print(f"  Metadata: {metadata}")
    
    # Check for actions and states data for both embodiments
    for name in embodiment_names:
        if f"actions_{name}" in sample:
            actions = sample[f"actions_{name}"]
            print(f"  Actions_{name} shape: {np.array(actions).shape}")
        
        if f"states_{name}" in sample:
            states = sample[f"states_{name}"]
            print(f"  States_{name} shape: {np.array(states).shape}")
    
    # Initialize Image feature to decode the frames
    im = Image()
    
    # Decode frames from first embodiment
    frames1 = []
    for i, frame_encoded in enumerate(frames1_encoded):
        # Stop if we've reached the maximum number of frames to process
        if max_frames_to_save is not None and i >= max_frames_to_save:
            break
            
        frame_pil = im.decode_example(frame_encoded)
        frames1.append(frame_pil)
        
        # Save the frame if output_dir is provided
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            frame_path = os.path.join(output_dir, f"{video_id}_{embodiment1_name}_frame_{i:04d}.png")
            frame_pil.save(frame_path)
            print(f"Saved frame {i} from {embodiment1_name} to {frame_path}")
    
    # Decode frames from second embodiment
    frames2 = []
    for i, frame_encoded in enumerate(frames2_encoded):
        # Stop if we've reached the maximum number of frames to process
        if max_frames_to_save is not None and i >= max_frames_to_save:
            break
            
        frame_pil = im.decode_example(frame_encoded)
        frames2.append(frame_pil)
        
        # Save the frame if output_dir is provided
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            frame_path = os.path.join(output_dir, f"{video_id}_{embodiment2_name}_frame_{i:04d}.png")
            frame_pil.save(frame_path)
            print(f"Saved frame {i} from {embodiment2_name} to {frame_path}")
    
    print(f"Extracted {len(frames1)} frames from {embodiment1_name} and {len(frames2)} frames from {embodiment2_name}")
    if output_dir:
        print(f"Frames saved to {output_dir}")
    
    return frames1, frames2
    

if __name__ == "__main__":
    # Paired directories
    embodiment_pairs = {
        # "sphere-easy+robot": (
        #     "/share/portal/pd337/rhyme/datasets/kitchen_dataset/human_segments_paired_human",
        #     "/share/portal/pd337/rhyme/datasets/kitchen_dataset/robot_segments_paired_human",
        # ),
        # "sphere-medium+robot": (
        #     "/share/portal/pd337/rhyme/datasets/kitchen_dataset/singlehand_segments_paired_singlehand",
        #     "/share/portal/pd337/rhyme/datasets/kitchen_dataset/robot_segments_paired_singlehand",
        # ),
        "sphere-hard+robot": (
            "/share/portal/pd337/rhyme/datasets/kitchen_dataset/twohands_segments_paired_twohands",
            "/share/portal/pd337/rhyme/datasets/kitchen_dataset/robot_segments_paired_twohands",
        )
    }
    
    for pair_name, paired_dirs in embodiment_pairs.items():
        print(f'Processing paired embodiment: {pair_name}')
        
        # Extract embodiment names from pair_name by splitting by '+'
        embodiment_names = tuple(pair_name.split('+'))
        
        # Output directory
        dataset_output_dir = f"/share/portal/pd337/rhyme/datasets/huggingface/paired/{pair_name}"
        
        # Load train and eval masks (if available)
        train_mask_path = f"{paired_dirs[1]}/train_mask.json"
        
        train_mask = None
        eval_mask = None
        
        if os.path.exists(train_mask_path):
            with open(train_mask_path, 'r') as f:
                train_mask = json.load(f)
                eval_mask = [not v for v in train_mask]
        
        # Create and save paired dataset
        dataset_dict = create_paired_hf_dataset(
            paired_dirs=paired_dirs,
            embodiment_names=embodiment_names,  # Pass embodiment names 
            batch_size=5, 
            output_dir=dataset_output_dir,
            train_mask=train_mask,
            eval_mask=eval_mask,
        )
        
        # Load paired datasets
        train_dataset = load_hf_dataset(dataset_output_dir, split="train")
        eval_dataset = load_hf_dataset(dataset_output_dir, split="eval")
        
        # Verify the loaded datasets
        print(f"Training dataset loaded with {len(train_dataset)} samples")
        print(f"Evaluation dataset loaded with {len(eval_dataset)} samples")
        
        # Create a DatasetDict with train and eval splits
        from datasets import DatasetDict
        combined_dataset = DatasetDict({
            "train": train_dataset,
            "eval": eval_dataset
        })
        
        # Push to hub (optional)
        combined_dataset.push_to_hub(f"prithwishdan/RHyME", config_name=f"{pair_name}_paired")
        
        # View a sample from the training dataset
        # if len(train_dataset) > 0:
        #     frames_output_dir = os.path.join(dataset_output_dir, "sample_frames")
        #     frames1, frames2 = view_paired_sample_frames(
        #         dataset=train_dataset,
        #         embodiment_names=embodiment_names,  # Pass embodiment names
        #         sample_idx=0,  # View the first sample
        #         output_dir=frames_output_dir,  # Save frames to disk
        #         max_frames_to_save=5  # Save only the first 5 frames
        #     )