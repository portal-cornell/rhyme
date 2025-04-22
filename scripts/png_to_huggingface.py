from datasets import Dataset, Features, Sequence, Image, Value, load_from_disk, load_dataset
import numpy as np
import os
import glob
from tqdm.auto import tqdm
from PIL import Image as PILImage
import re
import json


def create_hf_dataset(video_dirs, max_frames=None, batch_size=100, output_dir=None, include_actions=True, include_states=True, train_mask=None, eval_mask=None):
    """
    Process video directories into a HuggingFace dataset format in batches.
    
    Args:
        video_dirs: List of directories containing video subdirectories
        max_frames: Maximum number of frames per video (None for all frames)
        batch_size: Number of videos to process in each batch
        output_dir: Directory to save the processed dataset (None to skip saving)
        include_actions: Whether to include actions.json data if available
        include_states: Whether to include states.json data if available
    
    Returns:
        HuggingFace Dataset object
    """
    im = Image()
    # Create features for the dataset
    features = Features({
        "video_id": Value("string"),
        "frames": Sequence(Image()),
        "metadata": Value("string")
    })
    
    
    # Collect all video paths first
    all_videos = []
    for class_idx, video_dir in enumerate(video_dirs):
        video_subdirs = sorted(glob.glob(os.path.join(video_dir, "*")), key=lambda x: int(os.path.basename(x)))
        for video_path in video_subdirs:
            all_videos.append((class_idx, video_path))
    
    total_videos = len(all_videos)
    total_videos = 5
    print(f"Found {total_videos} videos total")
    all_batch_datasets = []
    # Process videos in batches
    
    for batch_start in range(0, total_videos, batch_size):
        batch_end = min(batch_start + batch_size, total_videos)
        print(f"Processing batch {batch_start//batch_size + 1}: videos {batch_start} to {batch_end-1}")
        
        batch_dict = {
            "video_id": [],
            "frames": [],
            "metadata": []
        }
        
        # Process each video in the current batch
        for i in tqdm(range(batch_start, batch_end), desc="Processing videos"):
            class_idx, video_path = all_videos[i]
            frames = sorted(glob.glob(os.path.join(video_path, "*.png")), 
                key=lambda x: int(re.search(r'(\d+)', os.path.basename(x)).group()))
            
            if max_frames is not None and len(frames) > max_frames:
                # Sample evenly if we need to limit frames
                indices = np.linspace(0, len(frames) - 1, max_frames, dtype=int)
                frames = [frames[i] for i in indices]
            
            # Convert frames to PIL.Image objects
            frame_arrays = []
            for frame_path in frames:
                try:
                    img = PILImage.open(frame_path)
                    frame_arrays.append(im.encode_example(value=np.array(img)))
                except Exception as e:
                    print(f"Error loading {frame_path}: {e}")
            
            if frame_arrays:
                path_components = video_path.split(os.sep)
                relative_path = os.path.join(path_components[-2], path_components[-1])
                
                metadata = {
                    "original_path": relative_path,
                    "num_frames": len(frame_arrays)
                }
                
                # Check for actions.json and include if available and requested
                if include_actions:
                    actions_path = os.path.join(video_path, "actions.json")
                    if os.path.exists(actions_path):
                        try:
                            with open(actions_path, "r") as f:
                                actions_data = json.load(f)
                            
                            # Add actions data to batch_dict if not already there
                            if "actions" not in batch_dict:
                                batch_dict["actions"] = []
                            
                            # Convert to numpy array and flatten for storage
                            actions_array = np.array(actions_data, dtype=np.float32)
                            batch_dict["actions"].append(actions_array)
                            
                            # Add metadata about actions
                            metadata["has_actions"] = True
                            metadata["actions_shape"] = list(actions_array.shape)
                        except Exception as e:
                            print(f"Error loading actions from {actions_path}: {e}")
                            metadata["has_actions"] = False
                            if "actions" in batch_dict:
                                batch_dict["actions"].append(None)
                    else:
                        metadata["has_actions"] = False
                        if "actions" in batch_dict:
                            batch_dict["actions"].append(None)
                
                # Check for states.json and include if available and requested
                if include_states:
                    states_path = os.path.join(video_path, "states.json")
                    if os.path.exists(states_path):
                        try:
                            with open(states_path, "r") as f:
                                states_data = json.load(f)
                            
                            # Add states data to batch_dict if not already there
                            if "states" not in batch_dict:
                                batch_dict["states"] = []
                            
                            # Convert to numpy array and flatten for storage
                            states_array = np.array(states_data, dtype=np.float32)
                            batch_dict["states"].append(states_array)
                            
                            # Add metadata about states
                            metadata["has_states"] = True
                            metadata["states_shape"] = list(states_array.shape)
                        except Exception as e:
                            print(f"Error loading states from {states_path}: {e}")
                            metadata["has_states"] = False
                            if "states" in batch_dict:
                                batch_dict["states"].append(None)
                    else:
                        metadata["has_states"] = False
                        if "states" in batch_dict:
                            batch_dict["states"].append(None)

                batch_dict["video_id"].append(os.path.basename(video_path))
                batch_dict["frames"].append(frame_arrays)
                batch_dict["metadata"].append(metadata)
        
        # Convert metadata to JSON strings
        batch_dict["metadata"] = [json.dumps(m) for m in batch_dict["metadata"]]
        
        # Create HuggingFace dataset for this batch
        batch_dataset = Dataset.from_dict(batch_dict)
        all_batch_datasets.append(batch_dataset)
        
        print(f"Batch {batch_start//batch_size + 1} appended")
    
    
    # Concatenate all batch datasets
    from datasets import concatenate_datasets
    full_dataset = concatenate_datasets(all_batch_datasets)
    print(f"Full dataset created with {len(full_dataset)} samples")
    
    # Save the dataset to disk if output_dir is provided
    if output_dir:
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        full_dataset.save_to_disk(output_dir)
        print(f"Dataset saved to {output_dir}")
    
    return full_dataset


def load_hf_dataset(dataset_dir):
    """
    Load a HuggingFace dataset from disk.
    
    Args:
        dataset_dir: Directory where the dataset was saved
    
    Returns:
        HuggingFace Dataset object
    """
    try:
        dataset = load_from_disk(dataset_dir)
        print(f"Successfully loaded dataset from {dataset_dir} with {len(dataset)} samples")
        return dataset
    except Exception as e:
        print(f"Error loading dataset from {dataset_dir}: {e}")
        return None
        
        
def view_sample_frames(dataset, sample_idx=0, output_dir=None, max_frames_to_save=None):
    """
    Extract and optionally save individual frames from a dataset sample.
    
    Args:
        dataset: HuggingFace Dataset object
        sample_idx: Index of the sample to view (default: 0)
        output_dir: Directory to save extracted frames (default: None, doesn't save)
        max_frames_to_save: Maximum number of frames to save (default: None, saves all)
    
    Returns:
        List of PIL.Image objects
    """
    if sample_idx >= len(dataset):
        print(f"Error: Sample index {sample_idx} out of range. Dataset has {len(dataset)} samples.")
        return []
    
    sample = dataset[sample_idx]
    video_id = sample["video_id"]
    frames_encoded = sample["frames"]
    metadata = json.loads(sample["metadata"])
    
    print(f"Sample {sample_idx} details:")
    print(f"  Video ID: {video_id}")
    print(f"  Number of frames: {len(frames_encoded)}")
    print(f"  Metadata: {metadata}")
    
    # Check for actions and states data
    if "actions" in sample:
        actions = sample["actions"]
        print(f"  Actions shape: {np.array(actions).shape}")
    
    if "states" in sample:
        states = sample["states"]
        print(f"  States shape: {np.array(states).shape}")
    
    # Initialize Image feature to decode the frames
    im = Image()
    
    # Decode frames to PIL.Image objects
    frames = []
    for i, frame_encoded in enumerate(frames_encoded):
        # Stop if we've reached the maximum number of frames to process
        if max_frames_to_save is not None and i >= max_frames_to_save:
            break
            
        frame_pil = im.decode_example(frame_encoded)
        frames.append(frame_pil)
        
        # Save the frame if output_dir is provided
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            frame_path = os.path.join(output_dir, f"{video_id}_frame_{i:04d}.png")
            frame_pil.save(frame_path)
            print(f"Saved frame {i} to {frame_path}")
    
    print(f"Extracted {len(frames)} frames from sample {sample_idx}")
    if output_dir:
        print(f"Frames saved to {output_dir}")
    
    return frames


# Example usage:
if __name__ == "__main__":
    # Create and save dataset
    video_dirs = ["/share/portal/pd337/rhyme/datasets/kitchen_dataset/robot"]
    dataset_output_dir = "/share/portal/pd337/rhyme/datasets/huggingface/robot"  # Specify your output directory
    train_mask = "/share/portal/pd337/rhyme/datasets/kitchen_dataset/train_mask.json"  # Specify your video mask file
    with open(train_mask, 'r') as f:
        train_mask = json.load(f)

    eval_mask = "/share/portal/pd337/rhyme/datasets/kitchen_dataset/eval_mask.json"  # Specify your video mask file
    with open(eval_mask, 'r') as f:
        eval_mask = json.load(f)
    
    # Create and save dataset with optional action and state data
    dataset = create_hf_dataset(
        video_dirs=video_dirs, 
        batch_size=5, 
        output_dir=dataset_output_dir,
        include_actions=True, 
        include_states=True,
        train_mask=train_mask,
        eval_mask=eval_mask
    )
    
    # Load dataset
    loaded_dataset = load_hf_dataset(dataset_output_dir)
    
    # Verify the loaded dataset
    if loaded_dataset is not None:
        print(f"Dataset loaded successfully with {len(loaded_dataset)} samples")
        print(f"First sample video ID: {loaded_dataset[0]['video_id']}")
        print(f"First sample has {len(loaded_dataset[0]['frames'])} frames")
        
        # Check if actions and states were loaded
        if "actions" in loaded_dataset.features:
            print(f"Dataset includes action data")
        
        if "states" in loaded_dataset.features:
            print(f"Dataset includes state data")
        
        # Extract and save frames from the first sample
        frames_output_dir = "/share/portal/pd337/rhyme/datasets/huggingface/robot/frames"  # Specify directory to save extracted frames
        frames = view_sample_frames(
            dataset=loaded_dataset,
            sample_idx=0,  # View the first sample
            output_dir=frames_output_dir,  # Save frames to disk
            max_frames_to_save=None  # Save only the first 10 frames (None for all frames)
        )
        
    # push to hub train and test splits