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
        train_mask: Dictionary mapping video ids to boolean indicating if they're in training set
        eval_mask: Dictionary mapping video ids to boolean indicating if they're in evaluation set
    
    Returns:
        Dictionary containing 'train' and 'eval' HuggingFace Dataset objects
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
            video_id = os.path.basename(video_path)
            all_videos.append((class_idx, video_path, int(video_id)))

    total_videos = len(all_videos)
    print(f"Found {total_videos} videos total")
    
    # Apply train and eval masks
    train_videos = []
    eval_videos = []
    if train_mask is not None and eval_mask is not None:
        for class_idx, video_path, video_id in all_videos:
            if train_mask[video_id]:
                train_videos.append((class_idx, video_path, video_id))
            elif eval_mask[video_id]:
                eval_videos.append((class_idx, video_path, video_id))
    else:
        # If no masks provided, use all videos for training
        train_videos = all_videos
        
    print(f"Selected {len(train_videos)} videos for training")
    print(f"Selected {len(eval_videos)} videos for evaluation")
    
    # Process training videos
    train_dataset = process_video_batch(train_videos, batch_size, max_frames, include_actions, include_states)
    
    # Process evaluation videos
    eval_dataset = process_video_batch(eval_videos, batch_size, max_frames, include_actions, include_states)
    
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


def process_video_batch(videos, batch_size, max_frames, include_actions, include_states):
    """
    Process a batch of videos into a HuggingFace dataset.
    
    Args:
        videos: List of (class_idx, video_path, video_id) tuples
        batch_size: Number of videos to process in each batch
        max_frames: Maximum number of frames per video
        include_actions: Whether to include actions data
        include_states: Whether to include states data
        
    Returns:
        HuggingFace Dataset object
    """
    im = Image()
    all_batch_datasets = []
    total_videos = len(videos)
    
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
            class_idx, video_path, video_id = videos[i]
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

                batch_dict["video_id"].append(video_id)
                batch_dict["frames"].append(frame_arrays)
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
        # Return empty dataset if no videos were processed
        return Dataset.from_dict({"video_id": [], "frames": [], "metadata": []})


# Also update the load function to handle train/eval splits
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
    embodiment_map = {
            "human": "sphere-easy",
            "singlehand": "sphere-medium",
            "twohands": "sphere-hard",
            "robot": "robot"
        }
    for embodiment_type in embodiment_map.keys():
        print('Embodiment type:', embodiment_type)
        # Create and save dataset
        video_dirs = [f"/share/portal/pd337/rhyme/datasets/kitchen_dataset/{embodiment_type}"]
        dataset_output_dir = f"/share/portal/pd337/rhyme/datasets/huggingface/{embodiment_type}"  # Specify your output directory
        
        # Load train and eval masks
        train_mask_path = "/share/portal/pd337/rhyme/datasets/kitchen_dataset/train_mask.json"
        eval_mask_path = "/share/portal/pd337/rhyme/datasets/kitchen_dataset/eval_mask.json"
        
        with open(train_mask_path, 'r') as f:
            train_mask = json.load(f)

        with open(eval_mask_path, 'r') as f:
            eval_mask = json.load(f)
        
        # Create and save dataset with train/eval splits
        dataset_dict = create_hf_dataset(
            video_dirs=video_dirs, 
            batch_size=5, 
            output_dir=dataset_output_dir,
            include_actions=True, 
            include_states=True,
            train_mask=train_mask,
            eval_mask=eval_mask
        )
        
        # Load datasets
        loaded_datasets = load_hf_dataset(dataset_output_dir)
        
        # Verify the loaded datasets
        if loaded_datasets is not None:
            # Check training dataset
            train_dataset = loaded_datasets["train"]
            print(f"Training dataset loaded successfully with {len(train_dataset)} samples")
            
            if len(train_dataset) > 0:
                print(f"First training sample video ID: {train_dataset[0]['video_id']}")
                print(f"First training sample has {len(train_dataset[0]['frames'])} frames")
            
            # Check evaluation dataset
            eval_dataset = loaded_datasets["eval"]
            print(f"Evaluation dataset loaded successfully with {len(eval_dataset)} samples")
            
            if len(eval_dataset) > 0:
                print(f"First evaluation sample video ID: {eval_dataset[0]['video_id']}")
                print(f"First evaluation sample has {len(eval_dataset[0]['frames'])} frames")
            
            # Check if actions and states were loaded in training dataset
            if "actions" in train_dataset.features:
                print(f"Training dataset includes action data")
            
            if "states" in train_dataset.features:
                print(f"Training dataset includes state data")


            # Create a combined DatasetDict to properly use train and eval as splits
            from datasets import DatasetDict
            
            # Create a DatasetDict with train and eval splits
            combined_dataset = DatasetDict({
                "train": train_dataset,
                "eval": eval_dataset
            })
            
            
            # Push to hub as a single dataset with train and eval splits
            combined_dataset.push_to_hub("prithwishdan/RHyME", config_name=embodiment_map[embodiment_type])
            
            # Extract and save frames from the first training sample
            # if len(train_dataset) > 0:
            #     frames_output_dir = os.path.join(dataset_output_dir, "train_frames")
            #     frames = view_sample_frames(
            #         dataset=train_dataset,
            #         sample_idx=0,  # View the first sample
            #         output_dir=frames_output_dir,  # Save frames to disk
            #         max_frames_to_save=5  # Save only the first 5 frames
            #     )

