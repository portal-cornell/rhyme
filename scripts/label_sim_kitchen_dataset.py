# import numpy as np
# from PIL import Image
# import os
# from skimage.transform import resize
# from tqdm import tqdm
# from omegaconf import DictConfig
# import hydra
# import json
# import torch
# import torchvision.transforms as Tr
# from torchvision import transforms, datasets
# import torch.nn.functional as F
# from torch import nn, einsum
# import omegaconf
# from tqdm import tqdm
# import pandas as pd
# import seaborn as sns
# import cv2

# def repeat_last_proto(encode_protos, eps_len):
#     rep_proto = encode_protos[-1].unsqueeze(0).repeat(eps_len - len(encode_protos), 1)
#     return torch.cat([encode_protos, rep_proto])

# def load_images(folder_path, resize_shape=None):
#     images = []  # initialize an empty list to store the images

#     # get a sorted list of filenames in the folder
#     filenames = sorted(
#         [f for f in os.listdir(folder_path) if f.endswith(".png")],
#         key=lambda x: int(os.path.splitext(x)[0]),
#     )

#     # loop through all PNG files in the sorted list
#     for filename in filenames:
#         # open the image file using PIL library
#         img = Image.open(os.path.join(folder_path, filename))
#         # convert the image to a NumPy array
#         img_arr = np.array(img)
#         if resize_shape is not None:
#             img_arr = cv2.resize(img_arr, resize_shape)
#         images.append(img_arr)  # add the image array to the list

#     # convert the list of image arrays to a NumPy array
#     images_arr = np.array(images)
#     return images_arr


# def load_model(cfg):
#     exp_cfg = omegaconf.OmegaConf.load(os.path.join(cfg.exp_path, ".hydra/config.yaml"))
#     model = hydra.utils.instantiate(exp_cfg.Model).to(cfg.device)

#     loadpath = os.path.join(cfg.exp_path, f"epoch={cfg.ckpt}.ckpt")
#     checkpoint = torch.load(loadpath, map_location=cfg.device)

#     model.load_state_dict(checkpoint["state_dict"])
#     model.to(cfg.device)
#     model.eval()
#     print("model loaded")
#     return model


# def convert_images_to_tensors(images_arr, pipeline):
#     images_tensor = np.transpose(images_arr, (0, 3, 1, 2))  # (T,dim,h,w)
#     images_tensor = torch.tensor(images_tensor, dtype=torch.float32) / 255
#     images_tensor = pipeline(images_tensor)

#     return images_tensor


# @hydra.main(
#     version_base=None,
#     config_path="../config/simulation",
#     config_name="label_sim_kitchen_dataset",
# )
# def label_dataset(cfg: DictConfig):
#     model = load_model(cfg)

#     normalize = transforms.Normalize(
#         mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
#     )
#     pipeline = nn.Sequential(Tr.CenterCrop((112, 112)), normalize)

#     datasets = ["robot", cfg.cross_embodiment] if cfg.include_robot else [cfg.cross_embodiment]

#     for demo_type in datasets:
#         data_path = os.path.join(cfg.data_path, demo_type)
#         all_folders = os.listdir(data_path)
#         all_folders = sorted(all_folders, key=lambda x: int(x))
#         if cfg.plot_top_k is not None:
#             all_folders = all_folders[: cfg.plot_top_k]
#         for folder_path in tqdm(all_folders, disable=not cfg.verbose):
#             # store the proto in proto pretrain folder
#             if demo_type != 'robot':
#                 save_folder = os.path.join(
#                     cfg.exp_path, f"{demo_type}_encode_protos", f"ckpt_{cfg.ckpt}", folder_path
#                 )
#             else:
#                 save_folder = os.path.join(
#                     cfg.exp_path, "encode_protos", f"ckpt_{cfg.ckpt}", folder_path
#                 )
#             os.makedirs(save_folder, exist_ok=True)

#             data_folder = os.path.join(data_path, folder_path)

#             images_arr = load_images(data_folder, resize_shape=cfg.resize_shape)

#             images_tensor = convert_images_to_tensors(images_arr, pipeline).cuda()

#             eps_len = images_tensor.shape[0]
#             im_q = torch.stack(
#                 [
#                     images_tensor[j : j + model.slide + 1]
#                     for j in range(eps_len - model.slide)
#                 ]
#             )  # (b,slide+1,c,h,w)
#             state_representation = model.encoder_q.get_state_representation(im_q, None)
#             traj_representation = model.encoder_q.get_traj_representation(
#                 state_representation
#             )
#             traj_representation = repeat_last_proto(traj_representation, eps_len)
#             traj_representation = traj_representation.detach().cpu().numpy()
#             traj_representation = np.array(traj_representation).tolist()

#             with open(os.path.join(save_folder, "traj_representation.json"), "w") as f:
#                 json.dump(traj_representation, f)


# if __name__ == "__main__":
#     label_dataset()


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
from datasets import load_dataset, Image
import concurrent.futures

def repeat_last_proto(encode_protos, eps_len):
    rep_proto = encode_protos[-1].unsqueeze(0).repeat(eps_len - len(encode_protos), 1)
    return torch.cat([encode_protos, rep_proto])

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

def load_images_from_hf_dataset(sample, resize_shape=None, max_get_threads=4):
    """Load images from a Huggingface dataset sample"""
    # Create Image decoder for handling encoded frames
    image_decoder = Image()
    
    # Get encoded frames from the sample
    frames_encoded = sample["frames"]
    
    processed_frames = [None for _ in range(len(frames_encoded))]
    
    def process_image(image_index, frame_idx, frames_encoded, processed_frames, resize_shape):
        try:
            # Decode the encoded frame
            frame_pil = image_decoder.decode_example(frames_encoded[frame_idx])
            
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
            max_workers=max_get_threads) as executor:
        futures = set()
        for i in range(len(frames_encoded)):
            futures.add(
                executor.submit(process_image, i, i, frames_encoded, processed_frames, resize_shape))

        completed, futures = concurrent.futures.wait(futures)
        for f in completed:
            if not f.result():
                raise RuntimeError('Failed to process image!')
            
    sequence_data = np.stack(processed_frames)
    return sequence_data

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

    # Check if we should use Huggingface dataset
    if hasattr(cfg, 'use_huggingface') and cfg.use_huggingface:
        # Process Huggingface datasets
        datasets = ["robot", cfg.cross_embodiment] if cfg.include_robot else [cfg.cross_embodiment]
        
        for demo_type in datasets:
            # Determine dataset config to use
            config_name = "robot" if demo_type == "robot" else cfg.cross_embodiment.lower()
            
            # Load the dataset from Huggingface
            hf_dataset = load_dataset(cfg.dataset_name, config_name, split=cfg.split)
            print(f"Loaded {demo_type} dataset with {len(hf_dataset)} samples")
            
            # Process each sample in the dataset
            for sample in tqdm(hf_dataset, disable=not cfg.verbose):
                # Get the video ID
                video_id = sample["video_id"]
                
                # Create save folder
                if demo_type != 'robot':
                    save_folder = os.path.join(
                        cfg.exp_path, f"{demo_type}_encode_protos", f"ckpt_{cfg.ckpt}", str(video_id)
                    )
                else:
                    save_folder = os.path.join(
                        cfg.exp_path, "encode_protos", f"ckpt_{cfg.ckpt}", str(video_id)
                    )
                os.makedirs(save_folder, exist_ok=True)
                
                # Load images from the sample
                images_arr = load_images_from_hf_dataset(sample, resize_shape=cfg.resize_shape)
                
                # Convert images to tensors
                images_tensor = convert_images_to_tensors(images_arr, pipeline).cuda()

                eps_len = images_tensor.shape[0]
                
                # Skip if too few frames
                if eps_len <= model.slide:
                    print(f"Warning: Skipping sample with {eps_len} frames (need > {model.slide})")
                    continue
                
                im_q = torch.stack(
                    [images_tensor[j : j + model.slide + 1] for j in range(eps_len - model.slide)]
                )  # (b,slide+1,c,h,w)
                
                # Get state representations
                state_representation = model.encoder_q.get_state_representation(im_q, None)
                traj_representation = model.encoder_q.get_traj_representation(state_representation)
                traj_representation = repeat_last_proto(traj_representation, eps_len)
                traj_representation = traj_representation.detach().cpu().numpy()
                traj_representation = np.array(traj_representation).tolist()
                
                # Save trajectory representation
                with open(os.path.join(save_folder, "traj_representation.json"), "w") as f:
                    json.dump(traj_representation, f)
    else:
        # Original implementation for local datasets
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