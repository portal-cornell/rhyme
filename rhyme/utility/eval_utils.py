from PIL import Image
import numpy as np
import cv2
import torch
import os
import json
from omegaconf import DictConfig
import hydra
import omegaconf

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


def convert_images_to_tensors(images_arr, pipeline=None):
    images_tensor = np.transpose(images_arr, (0, 3, 1, 2))  # (T,dim,h,w)
    images_tensor = torch.tensor(images_tensor, dtype=torch.float32) / 255
    if pipeline is not None:
        images_tensor = pipeline(images_tensor)

    return images_tensor

def load_model(cfg):
    """
    Loads pretrained model from XSkill's vision encoder.


    Parameters
    ----------
    cfg : DictConfig
        Determines model path and checkpoint to be loaded.

    Returns 
    -------
    model : rhyme.model.core.Model
        Vision encoder from XSkill
    """
    exp_cfg = omegaconf.OmegaConf.load(os.path.join(cfg.exp_path, ".hydra/config.yaml"))
    model = hydra.utils.instantiate(exp_cfg.Model).to(cfg.device)

    loadpath = os.path.join(cfg.exp_path, f"epoch={cfg.ckpt}.ckpt")
    checkpoint = torch.load(loadpath, map_location=cfg.device)

    model.load_state_dict(checkpoint["state_dict"])
    model.to(cfg.device)
    model.eval()
    print("model loaded")  
    return model

def gif_of_clip(cfg, demo_type, ep_num, frame_num, slide, output_dir, cycle=False, save=True):
    """
    Computes latent representations of a video from embodiment type {demo_type}
    and episode number {ep_num} based on a provided model.


    Parameters
    ----------
    cfg : DictConfig
        Configures path to data and image details
    demo_type : str
        'robot' or 'human'
    ep_num : int
        Episode number
    frame_num : int 
        Start frame for video clip
    slide : int 
        Duration of clip
    output_dir : str
        Directory to save gif
    cycle : bool (optional)
        True => retrieved from cycle-back operation
    save : bool (optional)
        True => save images to output_dir

    Side Effects
    ------------
    - Saves gif to specified output_dir if save True

    Returns 
    -------
    pil_imgs : Image list
        Images from the specified clip
    """
    data_path = os.path.join(cfg.data_path, demo_type)
    data_folder = os.path.join(data_path, f'{ep_num}')

    images_arr = load_images(data_folder, resize_shape=cfg.resize_shape)

    sub_imgs = images_arr[frame_num:frame_num + slide + 1]

    pil_imgs = [Image.fromarray(img) for img in sub_imgs]
    if save:
        pil_imgs[0].save(os.path.join(output_dir, f'{demo_type}_{ep_num}_index{frame_num}{"_cycle" if cycle else ""}.gif'), save_all=True, append_images=pil_imgs[1:], duration=200, loop=0)
    
    return pil_imgs

def repeat_last_proto(encode_protos, eps_len):
    rep_proto = encode_protos[-1].unsqueeze(0).repeat(eps_len - len(encode_protos), 1)
    return torch.cat([encode_protos, rep_proto])

def traj_representations(cfg, model, pipeline, demo_type, ep_num, frame_list=None):
    """
    Computes latent representations of a video from embodiment type {demo_type}
    and episode number {ep_num} based on a provided model.


    Parameters
    ----------
    cfg : DictConfig
        Configures path to data and image details
    model : rhyme.model.core.Model
        Pretrained vision encoder model
    pipeline : torch.nn.modules.container.Sequential
        Image preprocessing pipeline
    demo_type : str
        'robot' or 'human'
    ep_num : int
        Episode number
    frame_list : int list (optional)
        Subset of frames to extract clips from
    

    Returns 
    -------
    traj_rep : torch.Tensor
        Tensor of shape (T, D=256), representing a sequence of latents 
        in the skill representation space
    z : torch.Tensor
        Tensor of shape (T, K), representing a distribution across prototypes
    """
    data_path = os.path.join(cfg.data_path, demo_type)
    data_folder = os.path.join(data_path, f'{ep_num}')

    images_arr = load_images(data_folder, resize_shape=cfg.resize_shape)

    images_tensor = convert_images_to_tensors(images_arr, pipeline).cuda() # (T, C, H, W)

    eps_len = images_tensor.shape[0]
    im_q = torch.stack(
        [
            images_tensor[j : j + model.slide + 1]
            for j in range(eps_len - model.slide)
        ]
    )  # (B,slide+1,C,H,W)

    clips = im_q
    if frame_list is not None:
        clips = clips[frame_list]

    z = model.encoder_q(im_q, None) # (T, K)
    state_rep = model.encoder_q.get_state_representation(clips, None) # (T, 2, D)
    traj_rep = model.encoder_q.get_traj_representation(state_rep) # (T, D)
    traj_rep = repeat_last_proto(traj_rep, eps_len)
    return traj_rep, z


@torch.no_grad()
def distributed_sinkhorn(out):
    Q = torch.exp(out / .05).t(
    )  # Q is K-by-B for consistency with notations from our paper
    B = Q.shape[1]  # number of samples to assign
    K = Q.shape[0]  # how many prototypes

    # make the matrix sums to 1
    sum_Q = torch.sum(Q)
    Q /= sum_Q

    for it in range(3):
        # normalize each row: total weight per prototype must be 1/K
        sum_of_rows = torch.sum(Q, dim=1, keepdim=True)
        Q /= sum_of_rows
        Q /= K

        # normalize each column: total weight per sample must be 1/B
        Q /= torch.sum(Q, dim=0, keepdim=True)
        Q /= B

    Q *= B  # the colomns must sum to 1 so that Q is an assignment
    return Q.t()

def batch_cosine_distance(x, y):
    C = torch.bmm(x, y.transpose(1, 2))
    x_norm = torch.norm(x, p=2, dim=2)
    y_norm = torch.norm(y, p=2, dim=2)
    x_n = x_norm.unsqueeze(2)
    y_n = y_norm.unsqueeze(2)
    norms = torch.bmm(x_n, y_n.transpose(1, 2))
    C = (1 - C / norms)
    return C

def tcc_loss_(emb1, emb2):
    """Compute the TCC loss between a pair of sequences."""
    similarity = -torch.cdist(emb1, emb2, p=2)/0.1
    beta = F.softmax(similarity, dim=-1)
    beta_ret = beta
    half_cycle = torch.bmm(beta, emb2)
    similarity = -torch.cdist(half_cycle, emb1, p=2)/0.1
    beta = F.softmax(similarity, dim=-1)
    beta_ret2 = beta
    cycle_back = torch.bmm(beta, emb1)

    # Compute L2 distance
    distances = torch.sqrt(torch.sum((cycle_back.squeeze() - emb1.squeeze())**2, dim=1))

    # Argsort the distances
    sorted_indices = torch.argsort(distances, descending=True)

    human_frames = torch.argmax(beta_ret[0], dim=1)
    cycled_frames = torch.argmax(beta_ret2[0], dim=1)
    dists_t = torch.abs(torch.Tensor(np.arange(len(cycled_frames))).to('cuda') - cycled_frames)
    dists = dists_t.argsort(descending=True)
    print(dists_t[dists[0]])
    return F.mse_loss(cycle_back, emb1), distances, sorted_indices, np.arange(len(cycled_frames))[dists[0]], human_frames[dists[0]].item(), cycled_frames[dists[0]].item()

def compute_tcc_loss(zc_r, zc_h):
    robot_cycle_back_loss = tcc_loss_(zc_r, zc_h)
    human_cycle_back_loss = tcc_loss_(zc_h, zc_r)
    return robot_cycle_back_loss + human_cycle_back_loss

def compute_optimal_transport_loss(zc_r, zc_h, eps = 1e-10):
    dists = [[] for _ in range(zc_r.shape[0])]
    for i in range(zc_r.shape[0]):
        for j in range(zc_h.shape[0]):
            cosin_dists = batch_cosine_distance(zc_r[i].unsqueeze(0), zc_h[j].unsqueeze(0))
            assignment = distributed_sinkhorn(1-cosin_dists[0])
            distance = torch.sum(assignment * cosin_dists[0])
            dists[i].append(distance)

    return dists