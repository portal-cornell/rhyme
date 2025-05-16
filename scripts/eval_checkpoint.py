import os
import pickle
import uuid
import hydra
import numpy as np
import torch
import torch.nn as nn
import wandb
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel
from omegaconf import DictConfig, OmegaConf
from rhyme.model.diffusion_model import get_resnet, replace_bn_with_gn
from rhyme.model.encoder import ResnetConv
import random
from tqdm import tqdm
import json
from rhyme.utility.observation_indices import ACTION_INDICES

def create_policy_nets(cfg):
    """
    Creates policy network architecture consisting of:
    1) Vision Encoder (ResNet)
    2) Prototype Predictor (Skill Alignment Transformer)
    3) Noise Predictor (Diffusion Model)


    Parameters
    ----------
    cfg : DictConfig
        Specifies model architecture


    Returns 
    -------
    nets : torch.nn.ModuleDict
        Dictionary with all three network components
    """
    pred_horizon = cfg.pred_horizon
    obs_horizon = cfg.obs_horizon
    proto_horizon = cfg.proto_horizon

    if cfg.vision_feature_dim == 512:
        vision_encoder = get_resnet("resnet18")
    else:
        vision_encoder = ResnetConv(embedding_size=cfg.vision_feature_dim)

    vision_encoder = replace_bn_with_gn(vision_encoder)
    vision_feature_dim = cfg.vision_feature_dim

    obs_dim = cfg.obs_dim
    action_dim = cfg.action_dim
    proto_dim = cfg.proto_dim

    if cfg.upsample_proto:
        noise_pred_net = hydra.utils.instantiate(
            cfg.noise_pred_net,
            global_cond_dim=vision_feature_dim * obs_horizon +
            obs_dim * obs_horizon +
            proto_horizon * cfg.upsample_proto_net.out_size,
        )
    else:
        noise_pred_net = hydra.utils.instantiate(
            cfg.noise_pred_net,
            global_cond_dim=vision_feature_dim * obs_horizon +
            obs_dim * obs_horizon + proto_horizon * proto_dim,
        )
    proto_pred_net = hydra.utils.instantiate(
        cfg.proto_pred_net,
        input_dim=vision_feature_dim * obs_horizon + obs_dim * obs_horizon,
    )

    # Complete policy architecture
    nets = nn.ModuleDict({
        "vision_encoder": vision_encoder,
        "proto_pred_net": proto_pred_net,
        "noise_pred_net": noise_pred_net,
    })

    if cfg.upsample_proto:
        upsample_proto_net = hydra.utils.instantiate(cfg.upsample_proto_net)
        nets["upsample_proto_net"] = upsample_proto_net

    return nets

def get_attempted_unspecified_tasks(task_list, initial_obs, final_obs, epsilon=1e-3):
    count = 0
    obs_diff = np.abs(final_obs - initial_obs)
    # remove "full _____" tasks that are identical to their corresponding non-full tasks
    unspecified_tasks = [task for task in ACTION_INDICES \
                            if task not in task_list \
                            and (len(task) < 5 or f"{task[5:]}" not in ACTION_INDICES) \
                            and task != 'hinge cabinet']
    for task in unspecified_tasks:
        if np.any(obs_diff[ACTION_INDICES[task]] > epsilon):
            count += 1

    return count

@hydra.main(
    version_base=None,
    config_path="../config/simulation",
    config_name="eval_checkpoint",
)
def main(cfg: DictConfig):
    """
    Uses trained XSkill policy to evaluate on (19) held out episodes,
    with subtasks: ['microwave', 'kettle', 'light switch', 'slide cabinet'].


    Parameters
    ----------
    cfg : DictConfig
        Specifies configurations for policy networks as well as evaluation details

    Side Effects
    ------------
    - Saves gifs of policy rollouts and a json with subtask completion percentages to trained_model_path folder

    Returns 
    -------
    None
    """
    save_dir = os.path.join(cfg.trained_model_path, 'post_training')
    model_cfg = OmegaConf.load(os.path.join(cfg.trained_model_path, 'hydra_config.yaml'))

    nets = create_policy_nets(cfg)
    noise_scheduler = hydra.utils.instantiate(cfg.noise_scheduler)

    device = torch.device("cuda")
    _ = nets.to(device)
    nets.eval()

    eval_callback = hydra.utils.instantiate(cfg.eval_callback)
    file = open(os.path.join(cfg.trained_model_path, 'stats.pickle'), 'rb')
    stats = pickle.load(file)

    with open(cfg.eval_cfg.eval_mask_path, "r") as f:
        eval_mask = json.load(f)
        eval_mask = np.array(eval_mask)
    
    eval_eps = np.arange(len(eval_mask))[eval_mask]


    result_dict = {
        'robot':{
            f'{ckpt_num}': {f'{speed}': {} for speed in cfg.robot_speeds} for ckpt_num in cfg.checkpoint_list
        },
        cfg.eval_cfg.demo_type:{
            f'{ckpt_num}': {f'{speed}': {} for speed in cfg.human_speeds} for ckpt_num in cfg.checkpoint_list
        }
    }

    human_type = cfg.eval_cfg.demo_type
    speeds = {
        'robot': cfg.robot_speeds,
        human_type: cfg.human_speeds
    }
    
    for i, ckpt_num in enumerate(cfg.checkpoint_list):
        nets.load_state_dict(torch.load(os.path.join(cfg.trained_model_path, f'ckpt_{ckpt_num}.pt')))
        for demo_type in [human_type, 'robot']:
            cfg.eval_cfg.demo_type = demo_type
            for speed in speeds[demo_type]:
                task_list = ["slide cabinet", "light switch", "kettle", "microwave"]
                eval_callback.task_progess_ratio = speed
                tasks_completed = 0
                all_correct_count = 0
                num_unspecified_tasks = 0
                for seed in eval_eps:
                    cfg.eval_cfg.demo_item = seed.item()
                    num_completed, _, initial_obs, final_obs = eval_callback.eval(
                        nets,
                        noise_scheduler,
                        stats,
                        cfg.eval_cfg,
                        save_dir,
                        seed,
                        epoch_num=None,
                        task_list=task_list,
                        model_cfg=model_cfg
                    )
                    tasks_completed += num_completed

                    if num_completed == len(task_list):
                        all_correct_count += 1

                    num_unspecified_tasks += get_attempted_unspecified_tasks(task_list, initial_obs, final_obs)

                # result_dict[demo_type][f'{ckpt_num}'][f'{speed}'] = tasks_completed / (4*len(eval_eps))
                result_dict[demo_type][f'{ckpt_num}'][f'{speed}']['share-of-tasks'] = tasks_completed / (4*len(eval_eps))
                result_dict[demo_type][f'{ckpt_num}'][f'{speed}']['all-tasks'] = all_correct_count / len(eval_eps)
                result_dict[demo_type][f'{ckpt_num}'][f'{speed}']['num-unspecified'] = num_unspecified_tasks / len(eval_eps)
                print(result_dict)
    
    with open(os.path.join(save_dir, "policy_results.json"), "w") as outfile:
        # result_dict = json.load(outfile)
        json.dump(result_dict, outfile)

    averages = {"robot": {f'{speed}': {'share-of-tasks': 0, 'all-tasks': 0, 'num-unspecified': 0} for speed in speeds['robot']}, human_type: {f'{speed}': {'share-of-tasks': 0, 'all-tasks': 0, 'num-unspecified': 0} for speed in speeds[human_type]}}
    counts = {"robot": {f'{speed}': 0 for speed in speeds['robot']}, human_type: {f'{speed}': 0 for speed in speeds[human_type]}}

    for demo_type, values in result_dict.items():
        for ckpt_num, acc_dicts in values.items():
            for exec_speed, acc in acc_dicts.items():
                # averages[demo_type][exec_speed] += acc
                averages[demo_type][exec_speed]['share-of-tasks'] += acc['share-of-tasks']
                averages[demo_type][exec_speed]['all-tasks'] += acc['all-tasks']
                averages[demo_type][exec_speed]['num-unspecified'] += acc['num-unspecified']
                counts[demo_type][exec_speed] += 1

    for demo_type, values in averages.items():
        for exec_speed, summed_acc in values.items():
            # averages[demo_type][exec_speed] /= counts[demo_type][exec_speed]
            averages[demo_type][exec_speed]['share-of-tasks'] /= counts[demo_type][exec_speed]
            averages[demo_type][exec_speed]['all-tasks'] /= counts[demo_type][exec_speed]
            averages[demo_type][exec_speed]['num-unspecified'] /= counts[demo_type][exec_speed]

    with open(os.path.join(save_dir, "policy_results_avg.json"), "w") as outfile:
        json.dump(averages, outfile)
            

if __name__ == '__main__':
    main()