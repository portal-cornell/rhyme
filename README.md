# One-Shot Imitation under Mismatched Execution


[Kushal Kedia](https://kushal2000.github.io/)<sup>\*</sup>,  [Prithwish Dan](https://pdan101.github.io/)<sup>\*</sup>, Angela Chao, Maximus A. Pace, [Sanjiban Choudhury](https://sanjibanc.github.io/) (<sup>*</sup>Equal Contribution)
<sup></sup>Cornell University



[Project Page](https://portal-cornell.github.io/rhyme/) | [Link to Paper](https://arxiv.org/pdf/2409.06615)


## Installation

Follow these steps to install `RHyME`:

1. Create and activate the conda environment:
   ```bash
   cd rhyme
   conda env create -f environment.yml
   conda activate rhyme
   pip install -e . 
   ```
2. Before running any scripts, make sure to set "base_dev_dir" to your working directory for the codebase. You may directly write this value into the config files under ./config/simulation, 
or alternatively override the argument in the command line when running scripts.

## Simulation Dataset

All datasets will be loaded in the code using HuggingFace API.

Datasets can be found at: https://huggingface.co/datasets/prithwishdan/RHyME



## Training

1. Pretrain visual encoder:
   ```bash
   python scripts/skill_discovery.py
   ```
   Example configs are provided:
   ```bash
   python scripts/skill_discovery.py --config-name=easy_pretrain_hf
   python scripts/skill_discovery.py --config-name=medium_pretrain_hf
   python scripts/skill_discovery.py --config-name=hard_pretrain_hf
   ```
   Additional options include:
   ```bash
   Model.use_opt_loss (default=False)
   Model.use_tcc_loss (default=False)
   ```
2. Convert images into latent vectors using pretrained visual encoder: 
   ```bash
   python scripts/label_sim_kitchen_dataset.py
   ```
   Additional options include:
   ```bash
   pretrain_model_name (Folder name of vision encoder in ./experiment/pretrain)
   ckpt (Checkpoint number)
   cross_embodiment (sphere-easy, sphere-medium, sphere-hard)
   ```
3. Compute and store sequence-level distance metrics between cross embodiment play data and robot data:
   ```
   python scripts/chopped_segment_wise_dists.py
   ``` 
   Additional options include:
   ```bash
   pretrain_model_name (Folder name of vision encoder in ./experiment/pretrain)
   ckpt (Checkpoint number) 
   num_chops (Number of clips to retrieve per robot video)
   cross_embodiment (sphere-easy, sphere-medium, sphere-hard)
   ```
4. "Imagine" the paired demonstrator dataset, and store it in the datasets folder:
   ```
   python scripts/reconstruction.py 
   ```
   Additional options include:
   ```bash
   pretrain_model_name (Folder name of vision encoder in ./experiment/pretrain)
   ckpt (Checkpoint number) 
   num_chops (Number of clips to retrieve per robot video)
   cross_embodiment (sphere-easy, sphere-medium, sphere-hard)
   ot_lookup (default=True)
   tcc_lookup (default=False)
   ```
5. Convert the imagined dataset into latent vectors:
   ```
   python scripts/label_retrieved_dataset.py
   ```
   Additional options include:
   ```bash
   pretrain_model_name (Folder name of vision encoder in ./experiment/pretrain)
   ckpt (Checkpoint number) 
   imagined_dataset (Folder name of imagined dataset in ./datasets/kitchen_dataset)
   ```
6. Train conditional diffusion policy to translate imagined demonstrator videos into robot actions:
   ```
   python scripts/skill_transfer_composing.py
   ```
   Additional options include:
   ```bash
   pretrain_model_name (Folder name of vision encoder in ./experiment/pretrain)
   pretrain_ckpt (Checkpoint number) 
   eval_cfg.demo_type (Specifies which demonstrator to evaluate on)
   cross_embodiment (Folder name of imagined dataset in ./datasets/kitchen_dataset)
   dataset.paired_data (True if using the imagined paired dataset)
   dataset.paired_percent (default=0.5, hybrid training on robot/imagined dataset)
   ```

## Evaluation

1. Evaluate policy on demonstrator videos:
   ```
   python scripts/eval_checkpoint.py
   ```
   Additional options include:
   ```bash
   pretrain_model_name (Folder name of vision encoder in ./experiment/pretrain)
   pretrain_ckpt (Checkpoint number) 
   eval_cfg.demo_type (Specifies which demonstrator to evaluate on)
   policy_name (Folder name of diffusion policy in ./experiment/diffusion_bc/kitchen)
   ```


### BibTeX
   ```bash
   @article{
      kedia2024one,
      title={One-shot imitation under mismatched execution},
      author={Kedia, Kushal and Dan, Prithwish and Chao, Angela and Pace, Maximus Adrian and Choudhury, Sanjiban},
      journal={arXiv preprint arXiv:2409.06615},
      year={2024}
   }
   ``` 

### Acknowledgement
* Much of the training pipeline is adapted from [XSkill](https://xskill.cs.columbia.edu/).
* Diffusion Policy is adapted from [Diffusion Policy](https://github.com/real-stanford/diffusion_policy)
* Many useful utilies are adapted from [XIRL](https://x-irl.github.io/).
