# One-Shot Imitation under Mismatched Execution


[Kushal Kedia](https://kushal2000.github.io/)<sup>\*</sup>,  [Prithwish Dan](https://pdan101.github.io/)<sup>\*</sup>, Angela Chao, Maximus A. Pace, [Sanjiban Choudhury](https://sanjibanc.github.io/) (<sup>*</sup>Equal Contribution)
<sup></sup>Cornell University



[Project Page](https://portal-cornell.github.io/rhyme/) | [Link to Paper](https://arxiv.org/pdf/2409.06615)


## 🚀 Installation

Follow these steps to install `RHyME`:

1. Create and activate the conda environment:
   ```bash
   cd rhyme
   conda env create -f environment.yml
   conda activate rhyme
   pip install -e . 
   ```

## 📦 Simulation Dataset

To set up the simulation dataset:

1. Instructions TBD

## 🚴‍♂️ Training

### Simulation

Datasets (Visual Encoder):
- robot 
- human (Sphere-Easy)
- singlehand (Sphere-Medium)
- twohands (Sphere-Hard)
- robot_segments_paired_human and human_segments_paired_human (Optional)
- robot_segments_paired_singlehand and singlehand_segments_paired_singlehand (Optional)
- robot_segments_paired_twohands and twohands_segments_paired_twohands (Optional)

Datasets (Diffusion Policy):
- robot
- imagined demonstrator dataset (will be created)


1. Pretrain visual encoder:
   ```bash
   python scripts/skill_discovery.py
   ```
   Additional options include:
   ```bash
   exp_name (name of model)
   cross_embodiment (human, singlehand, twohands)
   use_paired_data (True/False)
   paired_dataset.percentage_pairing (0-1)
   ```
   Example configs are provided:
   ```bash
   python scripts/skill_discovery.py --config-name=human_pretrain
   ```
2. Convert images into latent vectors using pretrained visual encoder: 
   ```bash
   python scripts/label_sim_kitchen_dataset.py
   ```
   Additional options include:
   ```bash
   cross_embodiment (human, singlehand, twohands)
   pretrain_model_name
   ckpt
   include_robot
   ```
3. Compute and store sequence-level distance metrics between cross embodiment play data and robot data:
   ```
   python scripts/chopped_segment_wise_dists.py
   ``` 
   Additional options include:
   ```bash
   cross_embodiment_segments (e.g. twohands_segments_paired_sample)
   pretrain_model_name
   ckpt 
   num_chops (number of clips to retrieve per robot video)
   ```
4. "Imagine" the paired demonstrator dataset, and store it in the datasets folder:
   ```
   python scripts/reconstruction.py
   ```
   Additional options include:
   ```bash
   cross_embodiment_segments (e.g. twohands_segments_paired_sample)
   pretrain_model_name
   ckpt 
   ot_lookup (True/False)
   tcc_lookup (True/False)
   num_chops (number of clips to retrieve per robot video)
   ```
5. Convert the imagined dataset into latent vectors:
   ```
   python scripts/label_sim_kitchen_dataset.py include_robot=False pretrain_model_name=NO_PAIRING_TWOHANDS cross_embodiment=NO_PAIRING_TWOHANDS_twohands_segments_paired_sample_generated_ot_2_ckpt40
   ```
   Additional options include:
   ```bash
   include_robot (True/False)
   pretrain_model_name
   cross_embodiment (now should be the name of the reconstructed dataset from OT)
   ```
6. Train conditional diffusion policy to translate imagined demonstrator videos into robot actions:
   ```
   python scripts/skill_transfer_composing.py pretrain_model_name=NO_PAIRING_TWOHANDS pretrain_ckpt=40 eval_cfg.demo_type=twohands cross_embodiment=NO_PAIRING_TWOHANDS_twohands_segments_paired_sample_generated_ot_2_ckpt40 dataset.paired_data=True dataset.paired_percent=0.5
   ```
   Additional options include:
   ```bash
   pretrain_model_name
   pretrain_ckpt
   eval_cfg.demo_type (specifies which demonstrator to evaluate on)
   cross_embodiment (reconstructed dataset from OT)
   dataset.paired_data (True if using the imagined paired dataset)
   dataset.paired_percent (hybrid training on robot/imagined dataset)
   ```

### BibTeX
   ```bash
   @article{
      kedia2024one,
      title={One-Shot Imitation under Mismatched Execution},
      author={Kedia, Kushal and Dan, Prithwish and Choudhury, Sanjiban},
      journal={arXiv preprint arXiv:2409.06615},
      year={2024}
   }
   ``` 

### Acknowledgement
* Much of the training pipeline is adapted from [XSkill](https://xskill.cs.columbia.edu/).
* Diffusion Policy is adapted from [Diffusion Policy](https://github.com/real-stanford/diffusion_policy)
* Many useful utilies are adapted from [XIRL](https://x-irl.github.io/).
