# Parallel Q Learning (PQL)
This repository provides a PyTorch implementation of the paper [Parallel *Q*-Learning: Scaling Off-policy Reinforcement Learning under Massively Parallel Simulation](https://arxiv.org/abs/2307.12983).

[Zechu Li*](https://supersglzc.github.io/), [Tao Chen*](https://taochenshh.github.io/), [Zhang-Wei Hong](https://williamd4112.github.io/), [Anurag Ajay](https://anuragajay.github.io/), [Pulkit Agrawal](https://people.csail.mit.edu/pulkitag/)

---

- [:books: Citation](#citation)
- [:gear: Installation](#installation)
    - [Install :zap: PQL](#install_pql)
    - [Install Isaac Gym](#install_isaac)
    - [System Requirements](#requirements)
- [:scroll: Usage](#usage)
    - [:pencil2: Logging](#usage_logging)
    - [:bulb: Train with :zap: PQL](#usage_pql)
    - [:bookmark: Baselines](#usage_baselines)
    - [:floppy_disk: Saving and Loading](#usage_saving_loading)
- [:clap: Acknowledgement](#acknowledgement)


## :books: Citation

```
@inproceedings{li2023parallel,
  title={Parallel $Q$-Learning: Scaling Off-policy Reinforcement Learning under Massively Parallel Simulation},
  author={Li, Zechu and Chen, Tao and Hong, Zhang-Wei and Ajay, Anurag and Agrawal, Pulkit},
  booktitle={International Conference on Machine Learning},
  year={2023},
  organization={PMLR}
}
```

## :gear: Installation

### Install :zap: PQL <a name="install_pql"></a>

1. Clone the package:

    ```bash
    git clone git@github.com:Improbable-AI/pql.git
    cd pql
    ```

2. Create Conda environment and install dependencies:

    ```bash
    ./create_conda_env_pql.sh
    pip install -e .
    ```


### Install Isaac Gym <a name="install_isaac"></a>

> **Note**
> In original paper, we use Isaac Gym Preview 3 and task configs in commit ca7a4fb762f9581e39cc2aab644f18a83d6ab0ba in IsaacGymEnvs.

1. Download and install Isaac Gym Preview 4 from https://developer.nvidia.com/isaac-gym

2. Unzip the file:
    ```bash
    tar -xf IsaacGym_Preview_4_Package.tar.gz
    ```

3. Install IsaacGym
    ```bash
    cd isaacgym/python
    pip install -e . --no-deps
    ```

5. Install IsaacGymEnvs

    ```bash
    git clone https://github.com/NVIDIA-Omniverse/IsaacGymEnvs.git
    cd IsaacGymEnvs
    pip install -e . --no-deps
    ```
    
6. Export LIBRARY_PATH
    
    ```bash
    export LD_LIBRARY_PATH=$(conda info --base)/envs/pql/lib/:$LD_LIBRARY_PATH
    ```

### System Requirements <a name="requirements"></a>
> **Warning**
> Note that wall-clock efficiency highly depends on the GPU type and will decrease with smaller/fewer GPUs (check Section 4.4 in the paper).

Isaac Gym requires an NVIDIA GPU. To train in the default configuration, we recommend a GPU with at least 10GB of VRAM. For smaller GPUs, you can decrease the number of parallel environments (`cfg.num_envs`), batch_size (`cfg.algo.batch_size`), replay buffer capacity (`cfg.algo.memory_size`), etc. :zap: PQL can run on 1/2/3 GPUs (set GPU ID `cfg.p_learner_gpu` and `cfg.v_learner_gpu`; default GPU ID for Isaac Gym env is `GPU:0`). 


## :scroll: Usage

### :pencil2: Logging <a name="usage_logging"></a>

We use Weights & Biases (W&B) for logging. 

1. Get a W&B account from https://wandb.ai/site

2. Get your API key from https://wandb.ai/authorize

3. set up your account in terminal
    ```bash
    export WANDB_API_KEY=$API Key$
    ```

### :bulb: Train with :zap: PQL <a name="usage_pql"></a>

Run :zap: PQL on Allegro Hand task. A full list of tasks in Isaac Gym is available [here](https://github.com/NVIDIA-Omniverse/IsaacGymEnvs/blob/main/docs/rl_examples.md).

```bash
python scripts/train_pql.py task=AllegroHand
```

Run :zap: PQL-D (with distributional RL)

```bash
python scripts/train_pql.py task=AllegroHand algo.distl=True algo.cri_class=DistributionalDoubleQ
```

Run :zap: PQL on a single GPU. The default is on 2 GPUs. Please specify the GPU id.

```bash
python scripts/train_pql.py task=AllegroHand algo.num_gpus=1 algo.p_learner_gpu=0 algo.v_learner_gpu=0
```

Run :zap: PQL on 3 GPUs. 

```bash
python scripts/train_pql.py task=AllegroHand algo.p_learner_gpu=1 algo.v_learner_gpu=2
```

### :bookmark: Baselines <a name="usage_baselines"></a>

Run DDPG baseline

```bash
python scripts/train_baselines.py algo=ddpg_algo task=AllegroHand
```

Run SAC baseline

```bash
python scripts/train_baselines.py algo=sac_algo task=AllegroHand
```

Run PPO baseline

```bash
python scripts/train_baselines.py algo=ppo_algo task=AllegroHand isaac_param=True
```


    

### :floppy_disk: Saving and Loading <a name="usage_saving_loading"></a>

Checkpoints are automatically saved as W&B [Artifacts](https://docs.wandb.ai/ref/python/artifact).

To load and visualize the policy, run

```bash
python scripts/visualize.py task=AllegroHand headless=False num_envs=10 artifact=$team-name$/$project-name$/$run-id$/$version$
```



## :clap: Acknowledgement

We thank the members of the Improbable AI lab for the helpful discussions and feedback on the paper. We are grateful to MIT Supercloud and the Lincoln Laboratory Supercomputing Center for providing HPC resources.
