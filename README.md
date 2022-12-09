# github.com/ashweta1/dronerl

Library to run multi-agent reinforcement learning for Drone configuration application.
======================================================================================

dronerl/centralized_mdp: MDP model for centralized model with joint observation and joint action for all agents
- cs229_gym_envs/drone_grid_env.py: My custom implementation of grid env based in GridWorldEnv, extended to support multiple agents and targets.
- main.py: Implementation of Q-learning to train, and a function to load and test the learnt policy.

dronerl/: Multi-agent RL training with PPO/DQN/DDPG using a parameter sharing approach to learn a single policy through ray[rllib], using PettingZooo MPE env.
- train_drone_nav.py: Script to run training using ray[rllib] on a ray cluster
- test_drone_nav.py: Script to test the learnt policy and generae stats.
- analysis/: scripts to generate plots and evaluate the iteration with best expected rewards.
- gym_envs/simple_spread_drone*: Most of the code in this subdirectory is a copy of PettingZoo simple_spread.py MPE env. Some tweaks and updates were needed to the reward function, and termination conditions, because of which I had to make local copies of these files.

Access Demos here:
==================
- DroneGridEnv demo with Q-learning policy: https://youtu.be/aYm5vWn4qRk
- PettingZoo's MPE demo with PPO based policy: https://youtu.be/rpW3ANoM2J8

To run this code
=================
You will need the following packages:<br>
pip install ray[rllib]<br>
pip install gym<br>
pip install tensorflow<br>
pip install pettingzoo<br>
pip install pettingzoo[mpe]<br>

Might need to modify step function in /.local/lib/python3.10/site-packages/ray/rllib/env/wrappers/pettingzoo\_env.py, as follows:
```
def step(self, action\_dict):
    ## Hack to support mpe gym env that returns 5 outputs including truncations.
    #obss, rews, dones, infos = self.par\_env.step(action\_dict)
    obss, rews, dones, _, infos = self.par_env.step(action_dict)
    dones["__all__"] = all(dones.values())
    return obss, rews, dones, infos
```

To train: <br>
python train\_drone\_nav.py  <num_agents> <algo_name[PPO,DQN,DDPG]> <num_training_episodes> <checkpoint_dir> <br>
Example: <br>
python3 train\_drone\_nav.py 6 PPO 120000 ~/ray\_results/ <br>

To test: <br>
python test\_drone\_nav.py  <num_agents> <algo_name[PPO/DQN/DDPG]> <num_test_trials> <input_checkpoint_path> <output_test_results_dir> [<render_mode(rgb_array/human)>] <br>
Example: <br>
python3 test\_drone\_nav.py 6 PPO 10 ~/ray\_results/PPO/PPO\_droneworld\_9b930\_00000\_0\_2022-12-07\_21-10-42/checkpoint\_000040/ ~/test\_results



