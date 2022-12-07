# dronerl

Library to run multi-agent reinforcement learning for Drone configuration application.

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
