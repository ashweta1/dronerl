import pickle
import sys
import time
import os

# Environment
import gym
import numpy as np
import ray
from numpy import inf, float32
from gym_env import simple_spread_drone_v2
from pettingzoo.mpe._mpe_utils.core import World
from pathlib import Path

# For registering and making the environment work with rlib
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
from ray.tune.registry import register_env
from ray.rllib.agents.ppo import PPOTrainer
from ray.rllib.algorithms.dqn import DQN
from ray.rllib.algorithms.ddpg import DDPG


def env_creator(args):
    num_agents = NUM_AGENTS
    max_cycles = MAX_CYCLES
    local_reward_ratio = LOCAL_REWARD_RATIO
    continuous_actions = False
    if ALGO_NAME == "DDPG":
        continuous_actions = True
    if RENDER_MODE is not None:
        return simple_spread_drone_v2.parallel_env(N=num_agents,
                                                   local_ratio=local_reward_ratio,
                                                   max_cycles=max_cycles,
                                                   continuous_actions=continuous_actions,
                                                   render_mode=RENDER_MODE)
    else:
        return simple_spread_drone_v2.parallel_env(N=num_agents,
                                                   local_ratio=local_reward_ratio,
                                                   max_cycles=max_cycles,
                                                   continuous_actions=continuous_actions)


def test_policy(env, policy_agent, max_timesteps):
    observations = env.reset()
    #env.reset_landmarks_in_pattern()
    #reset_my_world_landmarks(env)
    #print("Initial observations: ", observations)
    reward_sum = 0.0
    all_done = False
    for timestep in range(max_timesteps):
        # PPO trains a single policy for all agents
        # Get action dict for all agents by computing single action for each agent.
        actions_dict = {}
        for agent_key in observations.keys():
            action, _, _ = policy_agent.get_policy("policy_0").compute_single_action(observations[agent_key])
            actions_dict[agent_key] = action
        # Step all agents according to their respective actions
        observations, rewards, dones, _, infos = env.step(actions_dict)
        env.render()

        collisions = 0
        #print("Infos = ", infos)
        #print("truncations = ", truncations)
        for agent_key in infos:
            if "collision" in infos[agent_key]:
                collisions += 1


        # print(env.aec_env.unwrapped)
        # print(env.aec_env.unwrapped.terminations)

        reward_sum += sum(rewards.values())
        # print("New rewards: ", rewards)
        if len(dones) > 0 and all(value == True for value in dones.values()):
            all_done = True
            break

    if all_done:
        print("Trial Success in ", timestep, " time-steps.")
    else:
        print("Trial Failed to finish the challenge.")
    print("Trial reward: ", reward_sum)
    print("Trial collisions: ", collisions)
    return reward_sum, timestep, all_done, collisions


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # Test policy parameters
    if len(sys.argv) > 7:
      UsageError('Too many command-line arguments.')
    if len(sys.argv) < 6:
      print("Usage: ", sys.argv[0], " <num_agents> <algo_name[PPO/DQN/DDPG]> <num_test_trials> <input_checkpoint_path> <output_test_results_dir> [<render_mode(rgb_array/human)>]")
      exit(1)
    # Num agents (e.g. 3)
    NUM_AGENTS = int(sys.argv[1])
    # Algorithm [PPO, DQN, DDPG]
    ALGO_NAME = sys.argv[2]
    # Number of trials (e.g. 1000)
    NUM_TRIALS = int(sys.argv[3])
    # Check point path to read the policy from
    # Example: ray_results/PPO/PPO_droneworld_22c26_00000_0_2022-12-06_19-13-35/checkpoint_000480
    CHECKPOINT_PATH= sys.argv[4]
    # Output dir for test results
    OUTPUT_TEST_RESULTS_DIR = sys.argv[5]
    RENDER_MODE=None
    if len(sys.argv) > 6:
      RENDER_MODE = sys.argv[6]

    print("=========================================================")
    print("Starting test run with the following configuration:")
    print("NUM_AGENTS = ", NUM_AGENTS)
    print("ALGO_NAME = ", ALGO_NAME)
    print("NUM_TRIALS = ", NUM_TRIALS)
    print("CHECKPOINT_PATH = ", CHECKPOINT_PATH)
    print("OUTPUT_TEST_RESULTS_DIR = ", OUTPUT_TEST_RESULTS_DIR)
    print("RENDER_MODE = ", RENDER_MODE)
    print("=========================================================")

    # Environment settings
    MAX_CYCLES = 25
    LOCAL_REWARD_RATIO = 0  # give higher importance to global reward.

    env_name = "droneworld"

    # Register a parallel simiple_spread_v2 Petting zoo env with ray.
    register_env(env_name, lambda args: ParallelPettingZooEnv(env_creator(args)))

    # Create env
    env = env_creator({})
    print("Env observation space: ", env.observation_spaces)
    print("Env action space: ", env.action_spaces)

    # Load config from checkpoint path
    params_path = Path(CHECKPOINT_PATH).parent / "params.pkl"
    config = None
    with open(params_path, "rb") as f:
        config = pickle.load(f)
        print("Loaded config: ", config)
        # num_workers not needed since we are not training
        del config["num_workers"]
        del config["num_gpus"]
        # del config["num_cpus"]
        # ray.init(num_cpus=1)

    # Load policy
    agent = None
    if ALGO_NAME == "PPO":
        agent = PPOTrainer(env=env_name, config=config)
        agent.restore(CHECKPOINT_PATH)
    elif ALGO_NAME == "DQN":
        agent = DQN(env=env_name, config=config)
        agent.restore(CHECKPOINT_PATH)
    elif ALGO_NAME == "DDPG":
        agent = DDPG(env=env_name, config=config)
        agent.restore(CHECKPOINT_PATH)
    else:
        print("Algo not supported: ", ALGO_NAME)
        exit(1)


    # Test policy for the env
    total_reward = 0.0
    total_success = 0
    total_success_steps = 0
    total_collisions = 0
    trials_with_collisions = 0
    reward_done_list = list()
    #reward_done_list.append(["Trial Num", "Reward", "Time steps", "Success", "Collisions"])
    for i in range(NUM_TRIALS):
        print("Trial: ", i + 1)
        reward_sum, timesteps, all_done, collisions = test_policy(env, agent, max_timesteps=MAX_CYCLES)
        reward_done_list.append([i, reward_sum, timesteps, all_done, collisions])
        total_reward += reward_sum
        if all_done:
            total_success += 1
            total_success_steps += timesteps
        total_collisions += collisions
        if collisions > 0:
            trials_with_collisions += 1


    # Print and save results
    print("Checkpoint: ", CHECKPOINT_PATH)
    print("Average reward per trial: ", total_reward / NUM_TRIALS)
    print("% Success rate = ", (total_success / NUM_TRIALS) * 100, "%")
    if total_success > 0:
        print("Average time steps to success = ", total_success_steps / total_success)
    print("Average collisions per trial: ", total_collisions / NUM_TRIALS)
    print("% Trials with collisions = ", (trials_with_collisions / NUM_TRIALS) * 100, "%")

    timestr = time.strftime("%Y%m%d-%H%M%S")
    results_filename = os.path.join(OUTPUT_TEST_RESULTS_DIR, ALGO_NAME + "_reward_done_list." + timestr)
    print("Saving results in: ", results_filename)
    np.savetxt(results_filename, reward_done_list)

    env.close()
