import gym
import numpy as np
import ray
from numpy import inf, float32
from gym_env import simple_spread_drone_v2

# For registering and making the environment work with rlib
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
from ray.tune.registry import register_env

# For using rllib to train a model or policy
from ray.tune import tune
from ray.rllib.policy import policy


def env_creator(args):
    num_agents = NUM_AGENTS
    max_cycles = MAX_CYCLES
    local_reward_ratio = LOCAL_REWARD_RATIO
    continuous_actions = False
    if ALGO_NAME == "DDPG":
        continuous_actions = True
    return simple_spread_drone_v2.parallel_env(N=num_agents,
                                               local_ratio=local_reward_ratio,
                                               max_cycles=max_cycles,
                                               continuous_actions=continuous_actions)


def gen_policy_spec():
    model_config = {
        "gamma": GAMMA
    }
    observation_space = gym.spaces.Box(-inf, inf, (18,), float32)

    action_space = gym.spaces.Discrete(5)
    if ALGO_NAME == "DDPG":  # continuous
        action_space = gym.spaces.Box(0.0, 1.0, (5,), float32)

    return policy.PolicySpec(None, observation_space, action_space, model_config)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # Environment settings
    NUM_AGENTS = 3
    MAX_CYCLES = 25
    LOCAL_REWARD_RATIO = 0  # Cooperative task - optimize for global reward

    # Training config settings
    GAMMA = 0.9
    #LAMBDA = 0.9
    NUM_TRAINING_EPISODES = 120000
    CHECKPOINT_FREQ = 10

    # Algorithm
    #ALGO_NAME = "PPO"
    ALGO_NAME = "DQN"
    #ALGO_NAME = "DDPG"

    # Register a parallel simiple_spread_v2 Petting zoo env with ray.
    env_name = "droneworld"
    register_env(env_name, lambda args: ParallelPettingZooEnv(env_creator(args)))

    # Create training config
    config = {
        # Environment specific
        "env": env_name,
        # General
        "num_gpus": 0,
        "num_workers": 5,
        "log_level": "ERROR",
        "framework": "tf",
        #"lambda": LAMBDA,
        "gamma": GAMMA,
        # Method specific
        "multiagent": {
            "policies": {"policy_0": gen_policy_spec()},
            # alternatively, simply do: `PolicySpec(config={"gamma": 0.85})`},
            "policy_mapping_fn": (lambda agent_id: "policy_0")
            # "policy_mapping_fn": (lambda agent_id, episode, **kwargs: agent_id),
        }
    }

    # Run training and store checkpoints
    checkpoint_dir = "./ray_results/"
    tune.run(
        ALGO_NAME,
        name=ALGO_NAME,
        stop={"episodes_total": NUM_TRAINING_EPISODES},
        local_dir=checkpoint_dir,
        checkpoint_freq=CHECKPOINT_FREQ,
        config=config
    )
