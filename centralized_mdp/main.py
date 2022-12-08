# This is a sample Python script.
import csv
import math
import time

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.
import gym
import numpy as np
from copy import deepcopy
from cs229_gym_envs import DroneGridEnv


def random_action(size=20, num_drones=4):
    env = gym.make('cs229_gym_envs/DroneGridEnv-v0', render_mode="human", size=size, num_drones=num_drones)

    # Make sure there is a discount factor
    # States should have some properties. to suggest next action
    env.action_space.seed(42)
    observation, info = env.reset(seed=42)
    for _ in range(1000):
        observation, reward, terminated, truncated, info = env.step(env.action_space.sample())
        if terminated or truncated:
            observation, info = env.reset()
    env.close()


def transform_observation_space_to_index(observation: gym.spaces.Dict, size: int):
    # Convert agent observation into a flat row index for list of size (box_size ^ num_agents)
    # for example for box of each side 5, and 2 agent drones, this will be 625
    # Note that this size double exponentially increases and is not practical for any real numbers.
    state_index = 0
    i = 0
    for agent_space in observation["agent"]:
        agent_space: gym.spaces.Box
        #print("Observation agent space box coordinate: ", agent_space)
        grid_index = agent_space[0] * size + agent_space[1]
        box_size = size**2
        state_index += (box_size ** i) * grid_index
        i += 1
    #print("State index = ", state_index)
    return state_index


def transform_action_space_to_index(actions: gym.spaces.MultiDiscrete):
    # Convert agent actions into a flat column index
    action_index = 0
    # action is a multidiscrete space of size num_drones, which each action able to take n values.
    i = 0
    for a in actions:
        action_index += (a.n ** i) * a
    #print("Action index = ", action_index)
    return action_index


def transform_action_index_to_space(action_index: int, num_drones: int, action_space_size: int):
    actions = np.empty(num_drones)
    i = num_drones - 1
    action_index_copy = deepcopy(action_index)
    while i >= 0:
        a = math.floor(action_index_copy / (action_space_size ** i))
        actions[i] = a
        action_index_copy -= (a * (action_space_size ** i))
        i -= 1
    #action_space = gym.spaces.MultiDiscrete(actions)
    #print("Action space for index: ", action_index, " = ", actions)
    return actions


def q_learning(size=10, num_drones=1, targets=None):
    # 1. create the environment
    #env = gym.make('cs229_gym_envs/DroneGridEnv-v0', render_mode="human", size=size, num_drones=num_drones)
    env = gym.make('cs229_gym_envs/DroneGridEnv-v0', render_mode=None, size=size, num_drones=num_drones)
    # for key in env.observation_space.spaces:
    #     print("Observation space: ", key, ": ", env.observation_space.spaces[key])
    #     print("Number of spaces for ", key, ": ", len(env.observation_space.spaces[key]))
    #     print("Shape of each space: ", env.observation_space.spaces[key][0].shape)
    # print("Action space: ", env.action_space)
    # print("Action space shape: ", env.action_space.shape)
    # print("Action space n: ", env.action_space[0].n)

    # 2. Initialize the Q-table
    box_size = size ** 2
    action_size = env.action_space[0].n
    print("Per agent observation space size = ", box_size)
    print("Per agent action space size = ", action_size)
    # Example for a box of each side 5 and 2 drones, with 5 actions each, this will be of shape (625, 25),
    # and with 4 drones (390625, 625)
    #Q = np.zeros([box_size ** num_drones, action_size ** num_drones])
    Q = np.random.uniform(low=-1.0, high=1.0, size=([box_size ** num_drones, action_size ** num_drones]))

    print("Shape of Q: ", Q.shape)

    # print(Q)

    # 2. Parameters of Q-learning
    learning_rate = .1
    discount = .95
    epochs = NUM_EPOCHS
    batch_size = 100
    max_cycles = MAX_CYCLES
    rew_list = []  # rewards per episode calculate
    options = {"target" : targets}
    terminations = []
    episodes_total = 0

    # File to save training results in
    timestr = time.strftime("%Y%m%d-%H%M%S")
    progress_filename = 'train_results/progress_' + str(num_drones) + '.' + timestr
    print("Saving training progress in: ", progress_filename)
    f = open(progress_filename, 'w')
    write = csv.writer(f)
    write.writerow(["training_iteration", "episode_reward_mean", "episodes_total"])

    # File to save the modelled policy in.
    model_filename = 'trained_policy/qtable_' + str(num_drones) + '.' + timestr

    training_iteration = 0
    episodes_total = 0

    # 3. Q-learning Algorithm
    for _ in range(epochs):
        training_iteration += 1
        print("Training iteration # = ", training_iteration)

        rbatch = 0
        for _ in range(batch_size):
            episodes_total += 1
            #print("Episode # = ", episodes_total)

            # Reset environment
            observation, _ = env.reset(options=options)
            s = transform_observation_space_to_index(observation, size)

            rAll = 0
            for _ in range(0, max_cycles):
                env.render()

                # Choose random action
                # random_nums = np.random.randn(1, np.size(Q, axis=1))
                # print("Random action = ", random_nums)
                # a = np.argmax(Q[s, :] + random_nums * (1. / (i + 1)))

                # Choose action from Q table
                a = np.argmax(Q[s, :])
                # print("Chosen action index: ", a)
                action: gym.spaces.MultiDiscrete = transform_action_index_to_space(a, num_drones, action_size)
                # print("Chosen action: ", action)

                # Take the action and get new state & reward from environment
                observation, reward, terminated, _, info = env.step(action)
                s1 = transform_observation_space_to_index(observation, size)

                # Update Q-Table with new knowledge
                # print("Reward = ", reward)
                Q[s, a] = (1 - learning_rate) * Q[s, a] + learning_rate * (reward + discount * np.max(Q[s1, :]))
                rAll += reward

                # Update the state to new state
                s = s1

                # record successful terminations
                # if terminated:
                #     print("TERMINATED!!!")
                #     print(info)
                #     success = (info["distance"] == 0)
                #     terminations.append([success, reward])
                #     break

                #rew_list.append(rAll)
                # np.savetxt('rewards_list.txt', rew_list)
                # np.savetxt('terminations.txt', terminations)
                env.render()

            rbatch += rAll

        # Append training iteration (epoch)#, total number of episodes and episode reward mean for the iteration.
        episode_reward_mean = rbatch/batch_size
        progress = [training_iteration, episode_reward_mean, episodes_total]
        print(progress)
        write.writerow(progress)

    # Save the model at the end of the training
    print("Saving learnt Qtable to: ", model_filename)
    np.save(model_filename, Q)
    return Q

def test_policy(Q, size=10, num_drones=1, targets=None, num_trials=10):
    # Reset environment
    env = gym.make('cs229_gym_envs/DroneGridEnv-v0', render_mode="human", size=size, num_drones=num_drones)
    options = {"target": targets}
    action_size = env.action_space[0].n
    success = 0
    collisions = 0
    rAll = 0
    steps_to_success = 0
    for _ in range(num_trials):
        observation, _ = env.reset(options=options)
        s = transform_observation_space_to_index(observation, size)
        env.render()
        for step in range(0, MAX_CYCLES):
            # Choose action from Q table
            a = np.argmax(Q[s, :])
            # print("Chosen action index: ", a)
            action: gym.spaces.MultiDiscrete = transform_action_index_to_space(a, num_drones, action_size)
            # print("Chosen action: ", action)

            # Take the action and get new state & reward from environment
            observation, reward, terminated, _, info = env.step(action)
            s1 = transform_observation_space_to_index(observation, size)
            rAll += reward

            # Update the state to new state
            s = s1
            env.render()
            if terminated:
                if info["distance"] == 0:
                    success += 1
                    steps_to_success += (step + 1)
                if info["collision"]:
                    collisions += 1
                break

        # Take a second before starting another run.
        time.sleep(1)
    print("==================================================")
    print("Average reward = ", rAll / num_trials)
    print("Success rate = ", 100 * success/num_trials, "%")
    print("Avg steps to success = ", steps_to_success/num_trials)
    print("Collision rate = ", 100 * collisions / num_trials, "%")
    print("==================================================")


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    gym.envs.register(
        id='cs229_gym_envs/DroneGridEnv-v0',
        entry_point='cs229_gym_envs:DroneGridEnv',
        max_episode_steps=300,
    )
    NUM_EPOCHS = 1000
    MAX_CYCLES = 10
    qtable = q_learning(5, 3, [[1, 1], [3, 1], [2, 3]])
    #qtable = np.load("trained_policy/qtable_3.20221208-120440.npy")
    test_policy(qtable, 5, 3, [[1, 1], [3, 1], [2, 3]], num_trials=100)
    # random_action(size=20, num_drones=4)
