import gym
from gym import spaces
import pygame
import numpy as np
import sys

from gym.spaces import MultiDiscrete


class DroneGridEnv(gym.Env):
    _n_drones: int
    _agent_locations: list
    _target_locations: list
    _action_to_direction: dict

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode=None, size=20, num_drones=4):
        self._n_drones = num_drones  # number of drones
        self._agent_locations = []  # locations of agent drones: list _n_drones elements, each an array of size 2
        self._target_locations = []  # locations of targets: list _n_drones elements, each an array of size 2

        self.size = size  # the size of the square grid
        self.window_size = 512  # the size of the PyGame window

        # Observations are dictionaries with the agent's and the target's location. In this env, observations provide
        # information about the locations of all agents and targets on the 2-dimensional grid. Each location is
        # encoded as an element of {0, ..., `size`}^2, i.e. MultiDiscrete([size, size]).
        agent_spaces = []
        target_spaces = []
        for i in range(0, self._n_drones):
            agent_spaces.append(spaces.Box(0, size - 1, shape=(2,), dtype=int))
            target_spaces.append(spaces.Box(0, size - 1, shape=(2,), dtype=int))
        self.observation_space = spaces.Dict(
            {
                "agent": spaces.Tuple(spaces=agent_spaces),
                "target": spaces.Tuple(spaces=target_spaces)
            }
        )

        # Action space describes what actions the agents can take. We have 5 actions in our env: do nothing, right,
        # up, left and down, for each agent drone.
        actions = np.empty(self._n_drones)
        actions.fill(5)
        self.action_space = spaces.MultiDiscrete(actions)

        """
        The following dictionary maps abstract actions from `self.action_space` to 
        the direction we will walk in if that action is taken.
        I.e. 0 corresponds to "right", 1 to "up" etc.
        """
        self._action_to_direction = {
            0: np.array([0, 0]),  # do nothing
            1: np.array([1, 0]),  # go right
            2: np.array([0, 1]),  # go up
            3: np.array([-1, 0]),  # go left
            4: np.array([0, -1]),  # go down
        }

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        """
        If human-rendering is used, `self.window` will be a reference
        to the window that we draw to. `self.clock` will be a clock that is used
        to ensure that the environment is rendered at the correct framerate in
        human-mode. They will remain `None` until human-mode is used for the
        first time.
        """
        self.window = None
        self.clock = None

    def _get_obs(self) -> dict:
        return {"agent": self._agent_locations,
                "target": self._target_locations}

    def _get_info(self) -> dict:
        # Calculate distances between every agent drone and target location. We will do an approximation by randomly
        # choosing the order in which we find the nearest target for a drone.
        # TODO: Find a way to do this efficiently and do proper shortest distance considering all combinations.
        nearest_targets_found = np.empty(self._n_drones, dtype=int)
        nearest_targets_found.fill(-1)
        total_distance = 0.0
        #for i in np.random.choice(self._n_drones, size=self._n_drones, replace=False):
        for i in range(0, self._n_drones):
            # Find nearest target for drone i
            nearest_target_i: int
            nearest_distance_i = sys.maxsize
            for j in range(0, self._n_drones):
                if j in nearest_targets_found:
                    # skip any already regarded as nearest for another drone
                    continue
                # Calculate Forbenius norm distance
                distance = np.linalg.norm(self._agent_locations[i] - self._target_locations[j])
                # update nearest for i
                if distance < nearest_distance_i:
                    nearest_target_i = j
                    nearest_distance_i = distance
            # Update total distance and nearest targets
            nearest_targets_found[i] = nearest_target_i
            total_distance += nearest_distance_i
        #print("Nearest targets: ", nearest_targets_found)
        #print("Total distance: ", total_distance)
        not_is_unique_locations: bool = (len(np.unique(self._agent_locations, axis=0)) < len(self._agent_locations))
        return {"distance": total_distance, "collision": not_is_unique_locations}

    def reset(self, seed=None, options=None):
        """
        Reset method will be called to initiate a new episode.

        Reset method will be called before step method is called. Moreover, reset should be called whenever a done
        signal has been issued. The reset method should return a tuple of the initial observation and some auxiliary
        information
        """
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        # Choose the agent and target locations uniformly at random
        # TODO: Initialize target and agent locations from a pre-configured input file.
        self._agent_locations = []
        self._target_locations = []
        for i in range(0, self._n_drones):
            self._agent_locations.append(self.np_random.integers(0, self.size, size=2, dtype=int))
        if options is not None:
            for target in options["target"]:
                self._target_locations.append(np.array([target[0], target[1]]))
        else:
            for i in range(0, self._n_drones):
                self._target_locations.append(self.np_random.integers(0, self.size, size=2, dtype=int))

        #print("Agent locations: ", self._agent_locations)
        #print("Target locations: ", self._target_locations)

        observation = self._get_obs()
        info = self._get_info()
        #print("Observation: ", observation)
        #print("Info: ", info)

        if self.render_mode == "human":
            self._render_frame()

        return observation, info

    def step(self, action):
        """
        Step accepts an action, computes the state of the environment after applying that action and returns the
        4-tuple (observation, reward, done, info)
        :param action: MultiDiscrete action to be applied
        :return: observation, reward, done, info
        """
        # Map the actions an array of (element of {0,1,2,3,4}) to the direction the drone walks in
        #action: spaces.MultiDiscrete
        for i in range(0, self._n_drones):
            direction = self._action_to_direction[action[i]]
            # Use `np.clip` to make sure we don't leave the grid
            self._agent_locations[i] = np.clip(
                self._agent_locations[i] + direction, 0, self.size - 1
            )

        # An episode is done iff all agent drones have reached their target
        observation = self._get_obs()
        info = self._get_info()
        terminated = False
        if info["collision"] is True:
            reward = -1.0
            terminated = True
        else:
            distance = info["distance"]
            if distance == 0:
                reward = 10.0
                terminated = True
            else:
                reward = -0.1 * distance/(self.size**2)

        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, terminated, False, info

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))
        pix_square_size = (
                self.window_size / self.size
        )  # The size of a single grid square in pixels

        # First we draw the targets
        for target_location in self._target_locations:
            pygame.draw.rect(
                canvas,
                (255, 0, 0),
                pygame.Rect(
                    pix_square_size * target_location,
                    (pix_square_size, pix_square_size),
                ),
            )
        # Now we draw the agents
        for agent_location in self._agent_locations:
            pygame.draw.circle(
                canvas,
                (0, 0, 255),
                (agent_location + 0.5) * pix_square_size,
                pix_square_size / 3,
            )

        # Finally, add some gridlines
        for x in range(self.size + 1):
            pygame.draw.line(
                canvas,
                0,
                (0, pix_square_size * x),
                (self.window_size, pix_square_size * x),
                width=3,
            )
            pygame.draw.line(
                canvas,
                0,
                (pix_square_size * x, 0),
                (pix_square_size * x, self.window_size),
                width=3,
            )

        if self.render_mode == "human":
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to keep the framerate stable.
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
