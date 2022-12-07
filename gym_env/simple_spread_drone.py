# noqa
"""
# Simple Spread

```{figure} mpe_simple_spread.gif
:width: 140px
:name: simple_spread
```

This environment is part of the <a href='..'>MPE environments</a>. Please read that page first for general information.

| Import               | `from pettingzoo.mpe import simple_spread_v2` |
|----------------------|-----------------------------------------------|
| Actions              | Discrete/Continuous                           |
| Parallel API         | Yes                                           |
| Manual Control       | No                                            |
| Agents               | `agents= [agent_0, agent_1, agent_2]`         |
| Agents               | 3                                             |
| Action Shape         | (5)                                           |
| Action Values        | Discrete(5)/Box(0.0, 1.0, (5))                |
| Observation Shape    | (18)                                          |
| Observation Values   | (-inf,inf)                                    |
| State Shape          | (54,)                                         |
| State Values         | (-inf,inf)                                    |

```{figure} ../../_static/img/aec/mpe_simple_spread_aec.svg
:width: 200px
:name: simple_spread
```

This environment has N agents, N landmarks (default N=3). At a high level, agents must learn to cover all the landmarks while avoiding collisions.

More specifically, all agents are globally rewarded based on how far the closest agent is to each landmark (sum of the minimum distances). Locally, the agents are penalized if they collide with other agents (-1 for each collision). The relative weights of these rewards can be controlled with the
`local_ratio` parameter.

Agent observations: `[self_vel, self_pos, landmark_rel_positions, other_agent_rel_positions, communication]`

Agent action space: `[no_action, move_left, move_right, move_down, move_up]`

### Arguments

``` python
simple_spread_v2.env(N=3, local_ratio=0.5, max_cycles=25, continuous_actions=False)
```



`N`:  number of agents and landmarks

`local_ratio`:  Weight applied to local reward and global reward. Global reward weight will always be 1 - local reward weight.

`max_cycles`:  number of frames (a step for each agent) until game terminates

`continuous_actions`: Whether agent action spaces are discrete(default) or continuous

"""

import numpy as np
from gymnasium.utils import EzPickle

from pettingzoo.utils.conversions import parallel_wrapper_fn

from pettingzoo.mpe._mpe_utils.core import Agent, Landmark, World
from pettingzoo.mpe._mpe_utils.scenario import BaseScenario
from pettingzoo.mpe._mpe_utils.simple_env import make_env
from gym_env.simple_env_drone import SimpleEnvDrone


class raw_env(SimpleEnvDrone, EzPickle):
    def __init__(
            self,
            N=3,
            local_ratio=0.5,
            max_cycles=25,
            continuous_actions=False,
            render_mode=None,
    ):
        EzPickle.__init__(
            self, N, local_ratio, max_cycles, continuous_actions, render_mode
        )
        assert (
                0.0 <= local_ratio <= 1.0
        ), "local_ratio is a proportion. Must be between 0 and 1."
        scenario = Scenario()
        world = scenario.make_world(N)
        super().__init__(
            scenario=scenario,
            world=world,
            render_mode=render_mode,
            max_cycles=max_cycles,
            continuous_actions=continuous_actions,
            local_ratio=local_ratio,
        )
        self.metadata["name"] = "simple_spread_drone_v2"


env = make_env(raw_env)
parallel_env = parallel_wrapper_fn(env)


class Scenario(BaseScenario):
    def make_world(self, N=3):
        world = World()
        # set any world properties first
        world.dim_c = 2
        num_agents = N
        num_landmarks = N
        world.collaborative = True
        # add agents
        world.agents = [Agent() for i in range(num_agents)]
        for i, agent in enumerate(world.agents):
            agent.name = f"agent_{i}"
            agent.collide = True
            agent.silent = True
            agent.size = 0.1
        # add landmarks
        world.landmarks = [Landmark() for i in range(num_landmarks)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = "landmark %d" % i
            landmark.collide = False
            landmark.movable = False
        return world

    def reset_world(self, world, np_random):
        # random properties for agents
        for i, agent in enumerate(world.agents):
            agent.color = np.array([0.35, 0.35, 0.85])
        # random properties for landmarks
        for i, landmark in enumerate(world.landmarks):
            landmark.color = np.array([0.25, 0.25, 0.25])
        # set random initial states
        for agent in world.agents:
            agent.state.p_pos = np_random.uniform(-1, +1, world.dim_p)
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)
        for i, landmark in enumerate(world.landmarks):
            landmark.state.p_pos = np_random.uniform(-1, +1, world.dim_p)
            landmark.state.p_vel = np.zeros(world.dim_p)
        # if self.reset_landmarks_in_pattern(world):
        #     print("Arranged patterned landmarks")
        #     for l, a in zip(world.landmarks, world.agents):
        #         print(l.state.p_pos, a.state.p_pos)

    def get_min_dists(self, world):
        landmark_to_agent_mindists = []
        agent_to_landmark_mindists = []
        for lm in world.landmarks:
            dists = [
                np.sqrt(np.sum(np.square(a.state.p_pos - lm.state.p_pos)))
                for a in world.agents
            ]
            landmark_to_agent_mindists.append(min(dists))

        for a in world.agents:
            # print("Agent ", a.name, " position: ", a.state.p_pos)
            dists = [
                np.sqrt(np.sum(np.square(a.state.p_pos - lm.state.p_pos)))
                for lm in world.landmarks
            ]
            agent_to_landmark_mindists.append(min(dists))
        return landmark_to_agent_mindists, agent_to_landmark_mindists

    def benchmark_data(self, agent, world):
        rew = 0
        collisions = 0
        occupied_landmarks = 0
        agents_home = 0
        min_dists = 0
        landmark_to_agent, agent_to_landmark = self.get_min_dists(world)
        # Get distance from the closest agent for each landmark
        for d in landmark_to_agent:
            min_dists += d
            if d < 0.1:
                occupied_landmarks += 1
        for d in agent_to_landmark:
            min_dists += d
            if d < 0.1:
                agents_home += 1
        rew -= min_dists

        if agent.collide:
            for a in world.agents:
                if self.is_collision(a, agent):
                    rew -= 1
                    collisions += 1

        if occupied_landmarks == len(world.landmarks) and agents_home == len(world.agents):
            rew += 100

        return rew, collisions, min_dists, min(occupied_landmarks, agents_home)

    def is_collision(self, agent1, agent2):
        delta_pos = agent1.state.p_pos - agent2.state.p_pos
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        dist_min = agent1.size + agent2.size
        #print ("For collision between: ", agent1.name, " and ", agent2.name, ", dist = ", dist, " dist_min = ", dist_min)
        return True if dist < dist_min else False

    def reward(self, agent, world):
        # Agents are rewarded based on minimum agent distance to each landmark, penalized for collisions
        rew = 0
        if agent.collide:
            for a in world.agents:
                if self.is_collision(a, agent):
                    rew -= 1
        return rew

    def global_reward(self, world):
        rew = 0.0
        occupied_landmarks = 0
        agents_home = 0

        # Penalize for the distance between agents and landmarks
        landmark_to_agent, agent_to_landmark = self.get_min_dists(world)
        for d in landmark_to_agent:
            rew -= d
            if d < 0.1:
                occupied_landmarks += 1
        for d in agent_to_landmark:
            rew -= d
            if d < 0.1:
                agents_home += 1

        # reward for finishing the challenge for all landmarks to be occupied by agents
        if occupied_landmarks == len(world.landmarks) and agents_home == len(world.agents):
            rew += 100

        # penalize heavily for collisions
        collisions = set()
        for a1 in world.agents:
            for a2 in world.agents:
                if a1.name != a2.name and self.is_collision(a1, a2):
                    collisions.add(a1.name)
                    collisions.add(a2.name)
        if len(collisions) > 0:
            penalty = 10 * len(collisions)
            rew -= penalty
            #print("Reward penalized by ", penalty, " for collisions = ", collisions)

        return rew

    def get_terminations(self, world):
        terminations = {}
        for a in world.agents:
            terminations[a.name] = False

        occupied_landmarks = 0
        agents_home = 0
        landmark_to_agent, agent_to_landmark = self.get_min_dists(world)
        for d in landmark_to_agent:
            if d < 0.1:
                occupied_landmarks += 1
        for d in agent_to_landmark:
            if d < 0.1:
                agents_home += 1
        #("# Occupied landmarks: ", occupied_landmarks, " # Agents home: ", agents_home)

        if occupied_landmarks == len(world.landmarks) and agents_home == len(world.agents):
            # Mark all agents as done.
            #print("Marking all agents as done!")
            for a in world.agents:
                terminations[a.name] = True

        return terminations

    def get_collisions(self, world):
        collisions = set()
        for a1 in world.agents:
            for a2 in world.agents:
                if a1.name != a2.name and self.is_collision(a1, a2):
                    collisions.add(a1.name)
                    collisions.add(a2.name)
        # if len(collisions) > 0:
        #     print("Collisions = ", collisions)
        return collisions

    def observation(self, agent, world):
        # get positions of all entities in this agent's reference frame
        entity_pos = []
        for entity in world.landmarks:  # world.entities:
            entity_pos.append(entity.state.p_pos - agent.state.p_pos)
        # entity colors
        entity_color = []
        for entity in world.landmarks:  # world.entities:
            entity_color.append(entity.color)
        # communication of all other agents
        comm = []
        other_pos = []
        for other in world.agents:
            if other is agent:
                continue
            comm.append(other.state.c)
            other_pos.append(other.state.p_pos - agent.state.p_pos)
        return np.concatenate(
            [agent.state.p_vel] + [agent.state.p_pos] + entity_pos + other_pos + comm
        )

    def reset_landmarks_in_pattern(self, world):
        # set pre decided initial landmarks and states
        if len(world.agents) == 3 and world.dim_p == 2:
            world.landmarks[0].state.p_pos = np.array([-0.5, -0.5])
            world.landmarks[1].state.p_pos = np.array([0.5, -0.5])
            world.landmarks[2].state.p_pos = np.array([0, 0.5])
            world.agents[0].state.p_pos = np.array([-1.0, -1.0])
            world.agents[1].state.p_pos = np.array([1.0, -1.0])
            world.agents[2].state.p_pos = np.array([1.0, 1.0])
            return True
        return False
