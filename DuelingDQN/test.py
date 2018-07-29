from __future__ import division

import os
import tensorflow as tf

from VCM_environment import VCMEN
from RL_brain import DuelingDQN

MEMORY_SIZE = 1000
ACTION_SPACE = 8

if __name__ == "__main__":
  env = VCMEN()
  load_model_dir = os.path.join(
      os.path.dirname(os.path.abspath(__file__)), "models")
  agent = DuelingDQN(n_actions=ACTION_SPACE, n_features=144, memory_size=MEMORY_SIZE,
                     environment_name=env.name, dueling=True, load_model_dir=load_model_dir)
  state_t, reward_t, win = env.observe()

  step = 0
  while not win:
    step += 1
    print(state_t)
    # choose
    observation = state_t.flatten()
    action_t = agent.choose_action(observation)
    # act
    env.execute_action(action_t)
    state_t_1, reward_t, win = env.observe()
    state_t = state_t_1

  print(step)