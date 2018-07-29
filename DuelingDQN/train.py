import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from RL_brain import DuelingDQN
from VCM_environment import VCMEN

N_EPOCHS = 1500
MEMORY_SIZE = 500
ACTION_SPACE = 8

if __name__ == "__main__":

  env = VCMEN()
  agent = DuelingDQN(n_actions=ACTION_SPACE, n_features=144, memory_size=MEMORY_SIZE,
                      environment_name=env.name, e_greedy_increment=0.01, dueling=True)

  win_cnt = 0
  acc_r = [0]
  for foo in range(N_EPOCHS):
    step = 0
    env.reset()
    state_t, reward_t, win = env.observe()
    while True:
      step += 1
      # choose
      observation = state_t.flatten()
      action_t = agent.choose_action(observation)
      # act
      env.execute_action(action_t)
      # next stat
      state_t_1, reward_t, win = env.observe()
      observation_ = state_t_1.flatten()
      agent.store_transition(observation, action_t, reward_t, observation_)
      if foo > MEMORY_SIZE:
        agent.learn()

      acc_r.append(reward_t + acc_r[-1])
      if win == True:
        win_cnt += 1
        break
      if step > 400:
        break
      state_t = state_t_1
    print("EPOCH:{:03d}/{:03d} | WIN:{:03d} | STEP: {:03d}".format(foo,
                                                                   N_EPOCHS - 1, win_cnt, step))

  plt.figure(1)
  plt.plot(np.array(agent.cost_his), c='b', label='dueling')
  plt.legend(loc='best')
  plt.ylabel('cost')
  plt.xlabel('training steps')
  plt.grid()

  plt.figure(2)
  plt.plot(np.array(acc_r), c='b', label='dueling')
  plt.legend(loc='best')
  plt.ylabel('accumulated reward')
  plt.xlabel('training steps')
  plt.grid()

  plt.show()

  agent.save_model()