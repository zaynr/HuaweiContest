import os
import numpy as np

MAZE_H = 12
MAZE_W = 12

DIST_ROW = 1
DIST_COL = 1

class VCMEN:
  def __init__(self):
    self.name = os.path.splitext(os.path.basename(__file__))[0]
    self.reward = 0
    self.reset()

  def reset(self):
    self.screen = np.zeros((MAZE_H, MAZE_W))
    self.counter = 1
    self.win = False

    # dest
    self.screen[DIST_ROW, DIST_COL] = -2

    # block 1
    self.screen[8, 8] = -1
    self.screen[9, 9] = -1
    self.screen[9, 7] = -1

    # block 2
    self.screen[3, 3] = -1
    self.screen[4, 4] = -1
    self.screen[4, 2] = -1

    self.player_row = 10
    self.player_col = 10

    self.dist = np.sqrt(np.square(self.player_row - DIST_ROW) + np.square(self.player_col - DIST_COL))

  '''
  actions:
    0: up
    1: down
    2: left
    3: right
    4: leftup
    5: leftdown
    6: rightup
    7: rightdown
  '''
  def execute_action(self, action):
    self.counter += 1
    new_row = self.player_row
    new_col = self.player_col
    self.reward = 0

    # move
    if action == 0:
      new_row -= 1
      if new_row < 0:
        # self.reward = -0.5
        return
    if action == 1:
      new_row += 1
      if new_row >= MAZE_H:
        # self.reward = -0.5
        return
    if action == 2:
      new_col -= 1
      if new_col < 0:
        # self.reward = -0.5
        return
    if action == 3:
      new_col += 1
      if new_col >= MAZE_W:
        # self.reward = -0.5
        return
    if action == 4:
      new_col -= 1
      if new_col < 0:
        # self.reward = -0.5
        return
      new_row -= 1
      if new_row < 0:
        # self.reward = -0.5
        return
    if action == 5:
      new_col -= 1
      if new_col < 0:
        # self.reward = -0.5
        return
      new_row += 1
      if new_row >= MAZE_H:
        # self.reward = -0.5
        return
    if action == 6:
      new_col += 1
      if new_col >= MAZE_W:
        # self.reward = -0.5
        return
      new_row -= 1
      if new_row < 0:
        # self.reward = -0.5
        return
    if action == 7:
      new_col += 1
      if new_col >= MAZE_W:
        # self.reward = -0.5
        return
      new_row += 1
      if new_row >= MAZE_H:
        # self.reward = -0.5
        return

    if self.screen[new_row, new_col] == -2:
      # self.reward = 0.9
      self.win = True
      return
    elif self.screen[new_row, new_col] == -1:
      self.reward = -0.9
      return
    elif self.screen[new_row, new_col] == 1:
      self.reward = -0.3

    new_dist = np.sqrt(np.square(new_row - DIST_ROW) + np.square(new_col - DIST_COL))
    self.reward += 1 - new_dist / self.dist

    self.screen[self.player_row, self.player_col] = 1
    self.dist = new_dist
    self.player_row = new_row
    self.player_col = new_col

  def observe(self):
    return self.screen, self.reward, self.win
