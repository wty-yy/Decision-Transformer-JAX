import gymnasium as gym
from gymnasium.wrappers import AtariPreprocessing, FrameStack
import cv2
from typing import List, Sequence
from pathlib import Path
import numpy as np

game2name = {
  'breakout': 'BreakoutNoFrameskip-v4',
  'pong': 'PongNoFrameskip-v4',
  'assault': 'AssaultNoFrameskip-v4',
  'boxing': 'BoxingNoFrameskip-v4',
  'qbert': 'QbertNoFrameskip-v4',
  'seaquest': 'SeaquestNoFrameskip-v0',
}
game2rtg = {
  'breakout': 90,
  'pong': 20,
  'assault': 800,
  'boxing': 90,
  'qbert': 14000,
  'seaquest': 1150,
}

class Env:
  def __init__(self, game: str = 'Breakout', seed: int = 42, auto_shoot: bool = True,
      show: bool | Sequence[int] = False, path_video_save_dir: str = None
    ):
    """
    This is a custom wrapper for Gymnasium Env with auto_shoot, show video, save video functions,
    use `reset()` and `step(action)` to interact with it.
    Args:
      - game (str): The name of Atari game.
      - seed (int): The env seed.
      - auto_shoot (bool): (Just for `breakout`) If taggled, the action 1 will act auto 
          when lose one life and reset the env, it can make evaluation faster.
      - show (bool | Sequence[int]): Show video in time or choose episode to show in window.
      - path_video_save_dir (str): The path to save all the episodes, saving format is:
          `path_video_save_dir/episode1.mp4, episode2.mp4, ...`
    """
    self.game = game.lower()
    env = gym.make(game2name[self.game])
    env = AtariPreprocessing(env)  # frame skip 4, scale to (84, 84), turn to gray
    env = FrameStack(env, 4)
    self.env = env
    self.auto_shoot = auto_shoot
    self.show = show
    if show:
      cv2.namedWindow("Game", cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
      cv2.resizeWindow("Game", 84*5, 84*5)
    self.ep = 0
    self.path_video_save_dir = path_video_save_dir
    if self.path_video_save_dir is not None:
      Path(self.path_video_save_dir).mkdir(exist_ok=True)
    self.writer = None
    self.env.reset(seed=seed)
  
  def reset(self):
    self.ep += 1
    ### Video Writer ###
    if self.path_video_save_dir is not None:
      path_video = str(Path(self.path_video_save_dir) / f"episode{self.ep}.mp4")
      self.writer = cv2.VideoWriter(path_video, cv2.VideoWriter_fourcc(*'mp4v'), 30, (84, 84), isColor=False)
    ### Reset Env ###
    done = True
    while done:
      s, info = self.env.reset()
      done = False
      if self.auto_shoot and self.game == 'breakout':
        s, _, t1, t2, info = self.env.step(1)
        done = t1 | t2
      self.lives = info['lives']
    s = np.array(s).transpose(1, 2, 0)
    return s, info
  
  def step(self, action):
    ### Interact Env ###
    s, r, t1, t2, info = self.env.step(action)
    if info['lives'] != self.lives:
      self.lives = info['lives']
      if self.auto_shoot and self.game == 'breakout':
        self.env.step(1)
        # s, r, t1, t2, info = self.env.step(1)
    s = np.array(s).transpose(1, 2, 0)
    ### Show and Write to File ###
    if self.show == True or (isinstance(self.show, list) and self.ep in self.show):
      cv2.imshow("Game", s[...,0])
      cv2.waitKey(33)  # 30.3 fps
    if self.writer is not None:
      self.writer.write(s[...,0])
    if (t1 | t2) and self.writer is not None:
      print(f"Save episode{self.ep} at {Path(self.path_video_save_dir)}/episode{self.ep}.mp4")
      self.writer.release()
    return s, r, t1, t2, info
    
if __name__ == '__main__':
  ### Test Env ###
  env = Env(game='Assault', seed=42, auto_shoot=True, show=True, path_video_save_dir="../logs/eval_videos")
  s, _ = env.reset()
  done = False
  while not done:
    action = env.env.action_space.sample()
    s, r, t1, t2, _ = env.step(action)
    done = t1 | t2
