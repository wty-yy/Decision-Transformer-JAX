from atari_gym import Env
from dt_model import GPT
from flax.training import train_state
import cv2, jax
import numpy as np

class Evaluator:
  def __init__(self, model: GPT, seed: int = 42, game: str = "Breakout"):
    self.model = model
    self.rng = jax.random.PRNGKey(seed)
    self.n_step = self.model.cfg.n_token // 3
    self.env = Env(seed=seed, game=game.lower())
  
  def get_action(self):
    n_step = self.n_step
    def pad(x):
      delta = max(n_step - len(x), 0)
      x = np.stack(x[-n_step:])
      if x.ndim == 1:
        x = x.reshape(1, -1)
        return np.pad(x, ((0, 0), (0, delta)))
      else:
        x = x.reshape(1, -1, 84, 84, 4)
        return np.pad(x, ((0, 0), (0, delta), (0, 0), (0, 0), (0, 0)))
    mask_len = np.array([min(3 * len(self.s) - 1, n_step * 3 - 1)], np.int32)  # the last action is awalys padding
    rng, self.rng = jax.random.split(self.rng)
    action = jax.device_get(self.model.predict(
      self.state,
      pad(self.s).astype(np.float32),
      pad(self.a).astype(np.int32),
      pad(self.rtg).astype(np.float32),
      pad(self.timestep).astype(np.int32),
      mask_len, rng, self.deterministic))[0]
    return action
  
  def __call__(self, state: train_state.TrainState, n_test: int = 10, rtg: int = 90, deterministic=False, show=False):
    if show:
      cv2.namedWindow("game", cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
      cv2.resizeWindow("game", 84*5, 84*5)
    self.state, self.deterministic = state, deterministic
    ret, score = [], []
    for i in range(n_test):
      ret.append(0); score.append(0)
      s = self.env.reset()
      s = np.array(s).transpose(1, 2, 0)
      done, timestep = False, 0
      self.s, self.a, self.rtg, self.timestep = [s], [0], [rtg], [0]
      while not done:
        a = self.get_action()
        s, r, done = self.env.step(a)
        s = np.array(s).transpose(1, 2, 0)
        self.s.append(s)
        self.a[-1] = a; self.a.append(0)  # keep s, a, r in same length, but last action is padding
        self.rtg.append(self.rtg[-1] - int(r > 0))
        timestep = min(timestep + 1, self.model.cfg.max_timestep - 1)
        self.timestep.append(timestep)
        ret[-1] += int(r > 0); score[-1] += r
        if show:
          cv2.imshow("game", s[...,0])
          cv2.waitKey(50)
      print(f"epoch {i} with result {ret[-1]}, score {score[-1]}, timestep {len(self.s)}")
    return ret, score

from ckpt_manager import CheckpointManager
from dt_model import GPTConfig, GPT, TrainConfig
class LoadToEvaluate:
  def __init__(self, path_weights, load_step):
    ckpt_mngr = CheckpointManager(path_weights)
    load_info = ckpt_mngr.restore(load_step)
    params, cfg = load_info['params'], load_info['config']
    self.model = GPT(cfg=GPTConfig(**cfg))
    self.model.create_fns()
    state = self.model.get_state(TrainConfig(**cfg), train=False)
    self.state = state.replace(params=params, tx=None, opt_state=None)
    self.evaluator = Evaluator(self.model, cfg['seed'])
  
  def evaluate(self, n_test: int = 10, rtg: int = 90, deterministic: bool = False, show: bool = False):
    result = self.evaluator(self.state, n_test=n_test, rtg=rtg, deterministic=deterministic, show=show)
    return result

if __name__ == '__main__':
  path_weights = r"/home/yy/Coding/GitHub/Decision-Transformer-JAX/logs/DT__Breakout__0__20240320_165338/ckpt"
  load_step = 5
  lte = LoadToEvaluate(path_weights, load_step)
  print(lte.evaluate(n_test=10, rtg=90, deterministic=False, show=False))
