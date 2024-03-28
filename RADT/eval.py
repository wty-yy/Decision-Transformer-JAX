from atari.atari_gymnasium import Env
from RADT_model import GPT
from flax.training import train_state
import cv2, jax
import numpy as np

class Evaluator:
  def __init__(self, model: GPT, game: str, seed: int = 42, auto_shoot: bool = True,
    show: bool = False, path_video_save_dir: str = None
  ):
    self.model = model
    self.rng = jax.random.PRNGKey(seed)
    self.n_step = self.model.cfg.n_token // 2
    self.env = Env(game=game, seed=seed, auto_shoot=auto_shoot, show=show, path_video_save_dir=path_video_save_dir)
  
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
    step_len = min(len(self.s), n_step)
    rng, self.rng = jax.random.split(self.rng)
    action = self.model.predict(
      self.state,
      pad(self.s).astype(np.float32) / 255.,
      pad(self.a).astype(np.int32),
      pad(self.rtg).astype(np.float32),
      pad(self.timestep).astype(np.int32),
      step_len, rng, self.deterministic)[0]
    return action
  
  def __call__(self, state: train_state.TrainState, n_test: int = 10, rtg: int = 90, deterministic=False):
    self.state, self.deterministic = state, deterministic
    ret, score = [], []
    for i in range(n_test):
      ret.append(0); score.append(0)
      s, _ = self.env.reset()
      done, timestep = False, 0
      self.s, self.a, self.rtg, self.timestep = [s], [0], [rtg], [0]
      while not done:
        a = self.get_action()
        s, r, t1, t2, _ = self.env.step(a)
        done = t1 | t2
        self.s.append(s)
        self.a[-1] = a; self.a.append(0)  # keep s, a, r in same length, but last action is padding
        self.rtg.append(max(self.rtg[-1] - int(r > 0), 1))
        # self.rtg.append(max(self.rtg[-1] - r, 1))
        timestep = min(timestep + 1, self.model.cfg.max_timestep - 1)
        self.timestep.append(timestep)
        ret[-1] += int(r > 0); score[-1] += r
      print(f"epoch {i} with result {ret[-1]}, score {score[-1]}, timestep {len(self.s)}")
    return ret, score

from ckpt_manager import CheckpointManager
from RADT_model import GPTConfig, GPT, TrainConfig
class LoadToEvaluate:
  def __init__(self, path_weights, load_step, auto_shoot: bool = True, show: bool = False, path_video_save_dir: str = None):
    ckpt_mngr = CheckpointManager(path_weights)
    load_info = ckpt_mngr.restore(load_step)
    params, cfg = load_info['params'], load_info['config']
    self.model = GPT(cfg=GPTConfig(**cfg))
    self.model.create_fns()
    state = self.model.get_state(TrainConfig(**cfg), train=False)
    self.state = state.replace(params=params, tx=None, opt_state=None)
    self.evaluator = Evaluator(self.model, game=cfg['game'], seed=cfg['seed'], auto_shoot=auto_shoot, show=show, path_video_save_dir=path_video_save_dir)
  
  def evaluate(self, n_test: int = 10, rtg: int = 90, deterministic: bool = True):
    result = self.evaluator(self.state, n_test=n_test, rtg=rtg, deterministic=deterministic)
    return result

if __name__ == '__main__':
  path_weights = r"../logs/RADT_wty_v0.2__Breakout__2__20240325_210821/ckpt"
  path_video_save_dir = r"../logs/eval_videos"
  load_step = 5
  lte = LoadToEvaluate(path_weights, load_step, show=False, path_video_save_dir=path_video_save_dir)
  ret, score = lte.evaluate(n_test=10, rtg=90, deterministic=False)
  print(ret, score)
  print(np.mean(ret), np.mean(score))  # avg score: 64.5
