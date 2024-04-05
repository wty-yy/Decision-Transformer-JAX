from starformer_model import StARformer
from flax.training import train_state
import cv2, jax
import numpy as np
from pathlib import Path
from atari.atari_gymnasium import Env, game2rtg

class Evaluator:
  def __init__(
      self, model: StARformer, game: str, seed: int = 42, auto_shoot: bool = True,
      show: bool = False, path_video_save_dir: str = None,
    ):
    self.model = model
    self.rng = jax.random.PRNGKey(seed)
    self.n_step = self.model.cfg.n_step
    self.n_vocab = self.model.cfg.n_vocab
    self.game = game.lower()
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
      # Use 'n_vocab' as first start action
      self.s, self.a, self.rtg, self.timestep = [s], [self.n_vocab], [rtg], [0]
      while not done:
        a = self.get_action()
        s, r, t1, t2, _ = self.env.step(a)
        done = t1 | t2
        self.s.append(s)
        self.a.append(a)
        if self.game in ['breakout', 'assault']:  # `breakout` reward in dqn-replay is step reward.
          self.rtg.append(max(self.rtg[-1] - int(r > 0), 1))
        else:
          self.rtg.append(max(self.rtg[-1] - r, 1))
        timestep = min(timestep + 1, self.model.cfg.max_timestep - 1)
        self.timestep.append(timestep)
        ret[-1] += int(r > 0); score[-1] += r
      print(f"epoch {self.env.ep} with result {ret[-1]}, score {score[-1]}, timestep {len(self.s)}")
    return ret, score

from utils.ckpt_manager import CheckpointManager
from starformer_model import StARformer, StARConfig, TrainConfig
class LoadToEvaluate:
  def __init__(self, path_weights, load_step, auto_shoot: bool = True, show: bool = False, path_video_save_dir: str = None):
    ckpt_mngr = CheckpointManager(path_weights)
    load_info = ckpt_mngr.restore(load_step)
    params, cfg = load_info['params'], load_info['config']
    self.model = StARformer(cfg=StARConfig(**cfg))
    self.model.create_fns()
    state = self.model.get_state(TrainConfig(**cfg), train=False)
    self.state = state.replace(params=params, tx=None, opt_state=None)
    self.evaluator = Evaluator(self.model, game=cfg['game'], seed=cfg['seed'], auto_shoot=auto_shoot, show=show, path_video_save_dir=path_video_save_dir)
    self.rtg = game2rtg[cfg['game'].lower()]
  
  def evaluate(self, n_test: int = 10, rtg: int = 90, deterministic: bool = False):
    result = self.evaluator(self.state, n_test=n_test, rtg=rtg, deterministic=deterministic)
    return result

if __name__ == '__main__':
  # path_weights = r"../logs/StARformer_JAX__star_reward_timestep__Breakout__0__20240327_011926/ckpt"
  # path_weights = r"../logs/StARformer_JAX__star_reward_timestep__Pong__0__20240329_080059/ckpt"
  # path_weights = r"../logs/StARformer_JAX__star_reward_timestep__Boxing__0__20240329_135845/ckpt"
  # path_weights = r"../logs/StARformer_JAX__star_reward_timestep__Qbert__0__20240329_200118/ckpt"
  # path_weights = r"../logs/StARformer_JAX__star_reward_timestep__Seaquest__0__20240329_201059/ckpt"
  path_weights = r"../logs/StARformer_JAX__star_reward_timestep__Assault__2__20240329_200343/ckpt"
  path_video_save_dir = r"../logs/eval_videos"
  load_step = 10
  lte = LoadToEvaluate(path_weights, load_step, show=False, path_video_save_dir=path_video_save_dir)
  ret, score = lte.evaluate(n_test=10, rtg=lte.rtg, deterministic=False)
  print(ret, score)
  print(np.mean(ret), np.mean(score))
  # Breakout: avg score (no deterministic): 146.6, 127.8, (deterministic): 65.2 (so bad)