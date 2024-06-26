"""
DT params: 2066336
RADT params: 2666912 (DT+22.5%, transformer size +50%)
StAR params: 14,370,080 (57.5 MB) (DT+695.4%)
"""
# import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import jax, jax.numpy as jnp
import flax.linen as nn
import flax, optax
import numpy as np
from flax.training import train_state
from typing import Callable, Sequence
from utils import Config
from functools import partial
from einops import rearrange

Dense = partial(nn.Dense, kernel_init=nn.initializers.normal(stddev=0.02))
Embed = partial(nn.Embed, embedding_init=nn.initializers.normal(stddev=0.02))

class TrainState(train_state.TrainState):
  dropout_rng: jax.Array

class StARConfig(Config):
  n_embd_global = 192
  n_head_global = 8
  n_embd_local = 64
  n_head_local = 4
  n_block = 6  # StARBlock number
  p_drop_embd = 0.1
  p_drop_resid = 0.1
  p_drop_attn = 0.1
  mode = 'star'  # 'star' + ['_reward'] + ['_timestep']

  def __init__(self, obs_dim, act_dim, n_step, max_timestep, **kwargs):
    self.obs_dim, self.act_dim, self.n_step, self.max_timestep = obs_dim, act_dim, n_step, max_timestep
    for k, v in kwargs.items():
      setattr(self, k, v)
    assert self.n_embd_global % self.n_head_global == 0, "n_embd_global must be devided by n_head_global"
    assert self.n_embd_local % self.n_head_local == 0, "n_embd_local must be devided by n_head_local"

class TrainConfig(Config):
  seed = 42
  weight_decay = 0.1
  lr = 6e-4
  total_epochs = 10
  batch_size = 64
  betas = (0.9, 0.95)  # Adamw beta1, beta2
  warmup_tokens = 512*20  # 375e6
  clip_global_norm = 1.0
  lr_fn: Callable

  def __init__(self, steps_per_epoch, n_step, **kwargs):
    self.steps_per_epoch = steps_per_epoch
    self.n_step = n_step
    for k, v in kwargs.items():
      setattr(self, k, v)

class CausalSelfAttention(nn.Module):
  n_embd: int  # NOTE: n_embd % n_head == 0
  n_head: int
  cfg: StARConfig

  @nn.compact
  def __call__(self, q, k = None, v = None, mask = None, train = True):
    assert q is not None, "The q must not be None"
    if k is None and v is None: k = v = q
    elif v is None: v = k
    D = self.n_embd // self.n_head  # hidden dim
    B, L, _ = q.shape  # Bachsize, token length, embedding dim
    if mask is not None:
      if mask.ndim == 2: mask = rearrange(mask, 'h w -> 1 1 h w')
      elif mask.ndim == 3: mask = rearrange(mask, 'b h w -> b 1 h w')
    q = Dense(self.n_embd)(q).reshape(B, L, self.n_head, D).transpose(0, 2, 1, 3)
    k = Dense(self.n_embd)(k).reshape(B, L, self.n_head, D).transpose(0, 2, 1, 3)
    v = Dense(self.n_embd)(v).reshape(B, L, self.n_head, D).transpose(0, 2, 1, 3)
    attn = q @ jnp.swapaxes(k, -1, -2) / jnp.sqrt(D)
    if mask is not None:
      attn = jnp.where(mask == 0, -1e18, attn)
    attn = jax.nn.softmax(attn)
    attn = nn.Dropout(self.cfg.p_drop_attn)(attn, deterministic=not train)
    y = (attn @ v).transpose(0, 2, 1, 3).reshape(B, L, self.n_embd)
    y = Dense(self.n_embd)(y)
    y = nn.Dropout(self.cfg.p_drop_resid)(y, deterministic=not train)
    return y

class TransformerBlock(nn.Module):
  n_embd: int
  n_head: int
  cfg: StARConfig

  @nn.compact
  def __call__(self, x, mask = None, train = True):
    csa = CausalSelfAttention(self.n_embd, self.n_head, self.cfg)
    x = x + csa(nn.LayerNorm()(x), mask=mask, train=train)
    dropout = nn.Dropout(self.cfg.p_drop_resid)
    mlp = nn.Sequential([
      Dense(4*self.n_embd), nn.gelu,
      Dense(self.n_embd)
    ])
    x = x + dropout(mlp(nn.LayerNorm()(x)), deterministic=not train)
    return x

class StARBlock(nn.Module):
  cfg: StARConfig

  @nn.compact
  def __call__(self, xl, xg, train = True):
    local_block = TransformerBlock(n_embd=self.cfg.n_embd_local, n_head=self.cfg.n_head_local, cfg=self.cfg)
    global_block = TransformerBlock(n_embd=self.cfg.n_embd_global, n_head=self.cfg.n_head_global, cfg=self.cfg)
    B, N, M, Dl = xl.shape  # Batch, Step Length, Group Token Length, n_embd_local
    B, N, Dg = xg.shape  # Batch, Step Length, n_embd_global
    xl = local_block(xl.reshape(B * N, M, Dl), train=train).reshape(B, N, M, Dl)
    zg = Dense(Dg)(xl.reshape(B, N, M * Dl))
    zg = jnp.concatenate([zg, xg], 2).reshape(B, 2 * N, Dg)
    mask = jnp.tri(2 * N)
    mask = mask.at[jnp.arange(N) * 2, jnp.arange(N) * 2 + 1].set(1)
    zg = global_block(zg, mask=mask, train=train)
    xg = zg[:, 1::2, :]
    return xl, xg

class StARformer(nn.Module):
  cfg: StARConfig

  @nn.compact
  def __call__(self, s, a, r, timestep, train = True):
    cfg = self.cfg
    nl, ng = cfg.n_embd_local, cfg.n_embd_global
    B, N, obs_dim = s.shape  # Batch, Step Length, obs_dim
    assert obs_dim == cfg.obs_dim, f"{obs_dim=} != {cfg.obs_dim=}"
    ### Embedding Global Token ###
    pos_embd = nn.Embed(N, ng, embedding_init=nn.initializers.zeros, name='pos_embd')(jnp.arange(N))  # (1, N, Ng)
    xg = Dense(ng, name='glob_encode')(s) + pos_embd  # (B, N, obs_dim) -> (B, N, Ng)
    ### Embedding Local Token ###
    if 'reward' in cfg.mode:
      r = nn.tanh(Dense(nl, name='reward_encode')(jnp.expand_dims(r, -1))).reshape(B, N, 1, nl)  # (B, N) -> (B, N, 1, Nl)
    else:
      r = jnp.zeros((B, N, 0, nl))
    # NOTE: Add `n_vocab` as start action
    act_embd = nn.Embed(cfg.act_dim, nl, embedding_init=nn.initializers.zeros, name='act_emdb')(jnp.arange(cfg.act_dim))
    a = Dense(nl, name='act_encode')(jnp.expand_dims(a, -1)) + act_embd  # (B, N, n_vocab, 1) -> (B, N, n_vocab, Nl)
    obs_embd = nn.Embed(cfg.obs_dim, nl, embedding_init=nn.initializers.zeros, name='obs_embd')(jnp.arange(cfg.obs_dim))  # (obs_dim, Nl)
    s = Dense(nl, name='obs_encode')(jnp.expand_dims(s, -1)) + obs_embd  # (B, N, obs_dim, Nl)
    ### Concatenate Group ###
    xl = jnp.concatenate([a, s, r], 2)  # (B, N, obs_dim + n_vocab + (1), Nl)
    if 'timestep' in cfg.mode:
      time_embd = nn.Embed(cfg.max_timestep+1, nl, embedding_init=nn.initializers.zeros, name='time_embd')(timestep).reshape(B, N, 1, nl)  # (B, N) -> (B, N, 1, Nl)
      xl = xl + time_embd.repeat(xl.shape[2], 2)
    ### StARformer ###
    xl = nn.Dropout(cfg.p_drop_embd)(xl, deterministic=not train)
    xg = nn.Dropout(cfg.p_drop_embd)(xg, deterministic=not train)
    for _ in range(cfg.n_block):
      xl, xg = StARBlock(cfg=self.cfg)(xl, xg, train)
    xg = nn.LayerNorm()(xg)
    xg = Dense(cfg.act_dim*2)(xg)
    xg = xg.at[..., cfg.act_dim].set(jax.nn.tanh(xg[..., cfg.act_dim]))
    return xg
    
  def get_state(self, train_cfg: TrainConfig, verbose: bool = False, load_path: str = None, train: bool = True) -> TrainState:
    def check_decay_params(kp, x):
      fg = x.ndim > 1
      for k in kp:
        if k.key in ['LayerNorm', 'Embed']:
          fg = False; break
      return fg
    def lr_fn():
      warmup_steps = train_cfg.warmup_tokens // (train_cfg.n_step * train_cfg.batch_size)
      warmup_fn = optax.linear_schedule(0.0, train_cfg.lr, warmup_steps)
      second_steps = max(train_cfg.total_epochs * train_cfg.steps_per_epoch - warmup_steps, 1)
      second_fn = optax.cosine_decay_schedule(
        train_cfg.lr, second_steps, 0.1
      )
      return optax.join_schedules(
        schedules=[warmup_fn, second_fn],
        boundaries=[warmup_steps]
      )
    drop_rng, verbose_rng, rng = jax.random.split(jax.random.PRNGKey(train_cfg.seed), 3)
    if not train:  # return state with apply function
      return TrainState.create(apply_fn=self.apply, params={'a': 1}, tx=optax.sgd(1), dropout_rng=rng)
    # s, a, r, timestep
    B, l, obs_dim, act_dim = train_cfg.batch_size, self.cfg.n_step, self.cfg.obs_dim, self.cfg.act_dim
    s, a, r, timestep = jnp.empty((B, l, obs_dim), float), jnp.empty((B, l, act_dim), float), jnp.empty((B, l), float), jnp.empty((B, l), int)
    dummy = (s, a, r, timestep)
    if verbose: print(self.tabulate(verbose_rng, *dummy, train=False))
    variables = self.init(rng, *dummy, train=False)
    print("StARformer params:", sum([np.prod(x.shape) for x in jax.tree_util.tree_flatten(variables)[0]]))
    decay_mask = jax.tree_util.tree_map_with_path(check_decay_params, variables['params'])
    train_cfg.lr_fn = lr_fn()
    state = TrainState.create(
      apply_fn=self.apply,
      params=variables['params'],
      # AdamW is Adam with weight decay
      tx=optax.chain(
        optax.clip_by_global_norm(train_cfg.clip_global_norm),
        optax.adamw(train_cfg.lr_fn, train_cfg.betas[0], train_cfg.betas[1], weight_decay=train_cfg.weight_decay, mask=decay_mask),
      ),
      dropout_rng=drop_rng,
    )
    if load_path is not None:
      with open(load_path, 'rb') as file:
        state = flax.serialization.from_bytes(state, file.read())
      print(f"Load weights from {load_path}")
    return state
  
  def create_fns(self):
    def model_step(state: TrainState, s, a, r, timestep, y, train: bool):
      dropout_rng, norm_rng, base_rng = jax.random.split(state.dropout_rng, 3)
      def loss_fn(params):
        # (B, l, n_embd_global)
        logits = state.apply_fn({'params': params}, s, a, r, timestep, train=train, rngs={'dropout': dropout_rng})
        mu, logsigma = jnp.array_split(logits, 2, -1)
        z = jax.random.normal(norm_rng, mu.shape)
        pred = z * jnp.exp(logsigma) + mu
        loss = ((pred - y) ** 2).mean()
        return loss
      loss, grads = jax.value_and_grad(loss_fn)(state.params)
      state = state.apply_gradients(grads=grads)
      state = state.replace(dropout_rng=base_rng)
      return state, loss
    self.model_step = jax.jit(model_step, static_argnames='train')

    def predict(state: TrainState, s, a, r, timestep, step_len: Sequence[int] = None, rng: jax.Array = None, deterministic: bool = False):
      logits = state.apply_fn({'params': state.params}, s, a, r, timestep, train=False)
      if step_len is not None:
        logits = logits[jnp.arange(logits.shape[0]), step_len-1, :]  # (B, n_vocab*2)
      mu, logsigma = jnp.array_split(logits, 2, -1)
      if deterministic:
        pred = mu
      else:
        z = jax.random.normal(rng, mu.shape)
        pred = np.clip(z * jnp.exp(logsigma) + mu, -1, 1)
      return pred
    self.predict = jax.jit(predict, static_argnames='deterministic')

  def save_model(self, state, save_path):
    with open(save_path, 'wb') as file:
      file.write(flax.serialization.to_bytes(state))
    print(f"Save weights to {save_path}")
  
if __name__ == '__main__':
  obs_dim = 17
  act_dim = 6
  n_step = 30
  max_timestep = 1000
  mode = 'star' # Total Parameters (d4rl): 4,679,628 (18.7 MB) <-> Total Parameters (atari): 14,370,080 (57.5 MB)
  mode = 'star_reward'  # Total Parameters (d4rl): 4,753,484 (19.0 MB) <-> Total Parameters (atari): 14,443,936 (57.8 MB)
  mode = 'star_reward_timestep'  # Total Parameters (d4rl): 4,817,548 (19.3 MB) <-> Total Parameters (atari): 14,636,000 (58.5 MB)
  model_cfg = StARConfig(obs_dim=obs_dim, act_dim=act_dim, n_step=n_step, max_timestep=max_timestep, mode=mode)
  print(dict(model_cfg))
  model = StARformer(model_cfg)
  # rng = jax.random.PRNGKey(42)
  # x = jax.random.randint(rng, (batch_size, n_len), 0, 6)
  # print(gpt.tabulate(rng, x, train=False))
  # variable = gpt.init(rng, x, train=False)
  # print("params:", sum([np.prod(x.shape) for x in jax.tree_util.tree_flatten(variable)[0]]))
  train_cfg = TrainConfig(steps_per_epoch=512, n_step=n_step)
  state = model.get_state(train_cfg, verbose=True)
