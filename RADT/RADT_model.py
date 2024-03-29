"""
DT params: 2066336
RADT params: 2666912 (+22.5%, transformer size +50%)
"""
import jax, jax.numpy as jnp
import flax.linen as nn
import flax, optax
import numpy as np
from flax.training import train_state
from typing import Callable, Sequence
from utils import Config
from functools import partial

Dense = partial(nn.Dense, kernel_init=nn.initializers.normal(stddev=0.02))
Embed = partial(nn.Embed, embedding_init=nn.initializers.normal(stddev=0.02))

class TrainState(train_state.TrainState):
  dropout_rng: jax.Array

class GPTConfig(Config):
  n_embd = 128
  n_head = 8
  n_block = 6
  p_drop_embd = 0.1
  p_drop_resid = 0.1
  p_drop_attn = 0.1

  def __init__(self, n_vocab, n_token, max_timestep, **kwargs):
    self.n_vocab, self.n_token, self.max_timestep = n_vocab, n_token, max_timestep
    for k, v in kwargs.items():
      setattr(self, k, v)
    assert self.n_embd % self.n_head == 0, "n_embd must be devided by n_head"

class TrainConfig(Config):
  seed = 42
  weight_decay = 0.1
  lr = 6e-4
  total_epochs = 5
  batch_size = 128
  betas = (0.9, 0.95)  # Adamw beta1, beta2
  warmup_tokens = 512*20  # 375e6
  clip_global_norm = 0.1
  lr_fn: Callable

  def __init__(self, steps_per_epoch, n_token, **kwargs):
    self.steps_per_epoch = steps_per_epoch
    self.n_token = n_token
    for k, v in kwargs.items():
      setattr(self, k, v)

class CausalSelfAttention(nn.Module):
  n_embd: int  # NOTE: n_embd % n_head == 0
  n_head: int
  p_drop_attn: float
  p_drop_resid: float

  @nn.compact
  def __call__(self, q: jnp.ndarray, k: jnp.ndarray = None, v: jnp.ndarray = None, train: bool = True):
    assert q is not None, "The q must not be None"
    if k is None and v is None: k = v = q
    elif v is None: v = k
    D = self.n_embd // self.n_head  # hidden dim
    B, L, _ = q.shape  # Bachsize, token length, embedding dim
    mask = jnp.expand_dims(jnp.tri(L), (0, 1))  # Only consider previous token values
    q = Dense(self.n_embd)(q).reshape(B, L, self.n_head, D).transpose(0, 2, 1, 3)
    k = Dense(self.n_embd)(k).reshape(B, L, self.n_head, D).transpose(0, 2, 1, 3)
    v = Dense(self.n_embd)(v).reshape(B, L, self.n_head, D).transpose(0, 2, 1, 3)
    attn = q @ jnp.swapaxes(k, -1, -2) / jnp.sqrt(D)
    attn = jnp.where(mask == 0, -1e18, attn)
    attn = jax.nn.softmax(attn)
    attn = nn.Dropout(self.p_drop_attn)(attn, deterministic=not train)
    y = (attn @ v).transpose(0, 2, 1, 3).reshape(B, L, self.n_embd)
    y = Dense(self.n_embd)(y)
    y = nn.Dropout(self.p_drop_resid)(y, deterministic=not train)
    return y

class CausalCrossAttention(nn.Module):
  causal_self_attention: CausalSelfAttention

  @nn.compact
  def __call__(self, q: jnp.ndarray, k: jnp.ndarray, train: bool = True):
    z = self.causal_self_attention()(q, k, train=train)
    zq = jnp.concatenate([z, q], -1)  # (B, L, 2D)
    alpha = nn.Dense(z.shape[-1], kernel_init=nn.initializers.zeros)(zq)
    return (1 + alpha) * z + q


class AdaLayerNorm(nn.Module):
  @nn.compact
  def __call__(self, sa: jnp.ndarray, rr: jnp.ndarray):
    D = rr.shape[-1]
    w = self.param('w_gamma', lambda _, shape: jnp.zeros(shape), (D,))
    b = self.param('b_gamma', lambda _, shape: jnp.zeros(shape), (D,))
    gamma = w * rr + b
    w = self.param('w_beta', lambda _, shape: jnp.zeros(shape), (D,))
    b = self.param('b_beta', lambda _, shape: jnp.zeros(shape), (D,))
    beta = w * rr + b
    sa = nn.LayerNorm()(sa)
    return (1 + gamma) * sa + beta

class AttentionBlock(nn.Module):
  cfg: GPTConfig

  @nn.compact
  def __call__(self, sa: jnp.ndarray, rr: jnp.ndarray, train: bool = True):
    attn_cfg = {key: getattr(self.cfg, key) for key in ['n_embd', 'n_head', 'p_drop_attn', 'p_drop_resid']}
    csa = partial(CausalSelfAttention, **attn_cfg)
    z = AdaLayerNorm()(sa, rr)
    sa = sa + csa()(z, train=train)
    z = AdaLayerNorm()(sa, rr)
    sa = sa + CausalCrossAttention(causal_self_attention=csa)(z, rr, train=train)
    z = AdaLayerNorm()(sa, rr)
    z = nn.Sequential([
      Dense(4*self.cfg.n_embd), nn.gelu,
      Dense(self.cfg.n_embd),
    ])(z)
    sa = sa + nn.Dropout(self.cfg.p_drop_resid)(z, deterministic=not train)
    return sa

class GPT(nn.Module):
  cfg: GPTConfig

  @nn.compact
  def __call__(self, s, a, rtg, timestep, train: bool):
    cfg = self.cfg
    B, l = rtg.shape
    assert cfg.n_token == l * 2, "The n_token should be 2 * n_step"
    ### Embedding ###
    rtg = nn.tanh(Dense(cfg.n_embd)(jnp.expand_dims(rtg, -1)))  # (B, l) -> (B, l, N_e)
    s = nn.Sequential([  # (B, l, 84, 84, 4) -> (B, l, N_e)
      nn.Conv(32, kernel_size=(8, 8), strides=4, padding='VALID'), nn.relu,  # (20, 20, 32)
      nn.Conv(64, kernel_size=(4, 4), strides=2, padding='VALID'), nn.relu,  # (9, 9, 64)
      nn.Conv(64, kernel_size=(3, 3), strides=1, padding='VALID'), nn.relu,  # (7, 7, 64)
      lambda x: jnp.reshape(x, (B, l, -1)),
      Dense(cfg.n_embd), nn.tanh
    ])(s)
    a = nn.tanh(Embed(cfg.n_vocab, cfg.n_embd)(a))  # (B, l) -> (B, l, N_e)
    time_embd = nn.Embed(cfg.max_timestep+1, cfg.n_embd, embedding_init=nn.initializers.zeros)(timestep)  # (B, l) -> (B, l, N_e)
    pos_embd = nn.Embed(cfg.n_token, cfg.n_embd, embedding_init=nn.initializers.zeros)(jnp.arange(cfg.n_token))  # (1, L, N_e)
    ### Build Token ###
    s, a = s.transpose(1, 0, 2), a.transpose(1, 0, 2)  # (B, l, N_e) -> (l, B, N_e)
    ### the last output is last action, so it's useless ###
    def stack(_, xs):  # stack xs elems in sequentially
      return _, jnp.stack([xs[0], xs[1]])
    sa = jax.lax.scan(stack, None, [s, a])[1].reshape(cfg.n_token, B, cfg.n_embd).transpose(1, 0, 2)  # (B, L, N_e)
    sa = sa + pos_embd + time_embd.repeat(2, 1)  # (B, L, N_e)
    rr = rtg.repeat(2, 1)
    ### GPT-1 ###
    sa = nn.Dropout(cfg.p_drop_embd)(sa, deterministic=not train)
    for _ in range(cfg.n_block):
      sa = AttentionBlock(cfg)(sa, rr, train=train)
    sa = nn.LayerNorm()(sa)
    sa = Dense(cfg.n_vocab, use_bias=False)(sa)
    return sa
    
  def get_state(self, train_cfg: TrainConfig, verbose: bool = False, load_path: str = None, train: bool = True) -> TrainState:
    def check_decay_params(kp, x):
      fg = x.ndim > 1
      for k in kp:
        if k.key in ['LayerNorm', 'Embed']:
          fg = False; break
      return fg
    def lr_fn():
      warmup_steps = train_cfg.warmup_tokens // (train_cfg.n_token * train_cfg.batch_size)
      warmup_fn = optax.linear_schedule(0.0, train_cfg.lr, warmup_steps)
      second_steps = max(train_cfg.total_epochs * train_cfg.steps_per_epoch - warmup_steps, 1)
      second_fn = optax.cosine_decay_schedule(
        train_cfg.lr, second_steps, 0.1
      )
      return optax.join_schedules(
        schedules=[warmup_fn, second_fn],
        boundaries=[warmup_steps]
      )
    rng = jax.random.PRNGKey(train_cfg.seed)
    if not train:  # return state with apply function
      return TrainState.create(apply_fn=self.apply, params={'a': 1}, tx=optax.sgd(1), dropout_rng=rng)
    # s, a, rtg, timestep
    B, l = train_cfg.batch_size, self.cfg.n_token // 2
    s, a, rtg, timestep = jnp.empty((B, l, 84, 84, 4), float), jnp.empty((B, l), int), jnp.empty((B, l), float), jnp.empty((B, l), int)
    examp = (s, a, rtg, timestep)
    if verbose: print(self.tabulate(rng, *examp, train=False))
    variables = self.init(rng, *examp, train=False)
    print("mini-GPT params:", sum([np.prod(x.shape) for x in jax.tree_util.tree_flatten(variables)[0]]))
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
      dropout_rng=rng,
    )
    if load_path is not None:
      with open(load_path, 'rb') as file:
        state = flax.serialization.from_bytes(state, file.read())
      print(f"Load weights from {load_path}")
    return state
  
  def create_fns(self):
    def model_step(state: TrainState, s, a, rtg, timestep, y, train: bool):
      dropout_rng, base_rng = jax.random.split(state.dropout_rng)
      def loss_fn(params):
        logits = state.apply_fn({'params': params}, s, a, rtg, timestep, train=train, rngs={'dropout': dropout_rng})
        logits = logits[:, ::2, :]  # (B, l, N_e)
        tmp = -jax.nn.log_softmax(logits).reshape(-1, logits.shape[-1])
        loss = tmp[jnp.arange(tmp.shape[0]), y.reshape(-1)].mean()
        acc = (jnp.argmax(logits, -1).reshape(-1) == y.reshape(-1)).mean()
        return loss, acc
      (loss, acc), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
      state = state.apply_gradients(grads=grads)
      state = state.replace(dropout_rng=base_rng)
      return state, (loss, acc)
    self.model_step = jax.jit(model_step, static_argnames='train')

    def predict(state: TrainState, s, a, rtg, timestep, step_len: Sequence[int] = None, rng: jax.Array = None, deterministic: bool = False):
      # print(s.shape, a.shape, rtg.shape, timestep.shape, mask_len.shape)
      logits = state.apply_fn({'params': state.params}, s, a, rtg, timestep, train=False)
      if step_len is not None:
        logits = logits[jnp.arange(logits.shape[0]), 2*step_len-2, :]  # (B, n_vocab)
      if deterministic:
        pred = jnp.argmax(logits, -1)
      else:
        pred = jax.random.categorical(rng, logits, -1)
      return pred
    self.predict = jax.jit(predict, static_argnames='deterministic')

  def save_model(self, state, save_path):
    with open(save_path, 'wb') as file:
      file.write(flax.serialization.to_bytes(state))
    print(f"Save weights to {save_path}")
  
if __name__ == '__main__':
  batch_size = 128
  n_vocab = 4
  n_token = 60
  n_embd = 128
  n_head = 8
  n_block = 6
  max_timestep = 3000
  # Total Parameters: 1,938,852 (7.8 MB)
  gpt_cfg = GPTConfig(n_vocab, n_token, max_timestep=max_timestep, n_embd=n_embd, n_head=n_head, n_block=n_block)
  print(dict(gpt_cfg))
  gpt = GPT(gpt_cfg)
  # rng = jax.random.PRNGKey(42)
  # x = jax.random.randint(rng, (batch_size, n_len), 0, 6)
  # print(gpt.tabulate(rng, x, train=False))
  # variable = gpt.init(rng, x, train=False)
  # print("params:", sum([np.prod(x.shape) for x in jax.tree_util.tree_flatten(variable)[0]]))
  train_cfg = TrainConfig(steps_per_epoch=512, n_token=n_token)
  state = gpt.get_state(train_cfg, verbose=True)
