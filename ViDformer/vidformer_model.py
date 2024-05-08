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

class ViDConfig(Config):
  n_embd_global = 192
  n_head_global = 8
  n_embd_local = 64
  n_head_local = 4
  n_block = 6  # StARBlock number
  p_drop_embd = 0.1
  p_drop_resid = 0.1
  p_drop_attn = 0.1

  def __init__(self, n_vocab, n_step, max_timestep, patch_size, **kwargs):
    self.n_vocab, self.n_step, self.max_timestep, self.patch_size = n_vocab, n_step, max_timestep, patch_size
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
  cfg: ViDConfig

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
  cfg: ViDConfig

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

class ViDBlock(nn.Module):
  cfg: ViDConfig

  @nn.compact
  def __call__(self, xl, xg, train = True):
    local_block = TransformerBlock(n_embd=self.cfg.n_embd_local, n_head=self.cfg.n_head_local, cfg=self.cfg)
    global_block = TransformerBlock(n_embd=self.cfg.n_embd_global, n_head=self.cfg.n_head_global, cfg=self.cfg)
    B, N, M, Dl = xl.shape  # Batch, Step Length, Group Token Length, n_embd_local
    B, N2, Dg = xg.shape  # Batch, Step Length * 2, n_embd_global
    xl = local_block(xl.reshape(B * N, M, Dl), train=train).reshape(B, N, M, Dl)
    zg = Dense(Dg)(xl.reshape(B, N, M * Dl))
    zg = jnp.concatenate([zg, xg], 1)  # shape=(B, N + N2, Dg)
    mask = jnp.tri(N + N2)
    zg = global_block(zg, mask=mask, train=train)
    return xl, zg

class ViDformer(nn.Module):
  cfg: ViDConfig

  @nn.compact
  def __call__(self, s, a, r, timestep, train = True):
    cfg = self.cfg
    nl, ng = cfg.n_embd_local, cfg.n_embd_global
    B, N, H, W, C = s.shape  # Batch, Step Length, Height, Width, Channel
    ### Embedding Global Token ###
    # pos_embd = nn.Embed(N, ng, embedding_init=nn.initializers.zeros)(jnp.arange(N))  # (1, N, Ng)
    # Action #
    a = Embed(cfg.n_vocab, ng)(a).reshape(B, N, ng)  # (B, N) -> (B, N, Ng)
    # Reward #
    r = nn.tanh(Dense(ng)(jnp.expand_dims(r, -1)))  # (B, N) -> (B, N, Ng)
    time_embd_g = nn.Embed(cfg.max_timestep+1, ng, embedding_init=nn.initializers.zeros)(timestep)  # (B, N) -> (B, N, Ng)
    xg = jnp.concatenate([a, r], 1) + time_embd_g.repeat(2, 1)
    ### Embedding Local Token ###
    # NOTE: Add `n_vocab` as start action
    p1, p2 = self.cfg.patch_size
    s = rearrange(s, 'B N (H p1) (W p2) C -> B N (H W) (p1 p2 C)', p1=p1, p2=p2)
    P = H * W // p1 // p2
    img_embd = nn.Embed(P, nl, embedding_init=nn.initializers.zeros)(jnp.arange(P))  # (P, Nl)
    s = Dense(nl)(s) + img_embd  # (B, N, P, Nl)
    ### Concatenate Group ###
    time_embd = nn.Embed(cfg.max_timestep+1, nl, embedding_init=nn.initializers.zeros)(timestep).reshape(B, N, 1, nl)  # (B, N) -> (B, N, 1, Nl)
    xl = s + time_embd.repeat(s.shape[2], 2)
    ### StARformer ###
    xl = nn.Dropout(cfg.p_drop_embd)(xl, deterministic=not train)
    xg = nn.Dropout(cfg.p_drop_embd)(xg, deterministic=not train)
    for i in range(cfg.n_block):
      xl, zg = ViDBlock(cfg=self.cfg)(xl, xg, train)
      zg = rearrange(zg, 'B (n N) Ng -> B Ng N n', n=3)
      if i != cfg.n_block - 1:  # xg.shape=(B, 2*N, Ng), get action, reward
        xg = zg[...,1:]
      else:  # xg.shape=(B, N, Ng), get state
        xg = zg[...,0:1]
      xg = rearrange(xg, 'B Ng N n -> B (n N) Ng')
    xg = nn.LayerNorm()(xg)
    xg = Dense(cfg.n_vocab, use_bias=False)(xg)
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
    B, l = train_cfg.batch_size, self.cfg.n_step
    s, a, r, timestep = jnp.empty((B, l, 84, 84, 4), float), jnp.empty((B, l), int), jnp.empty((B, l), float), jnp.empty((B, l), int)
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
      dropout_rng, base_rng = jax.random.split(state.dropout_rng)
      def loss_fn(params):
        # (B, l, n_embd_global)
        logits = state.apply_fn({'params': params}, s, a, r, timestep, train=train, rngs={'dropout': dropout_rng})
        tmp = -jax.nn.log_softmax(logits).reshape(-1, logits.shape[-1])
        loss = tmp[jnp.arange(tmp.shape[0]), y.reshape(-1)].mean()
        acc = (jnp.argmax(logits, -1).reshape(-1) == y.reshape(-1)).mean()
        return loss, acc
      (loss, acc), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
      state = state.apply_gradients(grads=grads)
      state = state.replace(dropout_rng=base_rng)
      return state, (loss, acc)
    self.model_step = jax.jit(model_step, static_argnames='train')

    def predict(state: TrainState, s, a, r, timestep, step_len: Sequence[int] = None, rng: jax.Array = None, deterministic: bool = False):
      # print(s.shape, a.shape, rtg.shape, timestep.shape, mask_len.shape)
      logits = state.apply_fn({'params': state.params}, s, a, r, timestep, train=False)
      if step_len is not None:
        logits = logits[jnp.arange(logits.shape[0]), step_len-1, :]  # (B, n_vocab)
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
  n_vocab = 4
  n_step = 30
  max_timestep = 3000
  patch_size = (7, 7)
  # DT: Total Parameters: 1,938,852 (7.8 MB)
  # star(offical, no reward and timestep) parameters: 14373220
  # mode = 'star'  # Total Parameters: 14,370,080 (57.5 MB)
  # mode = 'star_reward'  # Total Parameters: 14,443,936 (57.8 MB)
  # mode = 'star_reward_timestep'  # Total Parameters: 14,636,000 (58.5 MB)
  gpt_cfg = ViDConfig(n_vocab=n_vocab, n_step=n_step, max_timestep=max_timestep, patch_size=patch_size)
  print(dict(gpt_cfg))
  gpt = ViDformer(gpt_cfg)
  # rng = jax.random.PRNGKey(42)
  # x = jax.random.randint(rng, (batch_size, n_len), 0, 6)
  # print(gpt.tabulate(rng, x, train=False))
  # variable = gpt.init(rng, x, train=False)
  # print("params:", sum([np.prod(x.shape) for x in jax.tree_util.tree_flatten(variable)[0]]))
  train_cfg = TrainConfig(steps_per_epoch=512, n_step=n_step)
  state = gpt.get_state(train_cfg, verbose=True)
