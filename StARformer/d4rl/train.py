# import os
# os.environ['XLA_PYTHON_CLIENT_ALLOCATOR'] = 'platform'  # allocate GPU memory as needed
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parents[1]))
from starformer_model import StARConfig, TrainConfig, StARformer
from parse_and_writer import parse_args_and_writer, logs
from utils.ckpt_manager import CheckpointManager
from tqdm import tqdm
from eval import Evaluator
import numpy as np
from d4rl_data.dataset import DatasetBuilder, game2rtg

def train():
  ### Parse augment and TF Writer ###
  args, writer = parse_args_and_writer()
  ### Dataset ###
  ds_builder = DatasetBuilder(args.game, args.seed)
  train_ds = ds_builder.get_dataset(args.n_step, args.batch_size, args.num_workers)
  args.act_dim, args.max_timestep = ds_builder.data['action'].shape[1], int(max(ds_builder.data['timestep']))  # since we must get last idx value
  args.obs_dim = ds_builder.data['obs'].shape[1]
  args.steps_per_epoch = len(train_ds)
  ### Model ###
  gpt_cfg = StARConfig(**vars(args))
  model = StARformer(cfg=gpt_cfg)
  model.create_fns()
  train_cfg = TrainConfig(**vars(args))
  state = model.get_state(train_cfg=train_cfg, verbose=False)
  ### Checkpoint ###
  ckpt_manager = CheckpointManager(str(args.path_logs / 'ckpt'))
  write_tfboard_freq = min(100, len(train_ds))
  ### Evaluator ###
  evaluator = Evaluator(model, game=args.game, seed=args.seed)

  ### Train and Evaluate ###
  for ep in range(args.total_epochs):
    print(f"Epoch: {ep+1}/{args.total_epochs}")
    print("Training...")
    logs.reset()
    bar = tqdm(train_ds, ncols=80)
    for s, y, rtg, timestep in bar:
      s, y, rtg, timestep = s.numpy(), y.numpy(), rtg.numpy(), timestep.numpy()
      # Look out the target is diff with input action, we need a `n_vocab` as start action padding idx.
      a = np.concatenate([np.zeros((y.shape[0], 1, args.act_dim), np.float32), y[:,:-1,:]], 1)
      state, loss = model.model_step(state, s, a, rtg, timestep, y, train=True)
      logs.update(['train_loss'], [loss])
      bar.set_description(f"loss={loss:.4f}")
      if state.step % write_tfboard_freq == 0:
        logs.update(
          ['SPS', 'epoch', 'learning_rate'],
          [write_tfboard_freq / logs.get_time_length(), ep+1, train_cfg.lr_fn(state.step)]
        )
        logs.writer_tensorboard(writer, state.step)
        logs.reset()
    print("Evaluating...")
    ret, score = evaluator(state, n_test=3, rtg=game2rtg[args.game], deterministic=True)
    print(f"Mean eval return: {np.mean(ret):.1f}, Mean eval score: {np.mean(score):.1f}")
    logs.update(['eval_return', 'eval_score', 'epoch'], [np.mean(ret), np.mean(score), ep+1])
    logs.writer_tensorboard(writer, state.step)
    ckpt_manager.save(ep+1, state, vars(args))
  ckpt_manager.close()
  writer.close()
  if args.wandb:
    import wandb
    wandb.finish()

if __name__ == '__main__':
  train()
