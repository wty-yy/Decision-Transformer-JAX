import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parents[1]))
from dt_model import GPTConfig, TrainConfig, GPT
from parse_and_writer import parse_args_and_writer, logs
from dataset import DatasetBuilder
from ckpt_manager import CheckpointManager
from tqdm import tqdm

def train():
  ### Parse augment and TF Writer ###
  args, writer = parse_args_and_writer()
  ### Dataset ###
  ds_builder = DatasetBuilder(args.path_buffer, args.dataset_step, args.traj_per_buffer, args.seed)
  train_ds = ds_builder.get_dataset(args.n_token, args.batch_size, args.num_workers)
  args.n_vocab, args.max_timestep = int(max(ds_builder.data['action'])) + 1, int(max(ds_builder.data['timestep'])) + 1  # since we must get last idx value
  args.steps_per_epoch = len(train_ds)
  ### Model ###
  gpt_cfg = GPTConfig(**vars(args))
  gpt = GPT(cfg=gpt_cfg)
  gpt.create_fns()
  train_cfg = TrainConfig(**vars(args))
  state = gpt.get_state(train_cfg=train_cfg, verbose=False)
  ### Checkpoint ###
  ckpt_manager = CheckpointManager(str(args.path_logs / 'ckpt'))
  write_tfboard_freq = min(100, len(train_ds))

  ### Train and Evaluate ###
  for ep in range(args.total_epochs):
    print(f"Epoch: {ep+1}/{args.total_epochs}")
    print("Training...")
    logs.reset()
    bar = tqdm(train_ds, ncols=80)
    for s, a, rtg, timestep, y in bar:
      s, a, rtg, timestep, y = s.numpy(), a.numpy(), rtg.numpy(), timestep.numpy(), y.numpy()
      state, (loss, acc) = gpt.model_step(state, s, a, rtg, timestep, y, train=True)
      logs.update(['train_loss', 'train_acc'], [loss, acc])
      bar.set_description(f"loss={loss:.4f}, acc={acc:.4f}")
      if state.step % write_tfboard_freq == 0:
        logs.update(
          ['SPS', 'epoch', 'learning_rate'],
          [write_tfboard_freq / logs.get_time_length(), ep+1, train_cfg.lr_fn(state.step)]
        )
        logs.writer_tensorboard(writer, state.step)
        logs.reset()
    ckpt_manager.save(ep+1, state, vars(args))
  ckpt_manager.close()
  if args.wandb: wandb.close()

if __name__ == '__main__':
  train()