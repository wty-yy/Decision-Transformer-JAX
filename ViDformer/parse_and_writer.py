import argparse, time
from tensorboardX.writer import SummaryWriter
from pathlib import Path
from utils.logs import Logs, MeanMetric

def str2bool(x): return x in ['yes', 'y', 'True', '1']
def parse_args_and_writer(input_args=None, with_writer=True) -> tuple[argparse.Namespace, SummaryWriter]:
  parser = argparse.ArgumentParser()
  ### Gobal ###
  parser.add_argument("--name", type=str, default="ViDformer")
  parser.add_argument("--seed", type=int, default=0)
  parser.add_argument("--wandb", type=str2bool, default=False, const=True, nargs='?')
  ### Training ###
  parser.add_argument("--learning-rate", type=float, default=6e-4)
  parser.add_argument("--total-epochs", type=int, default=10)
  parser.add_argument("--batch-size", type=int, default=64)
  ### Model ###
  parser.add_argument("--n-embd-global", type=int, default=192)
  parser.add_argument("--n-head-global", type=int, default=8)
  parser.add_argument("--n-embd-local", type=int, default=64)
  parser.add_argument("--n-head-local", type=int, default=4)
  parser.add_argument("--n-block", type=int, default=6)
  parser.add_argument("--n-step", type=int, default=30)
  parser.add_argument("--patch-size", default=(7, 7))
  parser.add_argument("--weight-decay", type=float, default=1e-1)
  ### Dataset ###
  parser.add_argument("--path-buffer-root", type=str, default="/home/yy/Coding/GitHub/decision-transformer/atari/dqn_replay/")
  parser.add_argument("--game", type=str, default="Breakout")
  parser.add_argument("--dataset-step", type=int, default=500000, help="The number of step samples from replay buffer")
  parser.add_argument("--traj-per-buffer", type=int, default=10, help="The number of trajectory samples from each replay buffer")
  parser.add_argument("--num-workers", type=int, default=4)

  args = parser.parse_args(input_args)
  args.path_buffer = str(Path(args.path_buffer_root) / args.game / "1/replay_logs/")
  assert Path(args.path_buffer).exists(), "The path of replay buffer must exist"
  args.lr = args.learning_rate
  args.patch_size = [int(x) for x in args.patch_size]

  ### Create Path ###
  path_root = Path(__file__).parents[1]
  args.run_name = f"{args.name}__{args.mode}__{args.game}__{args.seed}__{time.strftime(r'%Y%m%d_%H%M%S')}"
  path_logs = path_root / "logs" / args.run_name
  path_logs.mkdir(parents=True, exist_ok=True)
  args.path_logs = path_logs
  if not with_writer:
    return args

  if args.wandb:
    import wandb
    wandb.init(
      project="Decision Transformer",
      sync_tensorboard=True,
      config=vars(args),
      name=args.run_name,
    )
  writer = SummaryWriter(str(path_logs / "tfboard"))
  return args, writer

logs = Logs(
  init_logs={
    'train_loss': MeanMetric(),
    'train_acc': MeanMetric(),
    'eval_return': MeanMetric(),
    'eval_score': MeanMetric(),
    'SPS': MeanMetric(),
    'epoch': 0,
    'learning_rate': MeanMetric(),
  },
  folder2name={
    'Metrics': ['learning_rate', 'SPS', 'epoch'],
    'Charts': ['train_loss', 'train_acc', 'eval_return', 'eval_score'],
  }
)