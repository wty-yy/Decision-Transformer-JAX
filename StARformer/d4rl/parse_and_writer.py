import argparse, time
from tensorboardX.writer import SummaryWriter
from pathlib import Path
from utils.logs import Logs, MeanMetric

def str2bool(x): return x in ['yes', 'y', 'True', '1']
def parse_args_and_writer(input_args=None, with_writer=True) -> tuple[argparse.Namespace, SummaryWriter]:
  parser = argparse.ArgumentParser()
  ### Gobal ###
  parser.add_argument("--name", type=str, default="StARformer_d4rl_JAX")
  parser.add_argument("--seed", type=int, default=0)
  parser.add_argument("--wandb", type=str2bool, default=False, const=True, nargs='?')
  ### Training ###
  parser.add_argument("--learning-rate", type=float, default=6e-4)
  parser.add_argument("--total-epochs", type=int, default=10)
  parser.add_argument("--batch-size", type=int, default=64)
  ### Model ###
  parser.add_argument("--mode", type=str, default="star_reward_timestep")
  parser.add_argument("--n-embd-global", type=int, default=192)
  parser.add_argument("--n-head-global", type=int, default=8)
  parser.add_argument("--n-embd-local", type=int, default=64)
  parser.add_argument("--n-head-local", type=int, default=4)
  parser.add_argument("--n-block", type=int, default=6)
  parser.add_argument("--n-step", type=int, default=30)
  parser.add_argument("--weight-decay", type=float, default=1e-1)
  ### Dataset ###
  parser.add_argument("--game", type=str, default="halfcheetah-medium-expert-v2")
  parser.add_argument("--num-workers", type=int, default=4)

  args = parser.parse_args(input_args)
  args.lr = args.learning_rate

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
    'eval_return': MeanMetric(),
    'eval_score': MeanMetric(),
    'SPS': MeanMetric(),
    'epoch': 0,
    'learning_rate': MeanMetric(),
  },
  folder2name={
    'Metrics': ['learning_rate', 'SPS', 'epoch'],
    'Charts': ['train_loss', 'eval_return', 'eval_score'],
  }
)