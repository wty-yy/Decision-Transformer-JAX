import os, random, bisect, torch
from torch.utils.data import Dataset, DataLoader
from dopamine.replay_memory import circular_replay_buffer
from typing import List
import numpy as np
from utils import Config
import logging
logging.basicConfig(level=logging.ERROR, format='%(asctime)s - %(levelname)s - %(message)s')

class ReplayConifg(Config):
  observation_shape=(84, 84)
  stack_size=4
  update_horizon=1
  gamma=0.99
  observation_dtype=np.uint8
  batch_size=32
  replay_capacity=100000
replay_config = ReplayConifg()

class FixedReplayBuffer:
  def __init__(self, path_buffer: str, buffer_id: List, **replay_kwargs):
    self.path_buffer = path_buffer
    self.buffer_id = buffer_id
    self.replay_kwargs = replay_kwargs
    self.load_single_buffer(buffer_id)
  
  def load_single_buffer(self, buffer_id):
    self.replay_buffer = circular_replay_buffer.OutOfGraphReplayBuffer(**self.replay_kwargs)
    self.replay_buffer.load(self.path_buffer, buffer_id)
    # print(f"Load replay buffer ckpt {buffer_id} from {self.path_buffer}")
  
  def sample_trainsition_batch(self, batch_size: int, indices: List):
    return self.replay_buffer.sample_transition_batch(batch_size=batch_size, indices=indices)

class DatasetBuilder:
  def __init__(self, path_buffer: int, dataset_step: int, traj_per_buffer: int, seed: int = 42):
    """
    Args:
      - path_buffer: .../game_name/id/replay_logs/{buffer1, buffer2, ...}
      - n_step: The number of steps loads from buffers. (Dataset size = 3 * n_step)
      - traj_per_buffer: The number of trajactories loads from each buffer at once.
    """
    random.seed(seed)
    self.path_buffer, self.dataset_step, self.traj_per_buffer = path_buffer, dataset_step, traj_per_buffer
    self.preload()
  
  def preload(self):
    # obs, action, reward (each step), done_idx, return (each trajectory)
    data = self.data = {'obs': [], 'action': [], 'reward': [], 'done_idx': [], 'return': [], 'timestep': []}
    used_traj_per_buffer = np.zeros(50, dtype=np.int32)
    n_traj = 0
    while len(data['obs']) < self.dataset_step:
      buffer_id = random.randint(0, 50)
      i = used_traj_per_buffer[buffer_id]
      print(f"Loading from buffer {buffer_id} which has {i} already loaded.")
      frb = FixedReplayBuffer(self.path_buffer, buffer_id, **dict(replay_config))
      traj_count, timestep = 0, 0
      pre_step = len(data['obs'])  # number of step for previous trajectory, used for rollback if the last traj is not completely.
      data['return'].append(0)
      while True:
        if i >= 100000:  # buffer max steps, finish, rollback to previous trajectory
          data['obs'] = data['obs'][:pre_step]
          data['action'] = data['action'][:pre_step]
          data['reward'] = data['reward'][:pre_step]
          data['timestep'] = data['timestep'][:pre_step]
          data['return'] = data['return'][:-1]
          break
        s, a, r, s_, a_, r_, terminal, idx = frb.sample_trainsition_batch(batch_size=1, indices=[i])
        data['obs'].append(s[0])  # (84, 84, 4)
        data['action'].append(a[0])
        data['reward'].append(r[0])
        data['timestep'].append(timestep)
        if terminal[0]:
          pre_step = len(data['obs'])
          data['done_idx'].append(len(data['obs'])-1)
          i += 6  # To skip useless step, between two trajectories
          if traj_count == self.traj_per_buffer or len(data['obs']) >= self.dataset_step:  # trajectory count is enough or dataset is full, finish
            break
          else: traj_count += 1
          data['return'].append(0)
          timestep = 0
        data['return'][-1] += r[0]
        i += 1
        timestep += 1
      n_traj += traj_count
      used_traj_per_buffer[buffer_id] = i
      print(f"This buffer has {i} loaded steps and there are now {len(data['obs'])} steps total divided into {n_traj}, average trajectory length {len(data['obs']) / n_traj:.2f}")
    assert len(data['return']) == len(data['done_idx']), "The number of trajectories should be same."
    for k in data.keys(): data[k] = np.array(data[k])  # convert to np.ndarray since they are freezed
    print(f"Max return {max(data['return'])}, max timestep is {max(data['timestep'])}, vocab size {max(data['action'])+1}")
    ### Build return-to-go ###
    st = 0
    rtg = data['rtg'] = np.zeros_like(data['reward'])
    for idx, i in enumerate(data['done_idx']):
      ret = data['return'][idx]
      for j in range(i, st, -1):
        rtg[j] = ret
        ret -= data['reward'][j]
      st = i
    data.pop('reward'); data.pop('return')  # free space
  
  def show_buffer(self):
    import cv2
    cv2.namedWindow('Replay buffer', cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    cv2.resizeWindow('Replay buffer', 5*84, 5*84)
    data = self.data
    cum_reward = 0
    i = np.argmax(data['return'])
    st = 0 if i == 0 else data['done_idx'][i-1] + 1
    for i in range(st, len(data['obs'])):
    # for st in data['done_idx']:
      # for i in range(st-10, st+10):
        cum_reward += data['reward'][i]
        terminal = i in data['done_idx']
        print(f"{i:06}: cum_reward={cum_reward}, reward={data['reward'][i]}, terminal={terminal}, action={data['action'][i]}")
        img = data['obs'][i][...,0]
        cv2.imshow('Replay buffer', img)
        cv2.waitKey(50 + (1000 if terminal else 0))  # 20fps
        # cv2.waitKey(0)
        if terminal: cum_reward = 0
    
  def get_dataset(self, n_token: int, batch_size: int, num_workers: int = 4):  # Only train dataset
    return DataLoader(
      StateActionReturnDataset(self.data, n_token),
      batch_size=batch_size,
      shuffle=True,
      persistent_workers=True,  # GOOD
      num_workers=num_workers,
      drop_last=True,
    )


class StateActionReturnDataset(Dataset):
  def __init__(self, data: dict, n_token: int):
    self.data, self.n_token = data, n_token
    self.n_step = n_token // 3
  
  def __len__(self):
    return len(self.data['obs']) - self.n_step - 1
  
  def __getitem__(self, idx):
    n_step, data = self.n_step, self.data
    done_idx = idx + n_step - 1
    # bisect_left(a, x): if x in a, return left x index, else return index with elem bigger than x
    # minus one for building the target action
    done_idx = min(data['done_idx'][bisect.bisect_left(data['done_idx'], idx)]-1, done_idx)
    idx = done_idx - n_step + 1
    s = data['obs'][idx:done_idx+1].astype(np.float32) / 255.     # (n_step, 84, 84, 4)
    a = data['action'][idx:done_idx+1].astype(np.int32)           # (n_step,)
    target = data['action'][idx+1:done_idx+2].astype(np.int32)    # (n_step,)
    rtg = data['rtg'][idx:done_idx+1].astype(np.float32)          # (n_step,)
    timestep = data['timestep'][idx:done_idx+1].astype(np.int32)  # (n_step,)
    return s, a, rtg, timestep, target
  
if __name__ == '__main__':
  path_buffer = r"/home/yy/Coding/GitHub/decision-transformer/atari/dqn_replay/Breakout/1/replay_logs"
  ds_builder = DatasetBuilder(path_buffer, dataset_step=5000, traj_per_buffer=20)
  # ds_builder.show_buffer()
  torch.multiprocessing.set_start_method('spawn')
  ds = ds_builder.get_dataset(30*3, 128)
  from tqdm import tqdm
  for s, a, rtg, timestep, y in tqdm(ds):
    ...
  for s, a, rtg, timestep, y in tqdm(ds):
    ...
