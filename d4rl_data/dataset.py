import random, bisect, torch
from torch.utils.data import Dataset, DataLoader
import d4rl, gym
import numpy as np
import pickle
from pathlib import Path

game2rtg = {
  'halfcheetah-medium-expert-v2': 15000
}

class DatasetBuilder:
  def __init__(self, game='halfcheetah-medium-expert-v2', seed=42):
    torch.manual_seed(seed)
    self.game = game
    self.data = None
    self.env = gym.make(self.game)
    self.preload()
  
  def preload(self):
    # obs, action, rtg (each step), done_idx (each trajectory)
    data = self.data = {}
    replay = self.env.get_dataset()
    self.replay = replay
    data['obs'], data['action'] = replay['observations'], replay['actions']
    n = self.datasize = len(data['obs'])
    print(replay.keys())
    data['done_idx'] = np.where(replay['terminals'] | replay['timeouts'])[0]
    ### Build return-to-go ###
    st = -1
    rtg = data['rtg'] = np.zeros((n,), np.float32)
    timestep = data['timestep'] = np.zeros(n, np.int32)
    for i in data['done_idx']:
      for j in range(i, st, -1):
        rtg[j] = replay['rewards'][j] + (0 if j == i else rtg[j+1])
        timestep[j] = j - st
      st = i
    data['info'] = f"\
Max rtg: {max(data['rtg'])}, Max timestep: {max(data['timestep'])},\
Vocab size: {data['action'].shape[1]}, Total steps: {len(data['obs'])}"
    print(data['info'])
  
  def debug(self):
    print(self.env.get_normalized_score(15000) * 100)
    import matplotlib.pyplot as plt
    plt.hist(self.data['rtg'])
    plt.show()
    
  def get_dataset(self, n_step: int, batch_size: int, num_workers: int = 4):  # Only train dataset
    return DataLoader(
      StateActionReturnDataset(self.data, n_step),
      batch_size=batch_size,
      shuffle=True,
      persistent_workers=True,  # GOOD
      num_workers=num_workers,
      drop_last=True,
    )


class StateActionReturnDataset(Dataset):
  def __init__(self, data: dict, n_step: int):
    self.data, self.n_step = data, n_step
  
  def __len__(self):
    return len(self.data['obs']) - self.n_step - 1
  
  def __getitem__(self, idx):
    n_step, data = self.n_step, self.data
    done_idx = idx + n_step - 1
    # bisect_left(a, x): if x in a, return left x index, else return index with elem bigger than x
    # minus one for building the target action
    done_idx = min(data['done_idx'][bisect.bisect_left(data['done_idx'], idx)], done_idx)
    idx = done_idx - n_step + 1
    s = data['obs'][idx:done_idx+1].astype(np.float32)            # (n_step, obs_dim)
    a = data['action'][idx:done_idx+1].astype(np.float32)         # (n_step, n_vocab)
    rtg = data['rtg'][idx:done_idx+1].astype(np.float32)          # (n_step,)
    timestep = data['timestep'][idx:done_idx+1].astype(np.int32)  # (n_step,)
    return s, a, rtg, timestep
  
if __name__ == '__main__':
  ds_builder = DatasetBuilder()
  ds_builder.debug()
  # ds = ds_builder.get_dataset(30, 128)
  # from tqdm import tqdm
  # for s, a, rtg, timestep in tqdm(ds):
  #   print(s.shape, a.shape, rtg.shape, timestep.shape)
  #   break
