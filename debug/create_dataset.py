import numpy as np
from debug.fixed_replay_buffer import FixedReplayBuffer
from torch.utils.data import Dataset, DataLoader

def create_dataset(num_buffers, num_steps, game, data_dir_prefix, trajectories_per_buffer):
    # -- load data from memory (make more efficient)
    obss = []
    actions = []
    returns = [0]
    done_idxs = []
    stepwise_returns = []

    transitions_per_buffer = np.zeros(50, dtype=int)
    num_trajectories = 0
    while len(obss) < num_steps:
        buffer_num = np.random.choice(np.arange(50 - num_buffers, 50), 1)[0]
        i = transitions_per_buffer[buffer_num]
        print('loading from buffer %d which has %d already loaded' % (buffer_num, i))
        frb = FixedReplayBuffer(
            data_dir=data_dir_prefix + game + '/1/replay_logs',
            replay_suffix=buffer_num,
            observation_shape=(84, 84),
            stack_size=4,
            update_horizon=1,
            gamma=0.99,
            observation_dtype=np.uint8,
            batch_size=32,
            replay_capacity=100000)
        if frb._loaded_buffers:
            done = False
            curr_num_transitions = len(obss)
            trajectories_to_load = trajectories_per_buffer
            while not done:
                states, ac, ret, next_states, next_action, next_reward, terminal, indices = frb.sample_transition_batch(batch_size=1, indices=[i])
                states = states.transpose((0, 3, 1, 2))[0] # (1, 84, 84, 4) --> (4, 84, 84)
                obss += [states]
                actions += [ac[0]]
                stepwise_returns += [ret[0]]
                if terminal[0]:
                    done_idxs += [len(obss)]
                    returns += [0]  # BUG: This line should push down
                    if trajectories_to_load == 0:
                        done = True
                    else:
                        trajectories_to_load -= 1
                returns[-1] += ret[0]
                i += 1
                if i >= 100000:
                    obss = obss[:curr_num_transitions]
                    actions = actions[:curr_num_transitions]
                    stepwise_returns = stepwise_returns[:curr_num_transitions]
                    returns[-1] = 0
                    i = transitions_per_buffer[buffer_num]
                    done = True
            num_trajectories += (trajectories_per_buffer - trajectories_to_load)
            transitions_per_buffer[buffer_num] = i
        print('this buffer has %d loaded transitions and there are now %d transitions total divided into %d trajectories' % (i, len(obss), num_trajectories))

    actions = np.array(actions)
    returns = np.array(returns)
    stepwise_returns = np.array(stepwise_returns)
    done_idxs = np.array(done_idxs)

    # -- create reward-to-go dataset
    start_index = 0
    rtg = np.zeros_like(stepwise_returns)
    for i in done_idxs:
        i = int(i)
        curr_traj_returns = stepwise_returns[start_index:i]
        for j in range(i-1, start_index-1, -1): # start from i-1
            rtg_j = curr_traj_returns[j-start_index:i-start_index]
            rtg[j] = sum(rtg_j)
        start_index = i
    print('max rtg is %d' % max(rtg))

    # -- create timestep dataset
    start_index = 0
    # BUG: timestep is wrong for the start and the end
    timesteps = np.zeros(len(actions)+1, dtype=int)
    for i in done_idxs:
        i = int(i)
        timesteps[start_index:i+1] = np.arange(i+1 - start_index)
        start_index = i+1
    print('max timestep is %d' % max(timesteps))
    # timesteps = np.zeros(len(actions), dtype=int)
    # for i in done_idxs:
    #     i = int(i)
    #     timesteps[start_index:i] = np.arange(i - start_index)
    #     start_index = i
    # print('max timestep is %d' % max(timesteps))

    return obss, actions, returns, done_idxs, rtg, timesteps

class StateActionReturnDataset(Dataset):

    def __init__(self, data, block_size, actions, done_idxs, rtgs, timesteps):        
        self.block_size = block_size
        self.vocab_size = max(actions) + 1
        self.data = data
        self.actions = actions
        self.done_idxs = done_idxs
        self.rtgs = rtgs
        self.timesteps = timesteps
    
    def __len__(self):
        return len(self.data) - self.block_size  # BUG: self.block_size // 3 - 1

    def __getitem__(self, idx):
        block_size = self.block_size // 3
        done_idx = idx + block_size
        for i in self.done_idxs:
            if i > idx: # first done_idx greater than idx
                done_idx = min(int(i), done_idx)
                break
        idx = done_idx - block_size
        states = np.array(self.data[idx:done_idx], np.float32).reshape(block_size, -1) # (block_size, 4*84*84)
        states = states / 255.
        actions = np.array(self.actions[idx:done_idx], np.int32)
        rtgs = np.array(self.rtgs[idx:done_idx], np.float32)
        timesteps = np.array(self.timesteps[idx:idx+1], np.int32)  # same timestep for each block is WRONG?
        # timesteps = torch.tensor(self.timesteps[idx:done_idx], dtype=torch.long).unsqueeze(1)  # Try this
        # torch.long == torch.int64, why use difference?

        return states, actions, rtgs, timesteps

def get_dataset(args):
  obss, actions, returns, done_idxs, rtgs, timesteps = create_dataset(50, args.dataset_step, args.game, args.path_buffer_root, args.traj_per_buffer)
  data = StateActionReturnDataset(obss, args.n_token, actions, done_idxs, rtgs, timesteps)
  loader = DataLoader(data, shuffle=True, persistent_workers=True,
                      batch_size=args.batch_size, num_workers=4, drop_last=True)
  return data, loader