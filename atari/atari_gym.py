import atari_py, cv2
import numpy as np
from collections import deque
import random

class Env():
  def __init__(self, stack_length: int = 4, seed: int = 42, max_episode_length: int = 108e3, game: str = 'breakout'):
    self.ale = atari_py.ALEInterface()
    self.ale.setInt('random_seed', seed)
    self.ale.setInt('max_num_frames_per_episode', max_episode_length)
    self.ale.setFloat('repeat_action_probability', 0)  # Disable sticky actions
    self.ale.setInt('frame_skip', 0)
    self.ale.setBool('color_averaging', False)
    self.ale.loadROM(atari_py.get_game_path(game))  # ROM loading must be done after setting options
    actions = self.ale.getMinimalActionSet()
    self.actions = dict([i, e] for i, e in zip(range(len(actions)), actions))
    self.lives = 0  # Life counter (used in DeepMind training)
    self.life_termination = False  # Used to check if resetting only from loss of life
    self.window = stack_length  # Number of frames to concatenate
    self.state_buffer = deque([], maxlen=stack_length)
    self.done_with_life_loss = False  # Consistent with model training mode

  def _get_state(self):
    state = cv2.resize(self.ale.getScreenGrayscale(), (84, 84), interpolation=cv2.INTER_LINEAR)
    state = np.array(state, np.float32) / 255.
    return state

  def _reset_buffer(self):
    for _ in range(self.window):
      self.state_buffer.append(np.zeros((84, 84), np.int32))

  def reset(self):
    if self.life_termination:
      self.life_termination = False  # Reset flag
      self.ale.act(0)  # Use a no-op after loss of life
      # self.ale.act(1)  # Use a 1 action after loss of life to start next game
    else:
      # Reset internals
      self._reset_buffer()
      self.ale.reset_game()
      # Perform up to 30 random no-ops before starting
      for _ in range(random.randrange(30)):  # at least 4 step
        self.ale.act(0)  # Assumes raw action 0 is always no-op
        # self.ale.act(1)  # Use a 1 action after loss of life to start next game
        if self.ale.game_over():
          self.ale.reset_game()
    # Process and return "initial" state
    observation = self._get_state()
    self.state_buffer.append(observation)
    self.lives = self.ale.lives()
    return np.stack(list(self.state_buffer), 0)

  def step(self, action):
    # Repeat action 4 times, max pool over last 2 frames
    frame_buffer = np.zeros((2, 84, 84), np.float32)
    reward, done = 0, False
    for t in range(4):
      reward += self.ale.act(self.actions.get(action))
      if t == 2:
        frame_buffer[0] = self._get_state()
      elif t == 3:
        frame_buffer[1] = self._get_state()
      done = self.ale.game_over()
      if done:
        break
    observation = np.max(frame_buffer, 0)
    self.state_buffer.append(observation)
    # Detect loss of life as terminal in training mode
    # lives = self.ale.lives()
    # if lives < self.lives and lives > 0:  # Lives > 0 for Q*bert
    #   self.life_termination = not done  # Only set flag when not truly done
    # self.lives = lives
    # if self.life_termination:
    #   if self.done_with_life_loss:
    #     done = True
      # else:
      #   self.ale.act(1)  # Start next game
      #   self.life_termination = False
    # Return state, reward, done
    return np.stack(list(self.state_buffer), 0), reward, done

  def action_space(self):
    return len(self.actions)

  def render(self):
    cv2.imshow('screen', self.ale.getScreenRGB()[:, :, ::-1])
    cv2.waitKey(1)

  def close(self):
    cv2.destroyAllWindows()

if __name__ == '__main__':
  cv2.namedWindow("Game", cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
  cv2.resizeWindow("Game", 5 * 84, 5 * 84)
  env = Env()
  done = False
  s = env.reset()
  while True:
    a = np.random.randint(0, 4)
    cv2.imshow("Game", s[-1,...])
    wk = cv2.waitKey(0) & 0xff
    if wk == ord('q'):
      cv2.destroyAllWindows()
      exit()
    s, r, done = env.step(a)
    print(f"{a=}, {r=}, {done=}")
    if done:
      env.reset()
    