import gym
import numpy as np

import helpers
from ring_buf import AtariRingBuf


class EnvWrapper:

    def __init__(self, env_name, img_size, frames_in_state_count, memory_size):
        self.env = gym.make(env_name)
        self.img_size = img_size
        self.frames_in_state_count = frames_in_state_count
        self.action_count = self.env.action_space.n
        self.state = None
        self.prev_state = None
        self.buffer = AtariRingBuf(memory_size, self.action_count)

    def render(self):
        return self.env.render()

    def reset(self):
        frame = self.env.reset()
        self.prev_state = self.state = helpers.get_start_state(frame, self.frames_in_state_count)
        return self.state

    def sample_action(self):
        return self.env.action_space.sample()

    def get_state_arr(self):
        return np.array(self.state) / 255

    def step(self, action, kth_frame):
        total_reward = 0
        return_frame = None
        return_is_done = None
        self.prev_state = self.state
        for i in range(kth_frame):
            return_frame, reward, return_is_done, _ = self.env.step(action)
            total_reward += reward
            if return_is_done:
                break
        total_reward = helpers.transform_reward(total_reward)
        self.state = helpers.get_next_state(self.prev_state, return_frame, self.frames_in_state_count)
        self.buffer.append(self.prev_state, action, self.state, total_reward, return_is_done)

        return total_reward, return_is_done

    def get_batch(self, batch_size):
        bstates, bactions, bnext_states, b_rewards, b_terminals = self.buffer.get_batch(batch_size, True)
        return np.transpose(np.array(bstates)/255, [0, 2, 3, 1]), bactions, np.transpose(np.array(bnext_states)/255, [0, 2, 3,
                                                                                                              1]), b_rewards, b_terminals
