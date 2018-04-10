import gym
import helpers


class EnvWrapper:

    def __init__(self, env_name, img_size, frames_in_state_count):
        self.env = gym.make(env_name)
        self.img_size = img_size
        self.frames_in_state_count = frames_in_state_count
        self.action_count = self.env.action_space.n
        self.state = None
        self.prev_state = None

    def render(self):
        return self.env.render()

    def reset(self):
        frame = self.env.reset()
        self.prev_state = self.state = helpers.get_start_state(frame, self.img_size, self.frames_in_state_count)
        return self.state

    def sample_action(self):
        return self.env.action_space.sample()

    def step(self, action, kth_frame=4):
        total_reward = 0
        return_frame = None
        return_is_done = None
        self.prev_state = self.state
        for i in range(kth_frame):
            return_frame, reward, return_is_done, _ = self.env.step(action)
            total_reward += helpers.transform_reward(reward)
            if return_is_done:
                break
        self.state = helpers.get_next_state(self.prev_state, return_frame, self.img_size, self.frames_in_state_count)
        return total_reward, return_is_done
