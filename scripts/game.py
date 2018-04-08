# Import the gym module
from collections import deque

import gym
import helpers
import atari_model_keras
import tensorflow as tf
import random
import numpy as np
import time

# Create a breakout environment

env = gym.make('BreakoutDeterministic-v4')

img_size = 84
frames_count = 4
learning_rate = 0.00025
batch_size = 32
n_classes = env.action_space.n
memory_size = 1000000
memory = deque()

model = atari_model_keras.atari_model(n_classes)

render = False


def choose_best_action(nn_model, frame):
    return nn_model.predict(frame)


def get_start_state(frame):
    """
    Creates an array of 4 frames from a single frame.
    :param frame: starting frame
    :return: array of 4 frames
    """
    processed_frame = helpers.preprocess(frame)
    start_state = np.empty((frames_count, img_size, img_size))
    start_state[:] = processed_frame
    return start_state


def get_next_state(stte, new_frame):
    """
    Removes first frame from state, and appends new_frame at the end.
    :param stte: current state
    :param new_frame: frame to append
    :return: new state
    """
    new_state = np.empty((frames_count, img_size, img_size))
    processed_frame = helpers.preprocess(new_frame)
    new_state[0:frames_count - 1] = stte[1:frames_count]
    new_state[frames_count - 1] = processed_frame
    return new_state


def save_model(sess):
    saver = tf.train.Saver()
    saver.save(sess, './atari_model', global_step=100000)


start_time = time.time()
iteration = 0
for i in range(1000000):
    total_game_reward = 0
    is_done = False
    frame = env.reset()
    next_state = get_start_state(frame)
    mem_counter = 0
    # Render
    if render:
        env.render()

    while not is_done:
        iteration += 1
        epsilon = helpers.get_epsilon_for_iteration(iteration)
        state = next_state
        # Choose the action
        if random.random() < epsilon:
            action = env.action_space.sample()
        else:
            action = model.predict([np.transpose(np.expand_dims(next_state, 0), (0, 2, 3, 1)), np.ones((1, n_classes))])
            action = np.argmax(action)

        frame, reward, is_done, _ = env.step(action)
        next_state = get_next_state(state, frame)
        reward = helpers.transform_reward(reward)

        total_game_reward += reward
        one_hot_action = np.zeros((1, n_classes))
        one_hot_action[0, action - 1] = 1

        # if iteration < memory_size:
        #     memory.append([state, one_hot_action, next_state, reward, is_done])

        if iteration > batch_size:
            atari_model_keras.fit_batch(model, 0.99, np.expand_dims(state, axis=0), one_hot_action,
                                        np.expand_dims(reward, axis=0), np.expand_dims(next_state, axis=0),
                                        np.expand_dims(is_done, axis=0))

        if iteration % 1000 == 0:
            print(int(time.time() - start_time), 's iteration ', iteration)

        # Render
        if render:
            env.render()

    print('Total reward for game ', i, ' was ', int(total_game_reward))
