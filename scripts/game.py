# Import the gym module
import os
import random
import time

import gym
import numpy as np
import psutil
import tensorflow as tf
from tensorflow.python.client import device_lib
import h5py
import atari_model_keras
import helpers
from RingBuffer import AtariRingBuf

# Create a breakout environment

print(device_lib.list_local_devices())

env = gym.make('BreakoutDeterministic-v4')

img_size = 84
frames_count = 4
learning_rate = 0.00025
batch_size = 32
n_classes = env.action_space.n
memory_size = 100000
memory = AtariRingBuf(memory_size)
freeze_iterations = 10000


model = atari_model_keras.atari_model(n_classes)
frozen_target_model = helpers.copy_model(model)

render = False

process = psutil.Process(os.getpid())
print(process.memory_info())


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

        memory.append(np.transpose(state, [1, 2, 0]), one_hot_action, np.transpose(next_state, [1, 2, 0]), reward,
                      is_done)

        if iteration > batch_size:
            bstates, bactions, bnext_states, b_rewards, b_terminals = memory.get_batch(batch_size)
            atari_model_keras.fit_batch(model, frozen_target_model, 0.99, bstates, bactions, bnext_states, b_rewards, b_terminals)

        if iteration % 1000 == 0:
            print(int(time.time() - start_time), 's iteration ', iteration)
            print(process.memory_info())

        if iteration % freeze_iterations == 0:
            frozen_target_model = helpers.copy_model(model)

        # Render
        if render:
            env.render()

    print('Total reward for game ', i, ' was ', int(total_game_reward))
