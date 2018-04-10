# Import the gym module
import os
import random
import time

import numpy as np
import psutil
from tensorflow.python.client import device_lib
import atari_model
import helpers
from env_wrapper import EnvWrapper
from ring_buf import AtariRingBuf

GAMMA = 0.99
IMG_SIZE = 84
FRAMES_IN_STATE_COUNT = 4
BATCH_SIZE = 32

MEMORY_SIZE = 100000
FREEZE_ITERATIONS = 10000
RENDER = False
print(device_lib.list_local_devices())

env = EnvWrapper('BreakoutDeterministic-v4', IMG_SIZE, FRAMES_IN_STATE_COUNT)

action_count = env.action_count

memory = AtariRingBuf(MEMORY_SIZE, action_count)
model = atari_model.model(action_count, IMG_SIZE, FRAMES_IN_STATE_COUNT)
frozen_target_model = helpers.copy_model(model)

process = psutil.Process(os.getpid())
print(process.memory_info())

start_time = time.time()
iteration = 0
for i in range(1000000):
    total_game_reward = 0
    is_done = False
    env.reset()
    mem_counter = 0

    if RENDER:
        env.render()

    while not is_done:
        iteration += 1

        if random.random() < helpers.get_epsilon_for_iteration(iteration):
            action = env.sample_action()
        else:
            action = atari_model.predict(model, env.state, action_count)

        reward, is_done = env.step(action)

        total_game_reward += reward

        memory.append(env.prev_state, action, env.state, reward, is_done)

        if iteration > BATCH_SIZE:
            bstates, bactions, bnext_states, b_rewards, b_terminals = memory.get_batch(BATCH_SIZE)
            atari_model.fit_batch(model, frozen_target_model, GAMMA, bstates, bactions, bnext_states, b_rewards,
                                  b_terminals)

        if iteration % 1000 == 0:
            print(int(time.time() - start_time), 's iteration ', iteration)
            print(process.memory_info())

        if iteration % FREEZE_ITERATIONS == 0:
            frozen_target_model = helpers.copy_model(model)

        if RENDER:
            env.render()

    print('Total reward for game ', i, ' was ', int(total_game_reward))
