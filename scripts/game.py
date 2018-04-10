# Import the gym module
import os
import random
import time

import psutil
from tensorflow.python.client import device_lib
import atari_model
import helpers
from env_wrapper import EnvWrapper

MODEL_PATH = "./model"

GAMMA = 0.99
IMG_SIZE = 84
FRAMES_IN_STATE_COUNT = 4
BATCH_SIZE = 32
BUCKET_SIZE = 20
MEMORY_SIZE = 120000
FREEZE_ITERATIONS = 10000
REPORT_ITERATIONS = 10000
SAVE_MODEL_ITERATIONS = 100000
REPLAY_START_SIZE = 50000

RENDER = False
print(device_lib.list_local_devices())

env = EnvWrapper('BreakoutDeterministic-v4', IMG_SIZE, FRAMES_IN_STATE_COUNT, MEMORY_SIZE)

action_count = env.action_count

model = atari_model.model(action_count, IMG_SIZE, FRAMES_IN_STATE_COUNT)
frozen_target_model = helpers.copy_model(model)
helpers.save_model(frozen_target_model, MODEL_PATH)

process = psutil.Process(os.getpid())
print(process.memory_info())

start_time = time.time()
iteration = 0
buckets = [0] * BUCKET_SIZE
for i in range(1000000):
    total_game_reward = 0
    is_done = False
    env.reset()

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

        if iteration > REPLAY_START_SIZE:
            bstates, bactions, bnext_states, b_rewards, b_terminals = env.get_batch(BATCH_SIZE)
            atari_model.fit_batch(model, frozen_target_model, GAMMA, bstates, bactions, bnext_states, b_rewards,
                                  b_terminals)

        if iteration % REPORT_ITERATIONS == 0:
            print(int(time.time() - start_time), 's iteration ', iteration)
            print('Scores :', buckets)
            print(process.memory_info())
            buckets = [0] * BUCKET_SIZE

        if iteration % FREEZE_ITERATIONS == 0:
            frozen_target_model = helpers.copy_model(model)

        if iteration % SAVE_MODEL_ITERATIONS == 0:
            helpers.save_model(model, MODEL_PATH + str(iteration))

        if RENDER:
            env.render()

    print('Total reward for game ', i, ' was ', int(total_game_reward))

    if total_game_reward >= BUCKET_SIZE:
        buckets[BUCKET_SIZE - 1] += 1
    else:
        buckets[int(total_game_reward)] += 1
