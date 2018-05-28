# Import the gym module
import os
import random
import time

import psutil
from tensorflow.python.client import device_lib

import atari_model
import helpers
from env_wrapper import EnvWrapper

# -------- MODEL CONSTS --------
GAME_ENV_NAME = 'BreakoutDeterministic-v4'
LEARN = True
RENDER = False
SAVE_MODEL_PATH = "./model"
STARTING_MODEL = None
GAMMA = 0.99
IMG_SIZE = 84
FRAMES_IN_STATE_COUNT = 4
BATCH_SIZE = 32
MEMORY_SIZE = 1000000
FREEZE_ITERATIONS = 10000
REPLAY_START_SIZE = 50000
LAST_EPSILON_DECREASE_ITERATION = 1000000
END_EPSILON = 0.1
START_EPSILON = 1.0

# -------- REPORT CONSTS --------
REPORT_ITERATIONS = 10000
SAVE_MODEL_ITERATIONS = 50000
BUCKET_SIZE = 15

print(device_lib.list_local_devices())

env = EnvWrapper(GAME_ENV_NAME, IMG_SIZE, FRAMES_IN_STATE_COUNT, MEMORY_SIZE)

action_count = env.action_count

if STARTING_MODEL is None:
    model = atari_model.model(action_count, IMG_SIZE, FRAMES_IN_STATE_COUNT)
else:
    model = helpers.load_model(STARTING_MODEL)
    print('Loaded model: ', STARTING_MODEL)

if LEARN:
    frozen_target_model = helpers.copy_model(model)

process = psutil.Process(os.getpid())
print('RAM :', helpers.convert_size(process.memory_info().rss))

start_time = time.time()
iteration = 0
buckets = [0] * BUCKET_SIZE
total_rewards = total_games = 0
while True:
    total_game_reward = 0
    is_done = False
    env.reset()

    if RENDER:
        env.render()

    while not is_done:
        iteration += 1

        eps = helpers.get_epsilon_for_iteration(iteration, LAST_EPSILON_DECREASE_ITERATION,
                                                END_EPSILON, START_EPSILON)
        if random.random() < eps:
            action = env.sample_action()
        else:
            action = atari_model.predict(model, env.get_state_arr(), action_count)

        reward, is_done = env.step(action, 1)

        total_game_reward += reward

        if LEARN and iteration > REPLAY_START_SIZE and iteration > BATCH_SIZE:
            bstates, bactions, bnext_states, b_rewards, b_terminals = env.get_batch(BATCH_SIZE)
            atari_model.fit_batch(model, frozen_target_model, GAMMA, bstates, bactions, bnext_states, b_rewards,
                                  b_terminals)

        if iteration % REPORT_ITERATIONS == 0:
            print(int(time.time() - start_time), 's iteration ', iteration)
            print('Scores :', buckets)
            if total_games > 0:
                print('Avg score :', total_rewards / total_games)
            print('RAM :', helpers.convert_size(process.memory_info().rss))
            buckets = [0] * BUCKET_SIZE
            total_rewards = 0
            total_games = 0

        if LEARN and iteration % FREEZE_ITERATIONS == 0:
            frozen_target_model = helpers.copy_model(model)

        if LEARN and iteration % SAVE_MODEL_ITERATIONS == 0:
            helpers.save_model(model, SAVE_MODEL_PATH + str(iteration))

        if RENDER:
            env.render()

    # print('Total reward for game ', i, ' was ', int(total_game_reward))
    total_rewards += total_game_reward
    total_games += 1
    if total_game_reward >= BUCKET_SIZE:
        buckets[BUCKET_SIZE - 1] += 1
    else:
        buckets[int(total_game_reward)] += 1
