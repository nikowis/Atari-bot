# Import the gym module
import random
import time

import atari_model
import helpers
from env_wrapper import EnvWrapper

# -------- MODEL CONSTS --------
IMG_SIZE = 84
FRAMES_IN_STATE_COUNT = 4
EPSILON = 0.05
GAME_ENV_NAME = 'BreakoutDeterministic-v4'
RENDER = False
STARTING_MODEL = './breakout-model8M'

GAMES_PER_MODEL = 10

env = EnvWrapper(GAME_ENV_NAME, IMG_SIZE, FRAMES_IN_STATE_COUNT, 1)
action_count = env.action_count
model = helpers.load_model(STARTING_MODEL)
print('Loaded model: ', STARTING_MODEL)

total_games_reward = 0
start_time = time.time()
for i_game in range(1, GAMES_PER_MODEL + 1):
    print('Model ', STARTING_MODEL, ' playing game ', i_game)
    env.reset()
    if RENDER:
        env.render()
    game_reward = 0
    is_done = False
    while not is_done:
        if random.random() < EPSILON:
            action = env.sample_action()
        else:
            action = atari_model.predict(model, env.get_state_arr(), action_count)
        reward, is_done = env.step(action, 1)
        game_reward += reward
        if RENDER:
            env.render()
    total_games_reward += game_reward

print('Model', STARTING_MODEL, ' played ', GAMES_PER_MODEL, ' games in ', time.time() - start_time, ' seconds.')
print('Average total score for a single game was ', total_games_reward / GAMES_PER_MODEL)
