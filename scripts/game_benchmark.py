# Import the gym module
import random
import time
import numpy as np
import matplotlib.pyplot as plt
import atari_model
import helpers
from env_wrapper import EnvWrapper

IMG_SIZE = 84
FRAMES_IN_STATE_COUNT = 4
EPSILON = 0.05
GAME_ENV_NAME = 'BreakoutDeterministic-v4'
RENDER = False
PRINT_LATEX = True
MODEL_PATH_PREFIX = './drive/app/models/'
# list of models with iteration count as file names
STARTING_MODELS = [0, 200, 600, 800, 1000, 1200, 1400, 1550, 1800, 2000, 2200,
                   2400, 2500, 2700, 3000, 3200, 3500, 3800, 4150, 4400, 4600, 4800,
                   5000, 5250, 5400, 5600, 5800, 6000, 6200, 6400, 6800, 7000,
                   7200, 7400, 7600, 7750, 8000]

GAMES_PER_MODEL = 5

env = EnvWrapper(GAME_ENV_NAME, IMG_SIZE, FRAMES_IN_STATE_COUNT, 1)
action_count = env.action_count
results = np.zeros((len(STARTING_MODELS), 2))
program_start_time = time.time()

for i in range(0, len(STARTING_MODELS)):
    model_name = STARTING_MODELS[i]
    model_path = MODEL_PATH_PREFIX + str(model_name)
    model = helpers.load_model(model_path)
    total_games_reward = 0
    start_time = time.time()
    for i_game in range(1, GAMES_PER_MODEL + 1):

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

    print('Model', model_path, ' played ', GAMES_PER_MODEL, ' game(s) in ', time.time() - start_time,
          ' seconds. Avg score for a single game: ', total_games_reward / GAMES_PER_MODEL)
    results[i, 0] = model_name
    results[i, 1] = total_games_reward / GAMES_PER_MODEL

x = results[:, 0]
y = results[:, 1]

plt.plot(x, y, markersize=3)
plt.xlabel('Model learning iterations (thousands)')
plt.ylabel('Average normalized score (' + str(GAMES_PER_MODEL) + ' games)')
plt.axis([0, STARTING_MODELS[len(STARTING_MODELS) - 1] + 0.05 * STARTING_MODELS[len(STARTING_MODELS) - 1]
             , 0, np.amax(y) + 0.05 * np.amax(y)])
plt.show()

if PRINT_LATEX:
    for i in range(0, len(STARTING_MODELS)):
        print('LaTex table rows:')
        print(results[i, 0], ' & ', results[i, 1], '\\\\\n\\hline')

print('Program finished in ', time.time() - program_start_time, ' seconds')

input("Press any key to exit...")
