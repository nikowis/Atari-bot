# Import the gym module
import gym
import helpers
import atari_model
import tensorflow as tf

# Create a breakout environment
img_size = 105 * 80
env = gym.make('BreakoutDeterministic-v4')
# Reset it, returns the starting frame
frames = env.reset()
# Render
env.render()

batch_size = 32
n_classes = env.action_space.n
memory = []

x = tf.placeholder(tf.float32, [None, img_size])
y = tf.placeholder(tf.float32, [None, n_classes])

model = atari_model.model(x, n_classes)

is_done = False
i = 0
state, reward, is_done, _ = env.step(env.action_space.sample())
while not is_done:
    i += 1
    # # Perform a random action, returns the new frame, reward and whether the game is over
    frame, reward, is_done, _ = env.step(env.action_space.sample())
    reward = helpers.transform_reward(reward)
    frame = helpers.preprocess(frame)

    # Render
    if not is_done:
        env.render()
