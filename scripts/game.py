# Import the gym module
import gym
import helpers
import atari_model
import tensorflow as tf
import random
import numpy as np

# Create a breakout environment

env = gym.make('BreakoutDeterministic-v4')

img_size = 84 * 84
learning_rate = 0.00025
batch_size = 32
n_classes = env.action_space.n
memory = []

x = tf.placeholder(tf.float32, [None, 84, 84])
y = tf.placeholder(tf.float32, [None, n_classes])

model = atari_model.model(x, n_classes)
loss = tf.losses.mean_squared_error(labels=y, predictions=model)
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(loss)

is_done = False
i = 1


def choose_best_action(model, frame):
    pred = model.predict(frame)
    return pred


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for i in range(1, 6):
        # Reset it, returns the starting frame
        frames = env.reset()
        # Render
        env.render()
        total_reward = 0
        frame, reward, is_done, _ = env.step(env.action_space.sample())
        reward = helpers.transform_reward(reward)
        frame = helpers.preprocess(frame)
        total_reward += reward

        iteration = 0
        while not is_done:
            iteration += 1
            epsilon = helpers.get_epsilon_for_iteration(iteration)
            epsilon = 0
            frame = np.reshape(frame, (1, frame.shape[0], frame.shape[1]))
            # Choose the action
            if random.random() < epsilon:
                action = env.action_space.sample()
            else:
                action = sess.run(model, feed_dict={x: frame})
                action = np.argmax(action)

            frame, reward, is_done, _ = env.step(action)

            reward = helpers.transform_reward(reward)
            frame = helpers.preprocess(frame)
            total_reward += reward
            # Render
            if not is_done:
                env.render()

        print('Total reward for game ', i, ' was ', int(total_reward))
