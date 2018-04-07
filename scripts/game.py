# Import the gym module
import gym
import helpers
import atari_model
import tensorflow as tf
import random
import numpy as np
import time

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

render = False


def choose_best_action(model, frame):
    pred = model.predict(frame)
    return pred


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    start_time = time.time()
    iteration = 0
    for i in range(1, 100000):
        total_game_reward = 0
        is_done = False
        next_state = env.reset()
        next_state = helpers.preprocess(next_state)
        next_state = np.reshape(next_state, (1, next_state.shape[0], next_state.shape[1]))
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
                action = np.argmax(sess.run(model, feed_dict={x: state}))

            next_state, reward, is_done, _ = env.step(action)

            reward = helpers.transform_reward(reward)
            next_state = helpers.preprocess(next_state)
            next_state = np.reshape(next_state, (1, next_state.shape[0], next_state.shape[1]))
            total_game_reward += reward
            one_hot_action = np.zeros((1, n_classes))
            one_hot_action[0, action - 1] = 1
            atari_model.train_neural_network(sess, model, loss, optimizer, x, y, state, one_hot_action, reward,
                                             next_state)

            if iteration % 500 == 0:
                print(int(time.time() - start_time), 's iteration ', iteration)

            # Render
            if render:
                env.render()

        print('Total reward for game ', i, ' was ', int(total_game_reward))
