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

img_size = 84
frames_count = 4
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


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    start_time = time.time()
    iteration = 0
    for i in range(1000000):
        total_game_reward = 0
        is_done = False
        frame = env.reset()
        next_state = get_start_state(frame)

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

            frame, reward, is_done, _ = env.step(action)
            next_state = get_next_state(state, frame)
            reward = helpers.transform_reward(reward)

            total_game_reward += reward
            one_hot_action = np.zeros((1, n_classes))
            one_hot_action[0, action - 1] = 1
            atari_model.train_neural_network(sess, model, loss, optimizer, x, y, state, one_hot_action, reward,
                                             next_state)

            if iteration % 1000 == 0:
                print(int(time.time() - start_time), 's iteration ', iteration)

            # Render
            if render:
                env.render()

        print('Total reward for game ', i, ' was ', int(total_game_reward))
