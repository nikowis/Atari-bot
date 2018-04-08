import tensorflow as tf
import helpers
import random
import numpy as np

learning_rate = 0.00025
batch_size = 32
memory = []
img_size = 84 * 84
gamma = 0.99


def model(x, n_classes):
    weights = {
        # filtr 8x8, 4 wejsciowe klatki, 32 output
        'W_conv1': tf.Variable(tf.random_normal([8, 8, 4, 32])),
        'W_conv2': tf.Variable(tf.random_normal([4, 4, 32, 64])),
        'W_conv3': tf.Variable(tf.random_normal([3, 3, 64, 64])),
        'W_fc1': tf.Variable(tf.random_normal([7 * 7 * 64, 512])),
        'out': tf.Variable(tf.random_normal([512, n_classes]))}

    biases = {'b_conv1': tf.Variable(tf.random_normal([32])),
              'b_conv2': tf.Variable(tf.random_normal([64])),
              'b_conv3': tf.Variable(tf.random_normal([64])),
              'b_fc1': tf.Variable(tf.random_normal([512])),
              'out': tf.Variable(tf.random_normal([n_classes]))}

    x = tf.reshape(x, shape=[-1, 84, 84, 4])

    conv1 = tf.nn.conv2d(x, weights['W_conv1'], strides=[1, 4, 4, 1], padding='VALID') + biases['b_conv1']
    conv1 = tf.nn.relu(conv1)

    conv2 = tf.nn.conv2d(conv1, weights['W_conv2'], strides=[1, 2, 2, 1], padding='VALID') + biases['b_conv2']
    conv2 = tf.nn.relu(conv2)

    conv3 = tf.nn.conv2d(conv2, weights['W_conv3'], strides=[1, 1, 1, 1], padding='VALID') + biases['b_conv3']
    conv3 = tf.nn.relu(conv3)

    fc1 = tf.reshape(conv3, [-1, 7 * 7 * 64])
    fc1 = tf.nn.relu(tf.matmul(fc1, weights['W_fc1']) + biases['b_fc1'])

    output = tf.matmul(fc1, weights['out']) + biases['out']

    return output


def train_neural_network(sess, model, loss, optimizer, x, y, state, one_hot_action, reward, next_state):
    next_Q_values = sess.run(model, feed_dict={x: next_state})

    Q_values = reward + gamma * np.max(next_Q_values, axis=1)
    _, c = sess.run([optimizer, loss], feed_dict={x: state, y: one_hot_action * Q_values})


# def train_neural_network(model, loss, optimizer, x, y, frame, action, reward, next_frame):
#     with tf.Session() as sess:
#         sess.run(tf.initialize_all_variables())
#
#         for _ in range(int(mnist.train.num_examples / batch_size)):
#             epoch_x, epoch_y = mnist.train.next_batch(batch_size)
#             _, c = sess.run([optimizer, loss], feed_dict={x: epoch_x, y: epoch_y})
#             epoch_loss += c
#
#         print('Epoch', epoch, 'completed out of', hm_epochs, 'loss:', epoch_loss)
def loss(mdl, y):
    return tf.losses.mean_squared_error(labels=y, predictions=mdl)


def optimizer(lss):
    return tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(lss)
