import tensorflow as tf
import helpers
import random
import numpy as np

learning_rate = 0.00025
batch_size = 32
memory = []
img_size = 84 * 84


def model(x, n_classes):
    weights = {
        # filtr 8x8, 1 wejsciowy obraz, 32 output
        'W_conv1': tf.Variable(tf.random_normal([8, 8, 1, 32])),
        'W_conv2': tf.Variable(tf.random_normal([4, 4, 32, 64])),
        'W_conv3': tf.Variable(tf.random_normal([3, 3, 64, 64])),
        'W_fc1': tf.Variable(tf.random_normal([7 * 7 * 64, 512])),
        'W_fc2': tf.Variable(tf.random_normal([512, 512])),
        'out': tf.Variable(tf.random_normal([512, n_classes]))}

    biases = {'b_conv1': tf.Variable(tf.random_normal([32])),
              'b_conv2': tf.Variable(tf.random_normal([64])),
              'b_conv3': tf.Variable(tf.random_normal([64])),
              'b_fc1': tf.Variable(tf.random_normal([512])),
              'b_fc2': tf.Variable(tf.random_normal([512])),
              'out': tf.Variable(tf.random_normal([n_classes]))}

    x = tf.reshape(x, shape=[-1, 84, 84, 1])

    conv1 = tf.nn.conv2d(x, weights['W_conv1'], strides=[1, 4, 4, 1], padding='VALID') + biases['b_conv1']
    conv1 = tf.nn.relu(conv1)

    conv2 = tf.nn.conv2d(conv1, weights['W_conv2'], strides=[1, 2, 2, 1], padding='VALID') + biases['b_conv2']
    conv2 = tf.nn.relu(conv2)

    conv3 = tf.nn.conv2d(conv2, weights['W_conv3'], strides=[1, 1, 1, 1], padding='VALID') + biases['b_conv3']
    conv3 = tf.nn.relu(conv3)

    fc1 = tf.reshape(conv3, [-1, 7 * 7 * 64])
    fc1 = tf.nn.relu(tf.matmul(fc1, weights['W_fc1']) + biases['b_fc1'])

    fc2 = tf.matmul(fc1, weights['W_fc2']) + biases['b_fc2']
    output = tf.matmul(fc2, weights['out']) + biases['out']

    return output


def train_neural_network(x,y):
    prediction = model(x)
    loss = tf.losses.mean_squared_error(labels=y, prediction=prediction)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(loss)

    hm_epochs = 10
    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())

        for epoch in range(hm_epochs):
            epoch_loss = 0
            for _ in range(int(mnist.train.num_examples / batch_size)):
                epoch_x, epoch_y = mnist.train.next_batch(batch_size)
                _, c = sess.run([optimizer, cost], feed_dict={x: epoch_x, y: epoch_y})
                epoch_loss += c

            print('Epoch', epoch, 'completed out of', hm_epochs, 'loss:', epoch_loss)

        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))

        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        print('Accuracy:', accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))
