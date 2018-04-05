import tensorflow as tf


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

    x = tf.reshape(x, shape=[-1, 105, 80, 1])

    conv1 = tf.nn.conv2d(x, weights['W_conv1'], strides=[1, 4, 4, 1], padding='SAME') + biases['b_conv1']
    conv1 = tf.nn.relu(conv1)

    conv2 = tf.nn.conv2d(conv1, weights['W_conv2'], strides=[1, 2, 2, 1], padding='SAME') + biases['b_conv2']
    conv2 = tf.nn.relu(conv2)

    conv3 = tf.nn.conv2d(conv2, weights['W_conv3'], strides=[1, 1, 1, 1], padding='SAME') + biases['b_conv3']
    conv3 = tf.nn.relu(conv3)

    fc1 = tf.reshape(conv3, [-1, 7 * 7 * 64])
    fc1 = tf.nn.relu(tf.matmul(fc1, weights['W_fc1']) + biases['b_fc1'])

    fc2 = tf.nn.relu(tf.matmul(fc1, weights['W_fc2']) + biases['b_fc2'])
    output = tf.matmul(fc2, weights['out']) + biases['out']
    return output
