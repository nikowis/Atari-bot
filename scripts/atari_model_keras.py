import keras
import numpy as np
import helpers
import random

n_classes = 4


def fit_batch(model, gamma, start_states, actions, rewards, next_states, is_terminal):
    """Do one deep Q learning iteration.

    Params:
    - model: The DQN
    - gamma: Discount factor (should be 0.99)
    - start_states: numpy array of starting states
    - actions: numpy array of one-hot encoded actions corresponding to the start states
    - rewards: numpy array of rewards corresponding to the start states and actions
    - next_states: numpy array of the resulting states corresponding to the start states and actions
    - is_terminal: numpy boolean array of whether the resulting state is terminal

    """
    # First, predict the Q values of the next states. Note how we are passing ones as the mask.
    trans_next_states = np.transpose(next_states, (0,2,3,1))
    trans_start_states = np.transpose(start_states, (0, 2, 3, 1))
    next_Q_values = model.predict([trans_next_states, actions])
    # The Q values of the terminal states is 0 by definition, so override them
    next_Q_values[is_terminal] = 0
    # The Q values of each start state is the reward + gamma * the max next state Q value
    Q_values = rewards + gamma * np.max(next_Q_values, axis=1)
    # Fit the keras model. Note how we are passing the actions as the mask and multiplying
    # the targets by the actions.
    model.fit(
        [trans_start_states, actions], actions * Q_values[:, None],
        epochs=1, batch_size=len(trans_start_states), verbose=0
    )


def atari_model(n_actions):
    # We assume a theano backend here, so the "channels" are first.
    ATARI_SHAPE = (84, 84, 4)
    # With the functional API we need to define the inputs.
    frames_input = keras.layers.Input(ATARI_SHAPE, name='frames')
    actions_input = keras.layers.Input((n_actions, ), name='mask')

    # Assuming that the input frames are still encoded from 0 to 255. Transforming to [0, 1].
    normalized = keras.layers.Lambda(lambda x: x / 255.0)(frames_input)

    conv_1 = keras.layers.Conv2D(32, (8, 8), strides=(4, 4), activation="relu")(normalized)
    conv_2 = keras.layers.Conv2D(64, (4, 4), strides=(2, 2), activation="relu")(conv_1)
    conv_3 = keras.layers.Conv2D(64, (3, 3), strides=(1, 1), activation="relu")(conv_2)
    conv_flattened = keras.layers.core.Flatten()(conv_3)
    hidden = keras.layers.Dense(512, activation='relu')(conv_flattened)
    output = keras.layers.Dense(n_actions)(hidden)
    filtered_output = keras.layers.multiply([output, actions_input])
    model = keras.models.Model(inputs=[frames_input, actions_input], outputs=filtered_output)
    optimizer = keras.optimizers.RMSprop(lr=0.00025, rho=0.95, epsilon=0.01)
    model.compile(optimizer, loss='mse')
    return model


def choose_best_action(model, state):
    return 1


def q_iteration(env, model, state, iteration):
    # Choose epsilon based on the iteration
    epsilon = helpers.get_epsilon_for_iteration(iteration)

    # Choose the action
    if random.random() < epsilon:
        action = env.action_space.sample()
    else:
        action = choose_best_action(model, state)

    # Play one game iteration (note: according to the next paper, you should actually play 4 times here)
    new_state, reward, is_done, _ = env.step(action)
    memory.add(state, action, new_state, reward, is_done)

    # Sample and fit
    batch = memory.sample_batch(32)
    atari_model.fit_batch(model, batch)
