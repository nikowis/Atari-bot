import keras
import numpy as np

LEARNING_RATE = 0.00025


def fit_batch(model, frozen_target_model, gamma, start_states, actions, next_states, rewards, is_terminal):
    next_Q_values = frozen_target_model.predict([next_states, np.ones(actions.shape)])
    next_Q_values[np.where(is_terminal)[0]] = 0
    Q_values = rewards + gamma * np.max(next_Q_values, axis=1, keepdims=True)
    model.fit(
        [start_states, actions], actions * Q_values,
        epochs=1, batch_size=len(start_states), verbose=0
    )


def model(n_actions, img_size, frame_count):
    frames_input = keras.layers.Input((img_size, img_size, frame_count), name='frames')
    actions_input = keras.layers.Input((n_actions,), name='mask')
    conv_1 = keras.layers.Conv2D(32, (8, 8), strides=(4, 4), activation="relu")(frames_input)
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


def predict(model, state, action_count):
    action = model.predict([np.transpose(np.expand_dims(state, 0), (0, 2, 3, 1)), np.ones((1, action_count))])
    return np.argmax(action)
