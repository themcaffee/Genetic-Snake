from collections import deque
import random
from pprint import pprint

from keras.layers import Dense, np, Dropout, Conv2D, Flatten, MaxPooling2D
from keras.models import Sequential
from keras import backend as K
from keras.optimizers import Adam


class DQNAgent(object):
    def __init__(self, state_height, state_width, state_channels, action_size, geneparam):
        self.state_height = state_height
        self.state_width = state_width
        self.state_channels = state_channels
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95    # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.9998
        self.learning_rate = 0.001
        self.geneparam = geneparam
        self.batch_size = 32
        self.model = self._build_model(self.geneparam)

    def _build_model(self, gene_param):
        nb_layers = len(gene_param)

        # Neural Net for Deep-Q learning Model
        model = Sequential()

        for i in range(0, nb_layers - 1):
            nb_neurons = gene_param['layers'][i]['nb_neurons']
            activation = gene_param['layers'][i]['activation']

            if i == 0:
                model.add(Conv2D(nb_neurons, kernel_size=(2,2), activation=activation, input_shape=(self.state_height, self.state_width, self.state_channels)))
            else:
                model.add(Conv2D(nb_neurons, kernel_size=(2,2), activation=activation))

            if i < 2:  # otherwise we hit zero
                model.add(MaxPooling2D(pool_size=(2, 2)))

            model.add(Dropout(0.2))

        model.add(Flatten())
        dense_nb_neurons = gene_param['layers'][nb_layers - 1]['nb_neurons']
        dense_activation = gene_param['layers'][nb_layers - 1]['activation']
        model.add(Dense(dense_nb_neurons, activation=dense_activation))
        model.add(Dropout(0.5))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse',
                      optimizer=Adam(lr=self.learning_rate))
        return model

    def _build_simple_model(self, gene_param):

        # Neural Net for Deep-Q learning Model
        model = Sequential()
        model.add(Conv2D(24, kernel_size=(3, 3), activation='relu', input_shape=(self.state_height, self.state_width, self.state_channels)))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Conv2D(24, kernel_size=(2, 2), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Flatten())
        model.add(Dense(24, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse',
                      optimizer=Adam(lr=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def _reshape_state(self, state):
        return np.reshape(state, (1, state.shape[0], state.shape[1], 1))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        state = self._reshape_state(state)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])  # returns action

    def replay(self):
        minibatch = random.sample(self.memory, self.batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            next_state = self._reshape_state(next_state)
            if not done:
                target = (reward + self.gamma * np.amax(self.model.predict(next_state)[0]))
            state = self._reshape_state(state)
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)
