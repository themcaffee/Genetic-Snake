from collections import deque
import random
from pprint import pprint

from keras.layers import Dense, np, Dropout
from keras.models import Sequential
from keras import backend as K
from keras.optimizers import Adam


class DQNAgent(object):
    def __init__(self, state_size, action_size, geneparam):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95    # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.99
        self.learning_rate = 0.001
        self.geneparam = geneparam
        self.model = self._build_model(self.geneparam)
        self.target_model = self._build_model(self.geneparam)
        self.update_target_model()

    def _huber_loss(self, target, prediction):
        # sqrt(1+error^2)-1
        error = prediction - target
        return K.mean(K.sqrt(1+K.square(error))-1, axis=-1)

    def _build_model(self, gene_param):
        nb_layers = len(gene_param)

        # Neural Net for Deep-Q learning Model
        model = Sequential()

        for i in range(0, nb_layers):
            nb_neurons = gene_param['layers'][i]['nb_neurons']
            activation = gene_param['layers'][i]['activation']

            if i == 0:
                model.add(Dense(nb_neurons, activation=activation, input_dim=self.state_size))
            else:
                model.add(Dense(nb_neurons, activation=activation))

            model.add(Dropout(0.2))

        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss=self._huber_loss,
                      optimizer=Adam(lr=self.learning_rate))
        return model

    def update_target_model(self):
        # copy weights from model to target_model
        self.target_model.set_weights(self.model.get_weights())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])  # returns action

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = self.model.predict(state)
            if done:
                target[0][action] = reward
            else:
                a = self.model.predict(next_state)[0]
                t = self.target_model.predict(next_state)[0]
                target[0][action] = reward + self.gamma * t[np.argmax(a)]
            self.model.fit(state, target, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)
