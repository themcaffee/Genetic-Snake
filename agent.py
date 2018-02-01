from collections import deque
import random
from pprint import pprint

from keras.layers import Dense, np, Dropout
from keras.models import Sequential


class DQNAgent(object):
    def __init__(self, state_size, action_size, geneparam):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95    # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.geneparam = geneparam
        self.model = self._build_model(self.geneparam)

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
        model.compile(loss='mse',
                      optimizer=gene_param['optimizer'])
        return model

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
            target = reward
            if not done:
                target = reward + self.gamma * \
                                  np.amax(self.model.predict(next_state)[0])
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
