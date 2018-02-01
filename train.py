"""
Generic setup of the data sources and the model training. 

Based on:
    https://github.com/fchollet/keras/blob/master/examples/mnist_mlp.py
and also on 
    https://github.com/fchollet/keras/blob/master/examples/mnist_cnn.py

"""
from pprint import pprint

import gym as gym
from keras.datasets       import mnist, cifar10
from keras.models         import Sequential
from keras.layers import Dense, Dropout, Flatten, np
from keras.utils.np_utils import to_categorical
from keras.callbacks      import EarlyStopping, Callback
from keras.layers         import Conv2D, MaxPooling2D
from keras                import backend as K

import logging

# Helper: Early stopping.
from agent import DQNAgent
from gym import wrappers
import gym_ple


early_stopper = EarlyStopping( monitor='val_loss', min_delta=0.1, patience=2, verbose=0, mode='auto' )

#patience=5)
#monitor='val_loss',patience=2,verbose=0
#In your case, you can see that your training loss is not dropping - which means you are learning nothing after each epoch. 
#It look like there's nothing to learn in this model, aside from some trivial linear-like fit or cutoff value.


def train_and_score(geneparam, genehash, env_id, episodes):
    """Train the model, return test loss.

    Args:
        network (dict): the parameters of the network
        dataset (str): Dataset to use for training/evaluating

    """
    logging.info("Setting up environment")

    # Create the environment
    env = gym.make(env_id)
    outdir = '/tmp/genetic-agent-results/{}'.format(genehash)
    env = wrappers.Monitor(env, directory=outdir, force=True)
    env.seed(0)

    # Setup the agent
    state_size = env.observation_space.shape[0] * env.observation_space.shape[1] * env.observation_space.shape[2]
    action_size = env.action_space.n
    agent = DQNAgent(state_size, action_size, geneparam)

    # Run the simulation
    total_score = 0
    reward = 0
    done = False
    batch_size = 32

    for e in range(episodes):
        state = env.reset()
        # pprint(state)
        state = np.reshape(state, [1, state_size])
        while True:
            action = agent.act(state)

            next_state, reward, done, _ = env.step(action)
            total_score += reward
            next_state = np.reshape(next_state, [1, state_size])

            # Remember the previous state, action, reward, and done
            agent.remember(state, action, reward, next_state, done)

            state = next_state

            if done:
                print("episode: {}/{}".format(e, episodes))
                break

        if len(agent.memory) > batch_size:
            agent.replay(batch_size)

    env.close()

    return total_score
