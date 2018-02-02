"""
Generic setup of the data sources and the model training. 

Based on:
    https://github.com/fchollet/keras/blob/master/examples/mnist_mlp.py
and also on 
    https://github.com/fchollet/keras/blob/master/examples/mnist_cnn.py

"""
from pprint import pprint

import gym as gym
from keras.datasets import mnist, cifar10
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, np
from keras.utils.np_utils import to_categorical
from keras.callbacks import EarlyStopping, Callback
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
import tensorflow as tf

import logging

# Helper: Early stopping.
from agent import DQNAgent
from gym import wrappers
import gym_ple
from skimage.color import rgb2gray

early_stopper = EarlyStopping(monitor='val_loss', min_delta=0.1, patience=2, verbose=0, mode='auto')


# patience=5)
# monitor='val_loss',patience=2,verbose=0
# In your case, you can see that your training loss is not dropping - which means you are learning nothing after each epoch.
# It look like there's nothing to learn in this model, aside from some trivial linear-like fit or cutoff value.


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
    action_size = env.action_space.n
    state_height = env.observation_space.shape[0]
    state_width = env.observation_space.shape[1]
    state_channels = 1
    agent = DQNAgent(state_height, state_width, state_channels, action_size, geneparam)

    # Run the simulation
    total_score = 0

    for e in range(episodes):
        state = env.reset()
        # Convert state image to grayscale
        state = rgb2gray(state)
        episode_total = 0
        current_time = 0
        while True:
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            # Convert state image to gray scale
            next_state = rgb2gray(next_state)
            current_time += 1
            episode_total += reward
            total_score += reward

            # Remember the previous state, action, reward, and done
            agent.remember(state, action, reward, next_state, done)
            state = next_state

            if done:
                avg_score = total_score / (e + 1)
                print("episode: {}/{}, avg_score: {}, episode_score: {} ({}), e: {:.2}".format(
                    e, episodes, avg_score, episode_total, current_time, agent.epsilon))
                break

        if len(agent.memory) > agent.batch_size:
            agent.replay()

    env.close()

    return total_score
