import argparse
import sys
import logging

import numpy as np

from evolver import Evolver
from tqdm import tqdm
import gym
from gym import wrappers
import gym_ple
from pprint import pprint


def train_genomes(genomes, env_id, episodes):
    """Train each genome.

    Args:
        networks (list): Current population of genomes
        dataset (str): Dataset to use for training/evaluating

    """
    logging.info("***train_networks(networks, dataset)***")

    pbar = tqdm(total=len(genomes))

    for genome in genomes:
        genome.train(env_id, episodes)
        pbar.update(1)

    pbar.close()


def get_average_score(genomes):
    """Get the average score for a group of networks/genomes.

    Args:
        networks (list): List of networks/genomes

    Returns:
        int: The average score of a population of networks/genomes.

    """
    total_score = 0

    for genome in genomes:
        total_score += genome.total_score

    return total_score / len(genomes)


def generate(generations, population, all_possible_genes, env_id, episodes):
    """Generate a network with the genetic algorithm.

    Args:
        generations (int): Number of times to evolve the population
        population (int): Number of networks in each generation
        all_possible_genes (dict): Parameter choices for networks
        dataset (str): Dataset to use for training/evaluating

    """
    logging.info("***generate(generations, population, all_possible_genes)***")

    evolver = Evolver(all_possible_genes)

    genomes = evolver.create_population(population)

    # Evolve the generation.
    for i in range(generations):

        logging.info("***Now in generation %d of %d***" % (i + 1, generations))

        print_genomes(genomes)

        # Train and get score for networks/genomes.
        train_genomes(genomes, env_id, episodes)

        # Get the average score for this generation.
        average_score = get_average_score(genomes)

        # Print out the average score each generation.
        logging.info("Generation average: %d" % average_score)
        logging.info('-'*80) #-----------

        # Evolve, except on the last iteration.
        if i != generations - 1:
            # Evolve!
            genomes = evolver.evolve(genomes)

    # Sort our final population according to performance.
    genomes = sorted(genomes, key=lambda x: x.total_score, reverse=True)

    # Print out the top 5 networks/genomes.
    print_genomes(genomes[:5])


def print_genomes(genomes):
    """Print a list of genomes.

    Args:
        genomes (list): The population of networks/genomes

    """
    logging.info('-'*80)

    for genome in genomes:
        genome.print_genome()


def main():
    # You can set the level to logger.DEBUG or logger.WARN if you
    # want to change the amount of output.
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('--population', type=int, default=15, help='Select the population')
    parser.add_argument('--generations', type=int, default=8, help='Select the generations')
    parser.add_argument('--episodes', type=int, default=100)
    parser.add_argument('--env_id', nargs='?', default='Snake-v0', help='Select the environment to run')
    args = parser.parse_args()

    all_possible_genes = {
        'activation': ['relu', 'elu', 'tanh', 'sigmoid', 'hard_sigmoid','softplus','linear'],
        'optimizer':  ['rmsprop', 'adam', 'sgd', 'adagrad','adadelta', 'adamax', 'nadam']
    }
    all_possible_genes['nb_neurons'] = list(range(2, 128))
    all_possible_genes['nb_layers'] = list(range(2, 5))

    generate(args.generations, args.population, all_possible_genes, args.env_id, args.episodes)


if __name__ == '__main__':
    main()
