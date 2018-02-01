"""
Class that holds a genetic algorithm for evolving a network.

Inspiration:

    http://lethain.com/genetic-algorithms-cool-name-damn-simple/
"""
from __future__ import print_function

import random
import logging
import copy

from functools  import reduce
from operator   import add
from genome     import Genome
from idgen      import IDgen
from allgenomes import AllGenomes


class Evolver():
    """Class that implements genetic algorithm."""

    def __init__(self, all_possible_genes, retain=0.2, random_select=0.1, mutate_chance=0.3):
        """Create an optimizer.

        Args:
            all_possible_genes (dict): Possible genome parameters
            retain (float): Percentage of population to retain after
                each generation
            random_select (float): Probability of a rejected genome
                remaining in the population
            mutate_chance (float): Probability a genome will be
                randomly mutated

        """

        self.all_possible_genes = all_possible_genes
        self.retain             = retain
        self.random_select      = random_select
        self.mutate_chance      = mutate_chance

        #set the ID gen
        self.ids = IDgen()
        
    def create_population(self, count):
        """Create a population of random networks.

        Args:
            count (int): Number of networks to generate, aka the
                size of the population

        Returns:
            (list): Population of network objects

        """
        pop = []

        i = 0

        while i < count:
            
            # Initialize a new genome.
            genome = Genome( self.all_possible_genes, {}, self.ids.get_next_ID(), 0, 0, self.ids.get_Gen() )

            # Set it to random parameters.
            genome.set_genes_random()

            if i == 0:
                #this is where we will store all genomes
                self.master = AllGenomes( genome )
            else:
                # Make sure it is unique....
                while self.master.is_duplicate( genome ):
                    genome.mutate_one_gene()

            # Add the genome to our population.
            pop.append(genome)

            # and add to the master list
            if i > 0:
                self.master.add_genome(genome)

            i += 1

        #self.master.print_all_genomes()
        
        #exit()

        return pop

    @staticmethod
    def fitness(genome):
        """Return the accuracy, which is our fitness function."""
        return genome.accuracy

    def grade(self, pop):
        """Find average fitness for a population.

        Args:
            pop (list): The population of networks/genome

        Returns:
            (float): The average accuracy of the population

        """
        summed = reduce(add, (self.fitness(genome) for genome in pop))
        return summed / float((len(pop)))

    def breed(self, mom, dad):
        """
        Make a child from two parent genes
        :param mom: A genome parameter
        :param dad: A genome parameter
        :return: A child gene
        """
        child_gene = {}
        mom_gene = mom.geneparam
        dad_gene = dad.geneparam

        # Choose the optimizer
        child_gene['optimizer'] = random.choice([mom_gene['optimizer'], dad_gene['optimizer']])

        # Combine the layers
        max_len = max(len(mom_gene['layers']), len(dad_gene['layers']))
        child_layers = []
        for pos in range(max_len):
            from_mom = bool(random.getrandbits(1))
            # Add the layer from the correct parent IF it exists. Otherwise add nothing
            if from_mom and len(mom_gene['layers']) > pos:
                child_layers.append(mom_gene['layers'][pos])
            elif not from_mom and len(dad_gene['layers']) > pos:
                child_layers.append(dad_gene['layers'][pos])
        child_gene['layers'] = child_layers

        child = Genome(self.all_possible_genes, child_gene, self.ids.get_next_ID(), mom.u_ID, dad.u_ID, self.ids.get_Gen())

        #at this point, there is zero guarantee that the genome is actually unique

        # Randomly mutate one gene
        if self.mutate_chance > random.random():
            child.mutate_one_gene()

        #do we have a unique child or are we just retraining one we already have anyway?
        while self.master.is_duplicate(child):
            child.mutate_one_gene()

        self.master.add_genome(child)

        return child

    def evolve(self, pop):
        """Evolve a population of genomes.

        Args:
            pop (list): A list of genome parameters

        Returns:
            (list): The evolved population of networks

        """
        #increase generation 
        self.ids.increase_Gen()

        # Get scores for each genome
        graded = [(self.fitness(genome), genome) for genome in pop]

        #and use those scores to fill in the master list
        for genome in pop:
            self.master.set_accuracy(genome)

        # Sort on the scores.
        graded = [x[1] for x in sorted(graded, key=lambda x: x[0], reverse=True)]

        # Get the number we want to keep unchanged for the next cycle.
        retain_length = int(len(graded)*self.retain)

        # In this first step, we keep the 'top' X percent (as defined in self.retain)
        # We will not change them, except we will update the generation
        new_generation = graded[:retain_length]

        # For the lower scoring ones, randomly keep some anyway.
        # This is wasteful, since we _know_ these are bad, so why keep rescoring them without modification?
        # At least we should mutate them
        for genome in graded[retain_length:]:
            if self.random_select > random.random():
                gtc = copy.deepcopy(genome)
                
                while self.master.is_duplicate(gtc):
                    gtc.mutate_one_gene()

                gtc.set_generation( self.ids.get_Gen() )
                new_generation.append(gtc)
                self.master.add_genome(gtc)
        
        # Now find out how many spots we have left to fill.
        ng_length      = len(new_generation)
        print(str(ng_length))

        desired_length = len(pop) - ng_length

        children       = []

        # Add children, which are bred from pairs of remaining (i.e. very high or lower scoring) genomes.
        while len(children) < desired_length:

            # Get a random mom and dad, but, need to make sure they are distinct
            parents  = random.sample(range(ng_length-1), k=2)
            
            i_male   = parents[0]
            i_female = parents[1]

            male   = new_generation[i_male]
            female = new_generation[i_female]

            baby = self.breed(male, female)
            children.append(baby)

        new_generation.extend(children)

        return new_generation
