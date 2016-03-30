import logging
import math
# import numpy
import random
import threading
# import time
import pickle
from util import draw

from interface import interface

# BIO
NUM_EDGES = 84060
NUM_CUTS_PERCENT = .01
MUTATION_CHANCE = .01
SURVIVAL_PERCENT = 0.1
CROSSOVER_CHANCE = 0.75

WEIGHT_MUTATION_CHANCE = 0.9
# WEIGHT_MUTATION_CHANCE = 0.99
CONNECTION_MUTATION_CHANCE = 0.25
# CONNECTION_MUTATION_CHANCE = 0.05
LINK_MUTATION_CHANCE = 1.5
# LINK_MUTATION_CHANCE = 1.1
BIAS_MUTATION_CHANCE = 0.4
# BIAS_MUTATION_CHANCE = 0.04
NODE_MUTATION_CHANCE = 0.5
# NODE_MUTATION_CHANCE = 0.05
WEIGHT_STEP = 0.1
MAX_NODES = 100000
NUM_OUTPUTS = 4
POPULATION_SIZE = 20
MAX_GENERATIONS = 100
MAX_FITNESS = 110
# SURVIVAL_PERCENT = 0.1
NUM_INPUTS = interface.VISION_MATRIX_W * interface.VISION_MATRIX_H


class Evolution:
    def __init__(self, brain=None):
        self.population = []
        self.innovation_count = 0
        # Communication with Matlab Fitness function.
        self.brain = None # TODO. CONNECT WITH MATLAB
        self.gid_count = 0

    def initiate_population(self):
        for i in range(POPULATION_SIZE):
            genome = Genome(self)
            genome.init_genes()
            self.population.append(genome)

    # TODO: THIS IS WHERE WE GET THE FITNESS. CONNECT WITH THE MATLAB CODE.
    def evaluate_population(self, generation):
        for genome in self.population:
            if genome.fitness is None:
                genome.fitness = self.brain.evaluate_fitness(genome.)

    def evolve(self):
        # Loop until max fitness of a generation meets the desired fitness level.
        # or until the number of generations is met.

        for generation in range(MAX_GENERATIONS):
            self.evaluate_population(generation)
            # store_population(self.population, self.innovation_count, "population")
            for genome in self.population:
                if (genome.fitness == MAX_FITNESS):
                    # store_genome(genome, "max_fitness")
                    return genome
            # Kill off the worst of the genomes.
            survived_num = self.natural_selection()

            while len(self.population) < POPULATION_SIZE:
                if len(self.population) == 1 or not (random.random() < CROSSOVER_CHANCE):
                    copied_genome = self.population[random.randint(0, survived_num - 1)].copy_genome()
                    copied_genome.mutate()
                    self.population.append(copied_genome)
                else:
                    g1_index = random.randrange(0, survived_num)
                    g2_index = random.randrange(0, survived_num)
                    self.crossover(g1_index, g2_index)

        self.evaluate_population(generation)
        return max(self.population, key=lambda genome: genome.fitness)

    def new_innovation(self):
        self.innovation_count += 1
        return self.innovation_count

    def crossover(self, g1_index, g2_index):
        """
        Creates offspring from fittest genomes and
        mutates children to form new generation
        """
        g1 = self.population[g1_index]
        g2 = self.population[g2_index]
        desired_length = len(g1)

        child_genome = []

        g1_copy, g2_copy = g1[:], g2[:]
        intersect = set(g1).intersection(g2)

        for val in intersect:
            child_genome.append(val)

        while len(child_genome) < desired_length:
            rand_val = random.random()
            if rand_val < 0.7 and len(g1_copy) > 0:
                g1_val = g1_copy.pop()
                while g1_val in intersect and len(g1_copy) > 0:
                    g1_val = g1_copy.pop()
                if g1_val not in intersect:
                    child_genome.append(g1_val)
            else:
                g2_val = g2_copy.pop()
                while g2_val in intersect and len(g2_copy) > 0:
                    g2_val = g2_copy.pop()
                if g2_val not in intersect:
                    child_genome.append(g2_val)

        child_genome.mutate()
        self.population.append(child)

    def natural_selection(self):
        """
        Selects top SURVIVAL_SIZE-most fit genomes - Kills the rest
        """
        survival_size = max(1, int(SURVIVAL_PERCENT * len(self.population)))
        self.population = sorted(self.population, key=lambda genome: genome.fitness, reverse=True)[:survival_size]
        return len(self.population)


class Genome:
    def __init__(self):
        self.genes = []
        self.fitness = None

    def init_genes(self):
        gene_dict = {}
        num_genes = int(NUM_CUTS_PERCENT * NUM_EDGES)
        
        while len(gene_dict) < num_genes:
            edge_index = random.randrange(NUM_EDGES)
            gene_dict[edge_index] = True
        self.genes = gene_dict.keys()

    def mutate(self):
        # call all different mutates
        genes_dict = {}
        for gene in self.genes:
            genes_dict[gene] = True

        for x in range(len(self.genes)):
            if random.random() < MUTATION_CHANCE:
                new_index = random.randrange(NUM_EDGES)
                while new_index in genes_dict:
                    new_index = random.randrange(NUM_EDGES)
                self.genes[x] = new_index

    def copy_genome(self):
        genome = Genome()
        genome.genes = self.genes
        genome.fitness = self.fitness

        return genome

# Functional mutate function
def func_mutate(genome):
    """
    genome is an array of genes: [1, 15, 18, 59]
    its indexes are the weights to be cut.

    """
    # call all different mutates
    genes_dict = {}
    for gene in genome:
        genes_dict[gene] = True
    for x in range(len(genome)):
        if random.random() < MUTATION_CHANCE:
            new_index = random.randrange(NUM_EDGES)
            while new_index in genes_dict:
                new_index = random.randrange(NUM_EDGES)
            genome[x] = new_index
    return genome

# Functional initialize genes
def func_init_genes():
    gene_dict = {}
    num_genes = int(NUM_CUTS_PERCENT * NUM_EDGES)
    while len(gene_dict) < num_genes:
        edge_index = random.randrange(NUM_EDGES)
        gene_dict[edge_index] = True
    genes = gene_dict.keys()
    return genes