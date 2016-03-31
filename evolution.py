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

# Functional initialize genes
def random_genes():
    gene_dict = {}
    num_genes = int(NUM_CUTS_PERCENT * NUM_EDGES)
    while len(gene_dict) < num_genes:
        edge_index = random.randrange(NUM_EDGES)
        gene_dict[edge_index] = True
    genes = gene_dict.keys()
    return genes

def initiate_population(population):
    for i in range(POPULATION_SIZE):
        random_genes = random_genes()
        # Fitness of 0
        population.append([random_genes, 0.0])
    return population

# TODO: THIS IS WHERE WE GET THE FITNESS. CONNECT WITH THE MATLAB CODE.
def evaluate_population(self, population):
    for genome in population:
        # genome -> [genes, fitness]
        if genome[1] == 0:
            genome[1] = # TODO call mayanks function here for fitness(genome[0])

def natural_selection(population):
    """
    Selects top SURVIVAL_SIZE-most fit genomes - Kills the rest

    """
    survival_size = max(1, int(SURVIVAL_PERCENT * len(population)))
    # Population looks like [[[1,2,3,4,5,5], 1500], [[1,2,3,4,5,5], 1500], [[1,2,3,4,5,5], 1500]]
    # Each index is an array of 2 index: genome, and fitness
    # population[0] = [[1,2,3,4,5,5], 1500] -> genome, fitness 
    population = sorted(population, key=lambda genome: genome[1], reverse=True)[:survival_size]
    return population

def copy_genome(genome):
    new_genome = [genome[0][:], genome[1]]
    return new_genome

# Functional mutate function
def mutate(genome):
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

def crossover(self, g1_index, g2_index, population):
    """
    Creates offspring from fittest genomes and
    mutates children to form new generation
    """
    # Population looks like [[[1,2,3,4,5,5], 1500], [[1,2,3,4,5,5], 1500], [[1,2,3,4,5,5], 1500]]
    # Each index is an array of 2 index: genome, and fitness
    # population[0] = [[1,2,3,4,5,5], 1500] -> genome, fitness 
    g1 = population[g1_index][0]
    g2 = population[g2_index][0]
    desired_length = len(g1)

    # Genome: genes_list, fitness which is None initially
    genes_list = []
    child_genome = [genes_list, None]
    g1_copy, g2_copy = g1[:], g2[:]
    intersect = set(g1).intersection(g2)

    for val in intersect:
        genes_list.append(val)

    while len(genes_list) < desired_length:
        rand_val = random.random()
        if rand_val < 0.7 and len(g1_copy) > 0:
            g1_val = g1_copy.pop()
            while g1_val in intersect and len(g1_copy) > 0:
                g1_val = g1_copy.pop()
            if g1_val not in intersect:
                genes_list.append(g1_val)
        else:
            g2_val = g2_copy.pop()
            while g2_val in intersect and len(g2_copy) > 0:
                g2_val = g2_copy.pop()
            if g2_val not in intersect:
                genes_list.append(g2_val)

    genes_list = func_mutate(genes_list)
    child_genome[0] = genes_list
    population.append(child_genome)
    return population

population = initiate_population([])
for generation in range(MAX_GENERATIONS):
    population = evaluate_population(population)
    # store_population(self.population, self.innovation_count, "population")
    for genome in population:
        # IF genome fitness is max_fitness
        if (genome[1] == MAX_FITNESS):
            return genome

    # Kill off the worst of the genomes. # Return the population of the fittest genomes
    population = natural_selection(population)
    survived_num = len(population)

    while len(population) < POPULATION_SIZE:
        if len(population) == 1 or not (random.random() < CROSSOVER_CHANCE):
            copied_genome = copy_genome(population[random.randint(0, survived_num - 1)])
            mutate(copied_genome)
            population.append(copied_genome)
        else:
            g1_index = random.randrange(0, survived_num)
            g2_index = random.randrange(0, survived_num)
            crossover(g1_index, g2_index)

population = evaluate_population(population)
return max(population, key=lambda genome: genome[1])
