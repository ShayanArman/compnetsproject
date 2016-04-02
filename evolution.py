import logging
import math
# import numpy
import random
import threading
# import time
import pickle

# BIO
NUM_EDGES = 84060
NUM_CUTS_PERCENT = .01
MUTATION_CHANCE = .01
SURVIVAL_PERCENT = 0.30
CROSSOVER_CHANCE = 0.75
POPULATION_SIZE = 20
MAX_GENERATIONS = 100


# Functional initialize genes
def gen_random_genes():
    gene_dict = {}
    num_genes = int(NUM_CUTS_PERCENT * NUM_EDGES)
    while len(gene_dict) < num_genes:
        edge_index = random.randrange(NUM_EDGES)
        gene_dict[edge_index] = True
    genes = gene_dict.keys()
    return genes


def initiate_population(population):
    random_genes = []
    for i in range(POPULATION_SIZE):
        random_genes = gen_random_genes()
        # Fitness of 0
        population.append([random_genes, 0.0])
    return population


def fitness(genes):
    score = 0.0
    for x in genes:
        if x % 2 == 0:
            score -= 1000
        else:
            score += 1000
    return score


def number_of_evens(genes):
    evens = 0
    for gene in genes:
        if gene % 2 == 0:
            evens += 1
    return evens


# TODO: THIS IS WHERE WE GET THE FITNESS. CONNECT WITH THE MATLAB CODE.
def evaluate_population(population):
    for genome in population:
        # genome -> [genes, fitness]
        if genome[1] == 0:
            # TODO call mayanks function here for fitness(genome[0])
            genome[1] = fitness(genome[0])
    return population


def natural_selection(population):
    """
    Selects top SURVIVAL_SIZE-most fit genomes - Kills the rest

    """
    survival_size = max(1, int(SURVIVAL_PERCENT * len(population)))
    # Population looks like [[[1,2,3,4,5,5], 1500],
    # [[1,2,3,4,5,5], 1500], [[1,2,3,4,5,5], 1500]]
    # Each index is an array of 2 index: genome, and fitness
    # population[0] = [[1,2,3,4,5,5], 1500] -> genome, fitness
    population = sorted(
        population, key=lambda genome: genome[1], reverse=True)[:survival_size]
    return population


def copy_genome(genome):
    new_genome = [genome[0][:], genome[1]]
    return new_genome


# Functional mutate function
def mutate(genes):
    """
    genome is an array of genes: [1, 15, 18, 59]
    its indexes are the weights to be cut.

    """
    # call all different mutates
    genes_dict = {}
    for gene in genes:
        genes_dict[gene] = True
    for x in range(len(genes)):
        if random.random() < MUTATION_CHANCE:
            new_index = random.randrange(NUM_EDGES)
            while new_index in genes_dict:
                new_index = random.randrange(NUM_EDGES)
            genes[x] = new_index
    return genome


def crossover(g1_index, g2_index, population):
    """
    Creates offspring from fittest genomes and
    mutates children to form new generation
    """
    # Population looks like [[[1,2,3,4,5,5], 1500],
    # [[1,2,3,4,5,5], 1500], [[1,2,3,4,5,5], 1500]]
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
    genes_list = mutate(genes_list)
    child_genome[0] = genes_list
    population.append(child_genome)
    return population

population = initiate_population([])
for generation in range(MAX_GENERATIONS):
    population = evaluate_population(population)

    # Kill off the worst of the genomes.
    # Return the population of the fittest genomes
    population = natural_selection(population)
    survived_num = len(population)

    while len(population) < POPULATION_SIZE:
        if len(population) == 1 or not (random.random() < CROSSOVER_CHANCE):
            copied_genome = copy_genome(
                population[random.randint(0, survived_num - 1)])
            copied_genome = mutate(copied_genome[0])
            population.append(copied_genome)
        else:
            g1_index = random.randrange(0, survived_num)
            g2_index = random.randrange(0, survived_num)
            population = crossover(g1_index, g2_index, population)

population = evaluate_population(population)
print number_of_evens(max(population, key=lambda genome: genome[1])[0])
