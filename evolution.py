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
SURVIVAL_PERCENT = 0.3
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
        self.brain = None
        self.gid_count = 0

    def get_new_gid(self):
        self.gid_count += 1
        return self.gid_count - 1

    def initiate_population(self):
        for i in range(POPULATION_SIZE):
            genome = Genome(self)
            genome.init_genes()
            self.population.append(genome)

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
                    g1_index = random.randint(0, survived_num - 1)
                    g2_index = random.randint(0, survived_num - 1)
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
        # ensure g1 has greater fitness
        if (g2.fitness > g1.fitness):
            g1, g2 = g2, g1

        child = Genome(self)
        child.set_id()

        innovations2 = {}
        for gene in g2.genes:
            innovations2[gene.innovation] = gene

        for gene in g1.genes:
            gene1 = gene
            gene2 = innovations2.get(gene1.innovation)
            if (gene2 and round(random.random()) == 1):
                child.genes.append(gene2)
            else:
                child.genes.append(gene1)

        child.max_neuron = max(g1.max_neuron, g2.max_neuron)
        child.mutate()
        self.population.append(child)

    def natural_selection(self):
        """
        Selects top SURVIVAL_SIZE-most fit genomes - Kills the rest
        """
        survival_size = max(1, int(SURVIVAL_PERCENT * len(self.population)))
        self.population = sorted(self.population, key=lambda genome: genome.fitness, reverse=True)[:survival_size]
        return len(self.population)


class Genome:
    def __init__(self, evolution):
        self.genes = []
        self.fitness = None
        self.evolution = evolution

   	def init_genes(self):
   		gene_dict = {}
   		num_genes = int(NUM_CUTS_PERCENT * NUM_EDGES)
   		
   		while len(gene_dict) < num_genes:
   			edge_index = random.randrange(NUM_EDGES)
   			gene_dict[edge_index] = True
   		self.genes = gene_dict.keys()   		

    def mutate(self):
        # call all different mutates
        if random.random() < CONNECTION_MUTATION_CHANCE:
            self.point_mutate()

        for i in range(int(math.floor(LINK_MUTATION_CHANCE))):
            self.link_mutate(False)
        if random.random() < (LINK_MUTATION_CHANCE % 1):
            self.link_mutate(False)

        if random.random() < BIAS_MUTATION_CHANCE:
            self.link_mutate(True)

        if random.random() < NODE_MUTATION_CHANCE:
            self.node_mutate()

    def node_mutate(self):
        """Adds new node between two nodes"""
        if len(self.genes) == 0:
            return
        self.max_neuron += 1
        gene_index = random.randint(0, len(self.genes) - 1)
        gene = self.genes[gene_index]
        del self.genes[gene_index]

        gene1 = gene.copy_gene()
        gene1.out_index = self.max_neuron
        gene1.weight = 1
        gene1.innovation = self.evolution.new_innovation()
        self.genes.append(gene1)

        gene2 = gene.copy_gene()
        gene2.in_index = self.max_neuron
        gene2.innovation = self.evolution.new_innovation()
        self.genes.append(gene2)

        # self.logger.info("Node Mutate")

    def link_mutate(self, force_bias):
        """Creates new connection between nodes"""
        neurons = {}

        for o in range(NUM_OUTPUTS):
            neurons[MAX_NODES + o] = True

        for gene in self.genes:
            # avoid inputs and bias
            if gene.in_index > NUM_INPUTS:
                neurons[gene.in_index] = True
            if gene.out_index > NUM_INPUTS:
                neurons[gene.out_index] = True

        neuron_2 = random.choice(neurons.keys())

        if not force_bias:
            # add in inputs and bias for finding neuron_1
            for i in range(NUM_INPUTS + 1):
                neurons[i] = True
            neuron_1 = random.choice(neurons.keys())
        else:
            neuron_1 = NUM_INPUTS - 1

        if any(filter(lambda x: (x.in_index == neuron_1) and (x.out_index == neuron_2), self.genes)):
            # connection already exists
            return

        new_connection = Gene()
        new_connection.in_index = neuron_1
        new_connection.out_index = neuron_2
        new_connection.innovation = self.evolution.new_innovation()
        new_connection.weight = random.random() * 4 - 2

        self.genes.append(new_connection)

        # self.logger.info("Link Mutate")

    def point_mutate(self):
        """Changes all weights"""
        for gene in self.genes:
            if random.random() < WEIGHT_MUTATION_CHANCE:
                gene.weight += WEIGHT_STEP * random.random() * 2 - WEIGHT_STEP
            else:
                # limit weight range to (-2,2)
                gene.weight = random.random() * 4 - 2

        # self.logger.info("Point Mutate")

    def copy_genome(self):
        genome = Genome(self.evolution)
        genome.genes = self.genes
        genome.fitness = self.fitness

        return genome


class Neuron:
    def __init__(self):
        # A list containing edge objects with input and output neurons
        self.incoming = []
        self.value = 0.0


class NeuralNet:
    def __init__(self, genome):
        self.max_neuron = genome.max_neuron
        self.neurons = {}
        self.generate_network(genome)

    def generate_network(self, genome):
        """
        Create a network of neurons from the genome.

        """
        for i in range(NUM_INPUTS + 1):
            self.neurons[i] = Neuron()

        for o in range(NUM_OUTPUTS):
            self.neurons[MAX_NODES + o] = Neuron()

        genome.genes = sorted(genome.genes, key=lambda gene: gene.out_index)

        for gene in genome.genes:
            # TODO: Add if gene.enabled here if we include enabled genes.
            if self.neurons.get(gene.out_index) is None:
                self.neurons[gene.out_index] = Neuron()

            neuron = self.neurons[gene.out_index]
            neuron.incoming.append(gene)

            if self.neurons.get(gene.in_index) is None:
                self.neurons[gene.in_index] = Neuron()

    def evaluate_network(self, inputs):
        """
        Using the inputs to this Network. Evaluate the output of the Network.

        """
        inputs.append(1)

        if len(inputs) != NUM_INPUTS + 1:
            # self.logger.error("Incorrect number of neural network inputs.")
            return None

        for i in range(NUM_INPUTS + 1):
            self.neurons[i].value = inputs[i]

        for index, neuron in self.neurons.iteritems():
            input_sum = 0
            for j in range(len(neuron.incoming)):
                incoming = neuron.incoming[j]
                incoming_neuron = self.neurons[incoming.in_index]
                input_sum = input_sum + incoming.weight * incoming_neuron.value

            if len(neuron.incoming) > 0:
                neuron.value = self.sigmoid(input_sum)

        max_val = -2
        nop_flag = True

        for o in range(NUM_OUTPUTS):
            output_val = self.neurons[MAX_NODES + o].value
            if output_val > 0:
                nop_flag = False
            if output_val > max_val:
                max_val = output_val
                output_neuron_index = o

        if nop_flag:
            output_neuron_index = 4

        return interface.MoveRequests(output_neuron_index)

    def sigmoid(self, x):
        return 2 / (1 + math.exp(-4.9 * x)) - 1


def store_genome(genome, name):
    with open(name + ".pkl", "w") as f:
        representation = {
            "genes": genome.genes,
            "fitness": genome.fitness,
            "max_neuron": genome.max_neuron
        }
        pickle.dump(representation, f)


def store_population(population, innovation_count, name):
    population_rep = {
        "genomes": [],
        "innovation_count": innovation_count
    }
    for genome in population:
        genome_rep = {
            "genes": genome.genes,
            "fitness": genome.fitness,
            "max_neuron": genome.max_neuron
        }
        population_rep["genomes"].append(genome_rep)
    with open(name + ".pkl", "w") as f:
        pickle.dump(population_rep, f)


