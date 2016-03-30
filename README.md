# compnetsproject
How evolution works.

Initially, we create a population of x number of genomes. Let's say 100.
Each genome is basically a list of indexes of network edges that we want to cut.

Example. 
Genome_1 = [0, 18, 19, 26, 55, ...] however long we make each genome. Currently 1% of total edges.
After we've created the initial 100 genomes, we then find the fitness of each one.
This initial population of 100 is the first generation.

After finding the fitness values by interfacing with the Neural Network in matlab,
we then perform natural_selection (culling the underperforming or worst fit genomes)
The ones that are left are crossed over and mutated to create new genomes.

Example.
After the culling, let's say 10 genomes are left. These genomes are kept. In crossover,
2 random genomes are continually chosen until we have the population size again,
and their genes are crossed over in this process:
genome1 = [0, 14, 25, 10] <--- most fit out of these 2
genome2 = [9, 10, 14, 3]
child_genome = []
The genes that are the same for both, we put in the child_genome. So in this case:
child_genome = [10, 14], then we randomly, with the fittest function having the highest weight,
choose the remaining genes(weight indexes) from these 2 genomes.