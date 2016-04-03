function genetic_evolution
    import Constants;
    num_weights_cut = max(1, floor(Constants.NUM_EDGES * Constants.NUM_CUTS_PERCENT));
    population = [];
    fitness = [];
    for i =1:Constants.POPULATION_SIZE
        population(i, :) = randperm(Constants.NUM_EDGES, num_weights_cut);
        fitness(i) = 0.0;
    end
    population, fitness = evaluate_population(population, fitness);
    disp(population(1,:));
    disp(fitness(1));

function [genome] = mutate(genome)
    len_genome = size(genome);
    max = Constants.NUM_EDGES;
	for k = 1:len_genome(2)
		randVal = rand(1); %1 value random generator
		if randVal < Constants.MUTATION_CHANCE
			newI = floor(randVal*max);
			while ismember(newI, genome)
				newI = floor(rand(1)*max);
			end
			genome(k) = newI;
        end
    end

function [population, fitness] = natural_selection(population, fitness)
    population_size = size(population);
    survival_size = max(1, floor(Constants.SURVIVAL_PERCENT * population_size(1)));
    population, fitness = sort_two_vectors(population, fitness);
    population = population(1:survival_size, :, :);
    fitness = fitness(1:survival_size, :, :);

function [fitness] = evaluate_population(population, fitness)
    population_size = size(population);
    for i = 1:population_size(1)
        fitness(i) = get_fitness(population(i, :));
    end
    
function [fitness_val] = get_fitness(genome)
    genome_size = size(genome);
    score = 0.0;
    for j = 1:genome_size(2)
        if mod(genome(j), 2) == 0
            score = score + 10000;
        else
            score = score - 10000;
        end
    end
    fitness_val = score;
        
function [population, fitness] = sort_two_vectors(population, fitness)
    [fit_sorted, indices] = sort(fitness, 'descend');
    fitness = fit_sorted;
    population = population(indices, :, :);
