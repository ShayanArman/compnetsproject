clear all;
close all;
import neurop_project.m; %so we can call fitness function

function genetic_evolution
    import Constants;
    [population, fitness] = initiate_population();
    for generation = 1:Constants.MAX_GENERATIONS
        fitness = evaluate_population(population, fitness);
        % Keep only the best performing genomes.
        [population, fitness] = natural_selection(population, fitness);
        survived_num = size(population, 1);

        while size(population, 1) < Constants.POPULATION_SIZE
            if size(population, 1) == 1 || rand(1) > Constants.CROSSOVER_CHANCE
                genome_index = max(1, int8(rand(1)*survived_num));
                copied_genome = [];
                copied_genome(:, :) = population(genome_index, :);
            else
                g1_index = max(1, int8(rand(1)*survived_num));
                g2_index = max(1, int8(rand(1)*survived_num));
                [population, fitness] = crossover(g1_index, g2_index, population, fitness);
            end
        end
    end

    fitness = evaluate_population(population, fitness);
    disp(num_evens(population, fitness));
    % Crossover tests.
    % cats = [];
    % cats(1, :) = randperm(10, 5);
    % cats(2, :) = randperm(10, 5);
    % crossover(1, 2, cats, fitness);

function [population, fitness] = initiate_population()
    num_weights_cut = max(1, floor(Constants.NUM_EDGES * Constants.NUM_CUTS_PERCENT));
    population = [];
    fitness = [];
    for i =1:Constants.POPULATION_SIZE
        population(i, :) = randperm(Constants.NUM_EDGES, num_weights_cut);
        fitness(i) = 0.0;
    end

function [evens_count] = num_evens(population, fitness)
    [population, fitness] = sort_two_vectors(population, fitness);
    genome = population(1, :);
    evens_count = 0.0;
    for i =1:size(genome, 2)
        if mod(genome(1, i), 2) == 0
            evens_count = evens_count + 1;
        end
    end

function [genome] = mutate(genome)
    len_genome = size(genome);
    max = Constants.NUM_EDGES;
	for k = 1:size(genome, 2)
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
    [population, fitness] = sort_two_vectors(population, fitness);
    population = population(1:survival_size, :);
    fitness = fitness(:, 1:survival_size);

function [fitness] = evaluate_population(population, fitness)
    population_size = size(population);
    for i = 1:population_size(1)
        %cut_fitness = ga fitness function from neurosci_project
        fitness(i) = cut_fitness(population(i, :));
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

function [population, fitness] = crossover(g1_index, g2_index, population, fitness)
    g1 = population(g1_index, :);
    g2 = population(g2_index, :);
    desired_length = size(g1, 2);

    genes_list = [];
    g1_copy = g1(:, :);
    g2_copy = g2(:, :);
    intersect_g1_g2 = intersect(g1, g2);
    possible_g1 = setdiff(g1, intersect_g1_g2);
    possible_g2 = setdiff(g2, intersect_g1_g2);
    genes_list(:, :) = intersect_g1_g2(:, :);

    while size(genes_list, 2) < desired_length
        if rand(1) < Constants.CHOOSE_FROM_FITTEST_CHANCE && size(possible_g1, 2) > 0
            genes_list(size(genes_list, 2) + 1) = possible_g1(1);
            if size(possible_g1, 2) > 1
                possible_g1 = possible_g1(2:end);
            else
                possible_g1 = [];
            end
        elseif size(possible_g2, 2) > 0
            genes_list(1, size(genes_list, 2) + 1) = possible_g2(1);
            if size(possible_g2, 2) > 1
                possible_g2 = possible_g2(2:end);
            else
                possible_g2 = [];
            end
        end
    end
    genes_list = mutate(genes_list);
    population(size(population, 1) + 1, :) = genes_list;
    fitness(:, size(fitness, 2) + 1) = 0.0;

function [population, fitness] = sort_two_vectors(population, fitness)
    [fit_sorted, indices] = sort(fitness, 'descend');
    fitness = fit_sorted;
    population = population(indices, :);
