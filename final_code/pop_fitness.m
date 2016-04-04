function [fitness] = pop_fitness(population, genome_fitness)
    population_size = size(population, 1);
    fitness = zeros(population_size, 1);
    for i = 1:population_size
        fitness(i) = genome_fitness(population(i, :));
    end
end