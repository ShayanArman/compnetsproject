function [population] = initiate_population(genome_range, genome_size, population_size)
    population = zeros(population_size, genome_size);
    for i = 1:population_size
        population(i, :) = randperm(genome_range, genome_size);
    end
end