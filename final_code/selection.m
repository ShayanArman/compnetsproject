function [population, fitness] = selection(population, fitness, survival)
    population_size = size(population, 1);
    survival_size = int32(survival * population_size);
    [fitness, sort_ind] = sort(fitness, 'ascend');
    population = population(sort_ind, :);
    population = population(1:survival_size, :);
    fitness = fitness(1:survival_size);
end
