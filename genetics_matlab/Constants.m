classdef Constants
    properties (Constant)
        NUM_EDGES = 84060;
        NUM_CUTS_PERCENT = .01;
        MUTATION_CHANCE = .01;
        SURVIVAL_PERCENT = 0.30;
        CROSSOVER_CHANCE = 0.75;
        POPULATION_SIZE = 20;
        MAX_GENERATIONS = 10;
        CHOOSE_FROM_FITTEST_CHANCE = 0.01; 
    end
end