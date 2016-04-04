%Singular float fitness variable for GA
function [fitness] = cut_fitness(network, train_images, train_target, test_images, test_target, degrade_vector, boost_factor, train_iterations)
    [~, perf_j] = train_with_boost(network,...
                            train_images, train_target, test_images, test_target,...
                            degrade_vector, boost_factor, train_iterations);
    fitness = perf_j(train_iterations);
end
