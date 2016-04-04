%Applies boosting to network between each training iteration
%Returns performance over iterations as percent correctly classified
%First element in array is always performance of original network passed in
function [trained_boosted_network, perf] = train_with_boost(network, train_data, train_target, test_data, test_target, boosts, boost_factor, iterations)
    perf = zeros(iterations, 1);
    perf(1) = network_fitness(network, test_data, test_target);
    
    trained_boosted_network = network;    
    trained_boosted_network.trainParam.epochs = 1;
    trained_boosted_network.trainParam.showWindow = false;
    trainData = images2netinput(train_data);
    
    trained_boosted_network = boost_network(trained_boosted_network, boosts, boost_factor);
    perf(2) = network_fitness(trained_boosted_network, test_data, test_target);
        
    for i = 3:iterations
        trained_boosted_network = train(trained_boosted_network, trainData, train_target);
        trained_boosted_network = boost_network(trained_boosted_network, boosts, boost_factor);
        perf(i) = network_fitness(trained_boosted_network, test_data, test_target);
    end
end

%Takes all of the weights denoted by indicies in array boosts
%Multiplies them by the boost factor, reapplies them to the network
function [boosted_network] = boost_network(network, boosts, boost_factor)
    weights = getwb(network);
    weights(boosts) =  weights(boosts) * boost_factor;
    boosted_network = setwb(network, weights);
end


