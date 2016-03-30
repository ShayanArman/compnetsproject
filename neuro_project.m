function neuro_project
    clear; close all; clc;
    rng('default')

    [xTrainImages, tTrain] = digittrain_dataset;
    [xTestImages, tTest] = digittest_dataset;

    autoenc1 = trainAutoencoder(xTrainImages,100, 'MaxEpochs', 50, 'ShowProgressWindow',false);
    feat1 = encode(autoenc1,xTrainImages);

    autoenc2 = trainAutoencoder(feat1,50, 'MaxEpochs', 50, 'ShowProgressWindow',false);
    feat2 = encode(autoenc2,feat1);

    softnet = trainSoftmaxLayer(feat2,tTrain,'MaxEpochs', 50,'ShowProgressWindow',false);
    
    deepnet = stack(autoenc1,autoenc2,softnet);
    trained_boosted_network.trainParam.epochs = 50;
    trained_boosted_network.trainParam.showWindow = false;
    deepnet = train(deepnet, images2netinput(xTrainImages), tTrain);
    
    weightsBackup = getwb(deepnet);
    num_weights = length(weightsBackup);
    
    figure
    hold on
    
    boost_factor = 0; %Only perform cuts
   
    %No connections cut, the control case
    weights_to_boost = randperm(num_weights, int32(0 * num_weights));
    [degraded_net_0, perf] = train_with_boost(deepnet, xTrainImages, tTrain, xTestImages, tTest, weights_to_boost, boost_factor, 50);
    plot(0:49, perf);
    
    %five percent of connections cut
    weights_to_boost = randperm(num_weights, int32(0.05 * num_weights));
    [degraded_net_5, perf] = train_with_boost(deepnet, xTrainImages, tTrain, xTestImages, tTest, weights_to_boost, boost_factor, 50);
    plot(0:49, perf);
    
    %10 percent of connections cut
    weights_to_boost = randperm(num_weights, int32(0.1 * num_weights));
    [degraded_net_10, perf] = train_with_boost(deepnet, xTrainImages, tTrain, xTestImages, tTest, weights_to_boost, boost_factor, 50);
    plot(0:49, perf);
    
    legend('original', '5% cut', '10% cut');
    xlabel('Training Iteration');
    ylabel('Performance');
    title('Performance of various degradations over training iterations');
    
    testNet = degraded_net_10;
    
end

%converts images to array for input into neurons
function [test_data] = images2netinput(test_images)
    inputSize = numel(cell2mat(test_images(1)));
    test_data = zeros(inputSize,numel(test_images));
    for i = 1:numel(test_images)
        test_data(:,i) = test_images{i}(:);
    end
end

%Gets network fitness as percentage of correctly classified digits
function [fitness] = network_fitness(network, image_data, target)
    testData = images2netinput(image_data);
    y = network(testData);
    [~, networkOutputs] = max(y);
    [~, networkTargets] = max(target);
    fitness = sum(networkOutputs == networkTargets)/length(image_data);
end

%Takes all of the weights denoted by indicies in array boosts
%Multiplies them by the boost factor, reapplies them to the network
function [boosted_network] = boost_network(network, boosts, boost_factor)
    weights = getwb(network);
    weights(boosts) =  weights(boosts) * boost_factor;
    boosted_network = setwb(network, weights);
end

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
    for i = 2:iterations
        trained_boosted_network = boost_network(trained_boosted_network, boosts, boost_factor);
        trained_boosted_network = train(trained_boosted_network, trainData, train_target);
        perf(i) = network_fitness(trained_boosted_network, test_data, test_target);
    end
    trained_boosted_network = boost_network(trained_boosted_network, boosts, boost_factor);
end
