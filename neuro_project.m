function neuro_project
    clear; close all; clc;
    rng('default')
    
    initial_train_iterations = 50;

    [xTrainImages, tTrain] = digittrain_dataset;
    [xTestImages, tTest] = digittest_dataset;

    deepLayer1 = trainAutoencoder(xTrainImages,100, 'MaxEpochs', initial_train_iterations, 'ShowProgressWindow',false);
    featLayer1 = encode(deepLayer1,xTrainImages);

    deepLayer2 = trainAutoencoder(featLayer1,50, 'MaxEpochs', initial_train_iterations, 'ShowProgressWindow',false);
    featLayer2 = encode(deepLayer2,featLayer1);

    deepLayer3 = trainSoftmaxLayer(featLayer2,tTrain,'MaxEpochs', initial_train_iterations,'ShowProgressWindow',false);
    
    deepnet = stack(deepLayer1,deepLayer2,deepLayer3);
    trained_boosted_network.trainParam.epochs = initial_train_iterations;
    trained_boosted_network.trainParam.showWindow = false;
    deepnet = train(deepnet, images2netinput(xTrainImages), tTrain);
    
    boost_factor = 0; %Only perform cuts
    
    resp_degredations = 0:0.05:0.25;
    perf_response = test_degredations(deepnet, xTrainImages, tTrain, xTestImages, tTest, resp_degredations, boost_factor, 2, 10);    
    
    figure
    hold on
    perf_response = squeeze(perf_response(:,2,:))';
    boxplot(perf_response, 'Labels', resp_degredations)
    xlabel('degredation percent')
    ylabel('performance')
    title('Instaneous Network performance after degredation, 10 iterations')
    hold off
    
    
    boost_train_iterations = 50;
    recov_degradations = [0, 0.05, 0.1, 0.5, 0.9];
    perf_recovery = test_degredations(deepnet, xTrainImages, tTrain, xTestImages, tTest, recov_degradations, boost_factor, boost_train_iterations, 1);    
    
    figure
    hold on;
    perf_recovery = perf_recovery';
    plot(0 : 1 : boost_train_iterations-1, perf_recovery);
    legend(strread(num2str(recov_degradations),'%s'));
    xlabel('Training Iteration');
    ylabel('Performance');
    title('Performance of various degradations over training iterations');
    hold off
    
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

%Tests an array of degredation values, each trained for certain iterations,
%Attempted num_attempts number of times. Fills in 3d performance array.
function [perf] = test_degredations(network, train_images, train_target, test_images, test_target, degrade_amount, boost_factor, train_iterations, num_attempts)
    num_weights = length(getwb(network));
    perf = zeros(length(degrade_amount), train_iterations, num_attempts);
    for i = 1:num_attempts
        j_ind = 1;
        for j = degrade_amount
            weights_to_boost = randperm(num_weights, int32(j * num_weights));
            [~, perf_j] = train_with_boost(network,...
                                    train_images, train_target, test_images, test_target,...
                                    weights_to_boost, boost_factor, train_iterations);
            perf(j_ind, :, i) = perf_j;
            j_ind = j_ind + 1;
        end
    end
end