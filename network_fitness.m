%Gets network fitness as percentage of correctly classified digits
function [fitness] = network_fitness(network, image_data, target)
    testData = images2netinput(image_data);
    y = network(testData);
    [~, networkOutputs] = max(y);
    [~, networkTargets] = max(target);
    fitness = sum(networkOutputs == networkTargets)/length(image_data);
end