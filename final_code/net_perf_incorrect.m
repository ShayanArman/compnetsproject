%Gets network fitness as percentage of correctly classified digits
function [fitness] = net_perf_incorrect(network, features, target)
    networkOutputs = vec2ind(network(features));
    networkTargets = vec2ind(target);
    fitness = sum(networkOutputs ~= networkTargets)/length(features);
end
