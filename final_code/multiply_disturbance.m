%Takes all of the weights denoted by indicies in array boosts
%Multiplies them by the boost factor, reapplies them to the network
function [disturbed_net] = multiply_disturbance(net, weights_ind, boost)
    weights = getwb(net);
    weights(weights_ind) =  weights(weights_ind) * boost;
    disturbed_net = setwb(net, weights);
end