function [in_one, one_two, two_three] = percentage_layer_cuts( network )
%Returns the normalized percentage of layer cuts at each layer in the
%network. in_one is input to first autoencoder, one_two is between
%autoencoders, two_three is autoencoder to softmax
    figure
    
    weights = getwb(network);
    
    [~, input_weights, layer_weights] = separatewb(network, weights);
    
    in_one = length(find(input_weights{1} == 0)) / numel(input_weights{1});
    one_two = length(find(layer_weights{2} == 0)) / numel(layer_weights{2});
    two_three = length(find(layer_weights{6} == 0)) / numel(layer_weights{6});    
end

