%Returns the normalized percentage of layer cuts at each layer in the
%network. in_one is input to first autoencoder, one_two is between
%autoencoders, two_three is autoencoder to softmax
%Should probably work on making this generic to any network toplogy
function [cuts_1, cuts_2, cuts_3] = percentage_layer_cuts(genome)
    b1 = sum(genome >= 1 & genome <= 100); % / 100;
    w1 = sum(genome >= 101 & genome <= 78500);  % / 78400;
    b2 = sum(genome >= 78501 & genome <= 78550); % / 50;
    w2 = sum(genome >= 78551 & genome <= 83550); % / 5000;
    b3 = sum(genome >= 83551 & genome <= 83560); % / 10;
    w3 = sum(genome >= 83561 & genome <= 84060); % / 500;
    
    cuts_1 = (b1 + w1)/(100 + 78400);
    cuts_3 = (b2 + w2)/(50 + 5000);
    cuts_2 = (b3 + w3)/(10 + 500);
end

