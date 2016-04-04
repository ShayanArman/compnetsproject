function [ ] = plot_network( network )
    %Plot the network as an undirected graph, showing the 
    %connections that were CUT
    num_inputs = 784;
    num_layer_1 = 100;
    num_layer_2 = 50;
    num_outputs = 10;

    weights = getwb(network);
    [bias, input_weights, layer_weights] = separatewb(network, weights);

    s = [];
    t = [];
    for i=1:784
        connected = find(input_weights{1}(:, i) == 0);
        if ~isempty(connected)
                s = [s ones(1, length(connected))*i];
                t = [t connected' + num_inputs];
        end
    end

    for i=1:100
        connected = find(layer_weights{2}(:, i) == 0);
        if ~isempty(connected)
            s = [s ones(1,length(connected)) * i + num_inputs];
            t = [t connected' + num_inputs + num_layer_1];
        end
    end

    for i=1:50
        connected = find(layer_weights{6}(:, i) == 0);
        if ~isempty(connected)
            s = [s ones(1, length(connected))*i + num_inputs+num_layer_1];
            t = [t connected' + num_inputs + num_layer_1 + num_layer_2];
        end
    end
    G = graph(s,t);
    xData = [ones(1, 784) ones(1, 100)*2 ones(1, 50)*3 ones(1, 10)*4];
    yData = [1:784 300:399 325:374 345:354];
    figure(1)
    h = plot(G);
    h.XData = xData;
    h.YData = yData;
    h.LineWidth = 0.02;
end

