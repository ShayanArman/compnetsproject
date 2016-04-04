%Applies disturbance to network between each training iteration
function [disturbed_net, perf] = train_persistent_disturbance(net, x, t, epochs, disturbance, perf_func)
    perf = zeros(epochs, 1);
    disturbed_net = disturbance(net);
    disturbed_net.trainParam.epochs = 1;
    disturbed_net.trainParam.showWindow = false;
    for i = 1:epochs
        perf(i) = perf_func(disturbed_net);
        disturbed_net = train(disturbed_net, x, t);
        disturbed_net = disturbance(disturbed_net);
    end
end