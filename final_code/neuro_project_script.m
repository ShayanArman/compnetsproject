%% Setup Workspace
clear; close all; clc; rng(0);

%% Setup Training and Testing Data
inputSize = 784;
[xTrainImages, tTrain] = digittrain_dataset;
[xTestImages, tTest] = digittest_dataset;
xTrain = reshape(cell2mat(xTrainImages), inputSize, length(xTrainImages));
xTest =  reshape(cell2mat(xTestImages), inputSize, length(xTestImages));
perf_func = @(net) net_perf_incorrect(net, xTest, tTest);

%% Create and Train Initial Network
ees_net = untrained_enc_enc_soft_net(xTrain(:,1), tTrain(:,1));
num_weights = length(getwb(ees_net));

%Track performance and weights while training
epoch_step = 20;
train_iterations = 10;

ees_epochs = 0 : epoch_step : (train_iterations - 1) * epoch_step;
ees_perf_train = zeros(1, train_iterations);
ees_perf_test = zeros(1, train_iterations);
ees_weights = zeros(num_weights, train_iterations);

ees_net.trainParam.epochs = epoch_step;

for i = 1 : train_iterations
    ees_perf_train(i) = net_perf_incorrect(ees_net, xTrain, tTrain);
    ees_perf_test(i) = net_perf_incorrect(ees_net, xTest, tTest);
    ees_weights(:, i) = getwb(ees_net);
    ees_net = train(ees_net,xTrain,tTrain,'useParallel','yes','UseGPU','yes');
end

%Line plot of original network performance over training iterations
figure(1);
title('Error of original network vs training iteration');
hold on;
axis([0 max(ees_epochs) 0 1]);
plot(ees_epochs, ees_perf_train, 'b--') 
plot(ees_epochs, ees_perf_test, 'r');
legend('training performance','testing performance');
xlabel('training iteration');
ylabel('percentage of misclassified digits');

%% Check network performance over various degredations
disturbances = [0.02, 0.1, 0.3];
attempts = 4;
recover_epochs = 50;

d_perf_mean = zeros(length(disturbances), recover_epochs);
d_perf_best = zeros(length(disturbances), recover_epochs);

d_ind = 1;
d_perf_0 = perf_func(ees_net); 
for d = disturbances;
    this_perf = zeros(attempts, recover_epochs);
    for i = 1:attempts
        dist_net = ees_net;
        cut_weights = randperm(num_weights, int32(d * num_weights));
        cut_func = @(net) multiply_disturbance(net, cut_weights, 0);
        [dist_net, this_perf_attempt] = train_persistent_disturbance(dist_net,...
                                        xTrain,tTrain, recover_epochs, cut_func, perf_func);
        this_perf(i, :) = this_perf_attempt;
    end
    d_perf_mean(d_ind,:) = mean(this_perf);
    d_perf_best(d_ind,:) = min(this_perf);
    d_ind = d_ind + 1;
end

figure(2)
title('Error after cuts vs recovery iteration')
hold on
plot(0:recover_epochs, [d_perf_0 d_perf_mean(1,:)], 'r--');
plot(0:recover_epochs, [d_perf_0 d_perf_mean(2,:)], 'g--');
plot(0:recover_epochs, [d_perf_0 d_perf_mean(3,:)], 'b--');
plot(0:recover_epochs, [d_perf_0 d_perf_mean(4,:)], 'k--');
plot(0:recover_epochs, [d_perf_0 d_perf_best(1,:)], 'r');
plot(0:recover_epochs, [d_perf_0 d_perf_best(2,:)], 'g');
plot(0:recover_epochs, [d_perf_0 d_perf_best(3,:)], 'b');
plot(0:recover_epochs, [d_perf_0 d_perf_best(4,:)], 'k');
legend([strread(num2str(disturbances),'%s');strread(num2str(disturbances),'%s')]);
xlabel('recovery iteration');
ylabel('percentage of misclassified digits');
axis([0 recover_epochs 0 1]);