%% Setup Workspace
clear; close all; clc; rng(0);

%% Setup Training and Testing Data
inputSize = 784;
[xTrainImages, tTrain] = digittrain_dataset;
[xTestImages, tTest] = digittest_dataset;
xTrain = reshape(cell2mat(xTrainImages), inputSize, length(xTrainImages));
xTest =  reshape(cell2mat(xTestImages), inputSize, length(xTestImages));
perf_func = @(this_net) net_perf_incorrect(this_net, xTest, tTest);

%% Create, Train, and Plot perofmrance offi Initial Network
tic;
net = untrained_enc_enc_soft_net(xTrain(:,1), tTrain(:,1));
num_weights = length(getwb(net));

epoch_step = 50;
max_epoch = 200;
train_iterations = max_epoch/epoch_step + 1;

epochs = 0 : epoch_step : max_epoch;
perf_train = zeros(1, train_iterations);
perf_test = zeros(1, train_iterations);
weights = zeros(num_weights, train_iterations);

net.trainParam.epochs = epoch_step;
net.trainParam.showWindow = false;

for i = 1 : train_iterations
    perf_train(i) = net_perf_incorrect(net, xTrain, tTrain);
    perf_test(i) = net_perf_incorrect(net, xTest, tTest);
    weights(:, i) = getwb(net);
    net = train(net,xTrain,tTrain);
    
    disp('section one progess')
    disp(i / train_iterations);
end

figure(1);
title('Error of original network vs training iteration');
hold on;
axis([0 max(epochs) 0 1]);
plot(epochs, perf_train, 'b--')
plot(epochs, perf_test, 'r');
legend('training performance','testing performance');
xlabel('training iteration');
ylabel('percentage of misclassified digits');
hold off;

disp('section one complete in')
disp(toc)

disp('total network performance at iteration')
disp([epoch_step+max_epoch 1-perf_func(net)])

%% Test and Plot impact of randomly permuted cuts by percentage of weights
tic
disturbances = [0.02, 0.15, 0.4, 0.8];
attempts = 4;
recover_epochs = 200;

d_perf_mean = zeros(length(disturbances), recover_epochs);
d_perf_best = zeros(length(disturbances), recover_epochs);

d_ind = 1;
d_perf_0 = perf_func(net);
for d = disturbances;
    this_perf = zeros(attempts, recover_epochs);
    for i = 1:attempts
        dist_net = net;
        cut_weights = randperm(num_weights, int32(d * num_weights));
        cut_func = @(net) multiply_disturbance(net, cut_weights, 0);
        [dist_net, this_perf_attempt] = train_persistent_disturbance(dist_net,...
            xTrain,tTrain, recover_epochs, cut_func, perf_func);
        this_perf(i, :) = this_perf_attempt;
        
        disp('section two  progess')
        disp(((d_ind - 1)*attempts + i) / (length(disturbances)*attempts));
    end
    d_perf_mean(d_ind,:) = mean(this_perf);
    d_perf_best(d_ind,:) = min(this_perf);
    d_ind = d_ind + 1;
end

figure(2)
title('Recovery after cuts on fully trained network, various cut amounts')
hold on
plot(0:recover_epochs, [d_perf_0 d_perf_mean(1,:)], 'r--');
plot(0:recover_epochs, [d_perf_0 d_perf_mean(2,:)], 'g--');
plot(0:recover_epochs, [d_perf_0 d_perf_mean(3,:)], 'b--');
plot(0:recover_epochs, [d_perf_0 d_perf_mean(4,:)], 'm--');

plot(0:recover_epochs, [d_perf_0 d_perf_best(1,:)], 'r');
plot(0:recover_epochs, [d_perf_0 d_perf_best(2,:)], 'g');
plot(0:recover_epochs, [d_perf_0 d_perf_best(3,:)], 'b');
plot(0:recover_epochs, [d_perf_0 d_perf_mean(4,:)], 'm');

legend([strread(num2str(disturbances),'%s');strread(num2str(disturbances),'%s')]);
xlabel('recovery iteration');
ylabel('percentage of misclassified digits');
axis([0 recover_epochs 0 1]);

disp('section two complete')
disp(toc)

%% Test and Plot impact of 10% disturbance by original network performance
tic
disturbance = 0.15;
attempts = 4;
recover_epochs = 50;

s_perf_mean = zeros(train_iterations, recover_epochs + 1);
s_perf_best = zeros(train_iterations, recover_epochs + 1);
for s = 1:train_iterations;
    this_perf = zeros(attempts, recover_epochs);
    for i = 1:attempts
        dist_net = setwb(net, weights(:,s));
        cut_weights = randperm(num_weights, int32(disturbance * num_weights));
        cut_func = @(this_net) multiply_disturbance(this_net, cut_weights, 0);
        [dist_net, this_perf_attempt] = train_persistent_disturbance(dist_net,...
            xTrain,tTrain, recover_epochs, cut_func, perf_func);
        this_perf(i, :) = this_perf_attempt;
        
        disp('section three progess')
        disp(((s - 1)*attempts + i) / (train_iterations*attempts));
    end
    s_perf_mean(s,:) = [perf_func(setwb(net, weights(:,s))) mean(this_perf)];
    s_perf_best(s,:) = [perf_func(setwb(net, weights(:,s))) min(this_perf)];
end

figure(3)
title('Recovery after 15% cut, various starting errors')
hold on
plot(0:recover_epochs, s_perf_mean(1,:), 'r--');
plot(0:recover_epochs, s_perf_mean(2,:), 'g--');
plot(0:recover_epochs, s_perf_mean(3,:), 'b--');
plot(0:recover_epochs, s_perf_mean(4,:), 'k--');
plot(0:recover_epochs, s_perf_mean(5,:), 'm--');
plot(0:recover_epochs, s_perf_best(1,:), 'r');
plot(0:recover_epochs, s_perf_best(2,:), 'g');
plot(0:recover_epochs, s_perf_best(3,:), 'b');
plot(0:recover_epochs, s_perf_best(4,:), 'k');
plot(0:recover_epochs, s_perf_best(5,:), 'm');
legend([strread(num2str(perf_test),'%s');strread(num2str(perf_test),'%s')]);
xlabel('recovery iteration');
ylabel('percentage of misclassified digits');
axis([0 recover_epochs 0 1]);

disp('section three complete')
disp(toc)

%% Genetic search for best cut based on instant performance
tic
disturbance = 0.3;
genome_size = int32(disturbance * num_weights);
population_size = 30;
max_generations = 20;
fitness_recovery_epoch = 10;
mutation_rate = .02;
survival_rate = 0.60;
crossover_rate = 0.75;

genome_fitness_instant = @(genome) perf_func(multiply_disturbance(net, genome, 0));
genome_fitness_reverse = @(genome) 1 - perf_func(multiply_disturbance(net, genome, 0));
genome_fitness_recover = @(genome) perf_func(train_persistent_disturbance(net,...
    xTrain, tTrain, fitness_recovery_epoch, ...
    @(this_net) multiply_disturbance(this_net, genome, 0), perf_func));
use_fitness = genome_fitness_reverse;

population = initiate_population(num_weights, genome_size, population_size);
gen_fitness = zeros(max_generations, population_size);
gen_population = cell(max_generations, population_size);
unique_population = zeros(max_generations, 1);
percent_cuts_population = zeros(max_generations, 3);

for g = 1:max_generations
    fitness = pop_fitness(population, use_fitness); 
    gen_fitness(g, :) = fitness;
    gen_population(g, :) = mat2cell(population, ones(population_size,1), genome_size)';
    unique_population(g) = length(unique(population));
    [c1, c2, c3] = percentage_layer_cuts(population');
    percent_cuts_population(g, :) = [mean(c1) mean(c2) mean(c3)];
    
    [population, fitness] = selection(population, fitness, survival_rate);
    survived_num = size(population, 1);
    
    while size(population, 1) < population_size
        if size(population, 1) == 1 || rand(1) > crossover_rate;
            genome_index = randperm(survived_num, 1);
            new_genome = population(genome_index, :);
        else
            parents_idx = randperm(survived_num, 2);
            new_genome = crossover(population(parents_idx(1),:), population(parents_idx(2),:),...
                fitness(parents_idx(1)), fitness(parents_idx(2)));
        end
        new_genome = mutate(new_genome, mutation_rate, num_weights);
        population(size(population, 1) + 1, :) = new_genome;
    end
    disp('section four progress');
    disp(g / max_generations);
end

mean_gen_fitness = mean(gen_fitness');
best_gen_fitness = min(gen_fitness');

figure(4)
hold on
plot(1:max_generations, mean_gen_fitness, 'k--')
plot(1:max_generations, best_gen_fitness, 'k')
axis([1 max_generations 0 0.5])
title('Population fitness over generations')
xlabel('Generation')
ylabel('Error rate')
legend('mean genome error','best genome error')

figure(5)
hold on
plot(1:max_generations, percent_cuts_population)
legend('layer 1','layer 2','layer 3')
title('concentration of disturbances by layer')
xlabel('generation')
ylabel('% of layer cut')

figure(6)
hold on
plot(1:max_generations, unique_population)
axis([1 max_generations 0 num_weights])
plot([1 max_generations], [genome_size genome_size], 'r--')
title('Number of total unique weights by generation')
xlabel('Generation')
ylabel('# of unique weight cuts over the population')
legend('population uniqueness','genome size, minimum uniqueness')

disp('section four complete');
disp(toc)

%% Genetic result comparing fully random weight cuts with layer 1 only
population_l1 = initiate_population(784*100, genome_size, population_size);

gen_fitness_l1 = zeros(max_generations, population_size);


for g = 1:max_generations
    fitness = pop_fitness(population_l1, genome_fitness_instant);
    gen_fitness_l1(g, :) = fitness;
    
    [population_l1, fitness] = selection(population_l1, fitness, survival_rate);
    survived_num = size(population_l1, 1);
    
    while size(population_l1, 1) < population_size
        if size(population_l1, 1) == 1 || rand(1) > crossover_rate;
            genome_index = randperm(survived_num, 1);
            new_genome = population_l1(genome_index, :);
        else
            parents_idx = randperm(survived_num, 2);
            new_genome = crossover(population_l1(parents_idx(1),:), population_l1(parents_idx(2),:),...
                fitness(parents_idx(1)), fitness(parents_idx(2)));
        end
        new_genome = mutate(new_genome, mutation_rate, num_weights);
        population_l1(size(population_l1, 1) + 1, :) = new_genome;
    end
    disp('section four progress');
    disp(g / max_generations);
end
 
figure(5)
hold on
y_gen = zeros(max_generations, 1);
y_gen_l1 = zeros(max_generations, 1);

for i=1:max_generations
    y_gen(i, 1) = mean(gen_fitness(i, :));
    y_gen_l1(i, 1) = mean(gen_fitness_l1(i, :));
end

plot(y_gen);
plot(y_gen_l1);
title('Fitness of fully random cuts vs. first layer only')
xlabel('Iteration #')
ylabel('Error')
legend('Fully Random', 'First Layer Only');

%% Compare GA training time with 30% cuts to initial training time


% NOTE THIS DOESN'T SEEM TO WORK YET, not sure why
net_copy = untrained_enc_enc_soft_net(xTrain(:,1), tTrain(:,1));
net_copy.trainParam.epochs = 1;
net_copy.trainParam.showWindow = false;

[ga_net, ga_perf] = train_persistent_disturbance(net_copy,...
    xTrain, tTrain, 300, ...
    @(this_net) multiply_disturbance(this_net, population(1, :), 0), perf_func);

net_perf = zeros(size(ga_perf));
for i=1:length(ga_perf)
    net_perf(i) = perf_func(net_copy);
    net_copy = train(net_copy,xTrain,tTrain);
end

figure(6)
hold on
plot(net_perf)
plot(ga_perf)
title('Training progess of original net vs. GA cut net')
xlabel('Training Iteration')
ylabel('% Misclassified Digits')
legend('Original Network', 'GA Cut Network');
