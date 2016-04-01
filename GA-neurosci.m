% Genome evolution:
close all
clear all

%constant definitons
edgesNum = 84060; %not sure about total number of gene values?
cutPercent = 0.01;
mutPercent = 0.01;
survivalRate = 0.1;
crossRate = 0.75;
maxIter = 100;

%mutation constants
weightMut = 0.99;
connectionMut = 0.25;
linkMut = 1.5;
biasMut = 0.4;
nodeMut = 0.5;

w_Step = 0.1;
maxNodes = 100000;
output = 4;
popSize = 20;
maxGen = 100;
maxFitness = 110; %need to decide from mayank matlab fn

%initializing population = genomes + fitnes
%need to define genes (size of weights matrix) and randomize
%
genomes = weights; %from mayank fn?
genes = genomes;
selectGenes = zeros(size(genomes));
evolvedFitness = zeros(size(genomes));
evolvedGenes = zeros(size(genomes));
fitIndex = zeros(size(genomes));

[xTrainImages, tTrain] = digittrain_dataset;
[xTestImages, tTest] = digittest_dataset;
deepnet = train_deepnet(xTrainImages, tTrain, 50);
%computing fitness for each set of genomes in 
for i = size(genomes)
    genFitness = cut_fitness(deepnet, xTrainImages, tTrain, xTestImages, tTest, genomes(i), 0, 50)
	evolvedFitness(i) = genFitness(i);
end

%ranking genomes
for fitVal = 1:size(genomes)
	if fitness(fitVal) > fitness(fitVal+1)
		%selecting best fitness scores
		fitIndex(fitVal) = evolvedFitness(fitVal+1); %using temporary copy
		fitIndex(fitval+1) = evolvedFitness(fitVal);
		%selecting best genomes corresponding to best fitness scores
		selectGenes(fitVal) = genomes(fitVal+1);
		selectGenes(fitVal+1) = genomes(fitVal);
	end
end

%gene fitness cut off
cutoff = round(cutPercent*genomes);
%define cutoff function AKA natural selection function
function [childFitness, childGenome] = cutoffFn(genome, cutpercent, index)
	for j = 1:cutoff
		evolvedGenes(j) = selectGenes(j);
		evolvedFitness(j) = fitIndex(j);
	end
end 

%random permutation of weight/gene dropoff
randW = [zeros(1, edgesNum), ones(1, edgesNum)]
randW = randW(randperm(2*N))

function [childGenome] = mutate(genome)
	for k = 1:size(genome)

	end

end

function [childGenome] = crossover(genomes, gIndex1, gIndex2)
	genome1 = genomes[gIndex1]
	genome2 = genomes[gIndex2]
	genomeMembers = len(genome1)

	genesLst = []
	childGenome = [genesLst] %members of genes after selection
	childFitness = [] %empty list of fitnesses
	genomesCopy = [genome1, genome2]

	while length(geneLst) <  genomeMembers

	end

end

population = cuttofFn(population)
survivalSelec = length(population) %post selection number of individuals

%plotting fitness + genomes.
figure
hold on;
plot
