function GA_neurosci
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

popSize = 20;
maxGen = 100;

%initializing population = genomes + fitnes
%need to define genes (size of weights matrix) and randomize
%
genomes = weights; %from mayank fn?
genes = genomes;
selectGenes = zeros(size(genomes));
evolvedFitness = zeros(size(genomes));
evolvedGenes = zeros(size(genomes));
fitIndex = zeros(size(genomes));
population = []
genesLst = []

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

%function calls down here
population = cutoffFn(population)
survivalSelec = length(population) %post selection number of individuals

%plotting fitness + genomes for examining diff permutations vs fitness
figure;
hold on;
plot(population, fitness)
end


%define cutoff function AKA natural selection function
function [childFitness, childGenome] = cutoffFn(genome, cutpercent, index)
	for j = 1:cutoff
		evolvedGenes(j) = selectGenes(j);
		evolvedFitness(j) = fitIndex(j);
	end
end 

function [childGenome] = mutate(genome)
	max = edgesNum;
	for k = 1:size(genome)
		randVal = rand(1); %1 value random generator
		if randVal < mutPercent
			newI = randVal*max;
			while ismember(randVal, genome)
				newI = rand(1)*max;
			end
			genome(k) = newI;
	end
end

function [childGenome] = crossover(genomes, gIndex1, gIndex2)
	genome1 = genomes[gIndex1];
	genome2 = genomes[gIndex2];
	genomeMembers = len(genome1);

	genesLst = [];
	childGenome = [genesLst]; %members of genes after selection
	childFitness = []; %empty list of fitnesses
	genomeCopy1 = genome1;
	genomeCopy2 = genome2;

	GIntersect = intersection(genomeCopy1, genomeCopy2);

	while length(geneLst) < genomeMembers
		randVal = rand(1); 
		if randVal < 0.7 && length(genomeCopy1) > 0
			gVal1 = genomeCopy1(1)

			while ismember(gVal1, GIntersect)
				gVal1 = genomeCopy1(1);
			end
			if isMember(gVal1,GIntersect) == 0
				genesLst = [genesLst gVal1];
			end

        else
            g2Val = genomeCopy2(1)
            while ismember(gVal2, GIntersect)
                    gVal2 = genomeCopy2(1)
                end
                if isMember(gVal2,GIntersect) == 0
                    genesLst = [genesLst gVal2]
                end
            end	
    	end

	%need to mutate genesLst before appending
	childGenome[0] = genesLst
	population = [population childGenome]
end


