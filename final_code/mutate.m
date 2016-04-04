function [genome] = mutate(genome, mutation_rate, genome_range)
    genome_length = length(genome);
    mutation_site_count = int32(mutation_rate * genome_length);
    possible_mutations = 1:genome_range;
    possible_mutations(genome) = [];
    mutations = possible_mutations(randperm(length(possible_mutations), mutation_site_count));
    mutation_sites = randperm(genome_length, mutation_site_count);
    genome(mutation_sites) = mutations;
end