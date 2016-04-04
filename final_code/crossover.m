function [child_genome] = crossover(parent_1, parent_2, fit_1, fit_2)
    child_genome = intersect(parent_1, parent_2);
    possible_1 = setdiff(parent_1, child_genome);
    possible_2 = setdiff(parent_2, child_genome);
    while length(child_genome) < length(parent_1)
        %since fitness to be minized, give high chance to low parent
        if rand(1) < (fit_2 / (fit_1 + fit_2))
            child_genome = [child_genome possible_1(1)];
            possible_1 = possible_1(possible_1 ~= possible_1(1));
        else
            child_genome = [child_genome possible_2(1)];
            possible_2 = possible_2(possible_2 ~= possible_2(1));
        end
    end
end