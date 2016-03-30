import random
# [-1.5, 1.3, 8], [-1.8, 1.3, 8]

# [0, 14, 25], [9, 10, 14, 3]
# []

# Population1 [[0, 1, 2] [0, 8, 5] [0, 1, 2] [0, 8, 5]]
# Population2 [[0, 1, 2] [0, 8, 5] [0, 1, 2] [0, 8, 5], ]
g1 = [0, 14, 25, 10]
g2 = [9, 10, 14, 3]
desired_length = len(g1)

child_genome = []

g1_copy, g2_copy = g1[:], g2[:]
intersect = set(g1).intersection(g2)

for val in intersect:
    child_genome.append(val)

while len(child_genome) < desired_length:
    rand_val = random.random()
    if rand_val < 0.7 and len(g1_copy) > 0:
        g1_val = g1_copy.pop()
        while g1_val in intersect and len(g1_copy) > 0:
            g1_val = g1_copy.pop()
        if g1_val not in intersect:
            child_genome.append(g1_val)
    else:
        g2_val = g2_copy.pop()
        while g2_val in intersect and len(g2_copy) > 0:
            g2_val = g2_copy.pop()
        if g2_val not in intersect:
            child_genome.append(g2_val)