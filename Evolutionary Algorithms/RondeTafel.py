import csv
from typing import Any, Tuple, List
from random import randint, seed
from collections import deque

seed(0)

'''
    TODO:
    - Create populations
    - Fitness functions
    - Mutatation
    - Generate new generations
'''


def csv_test():
    with open('Evolutionary Algorithms/RondeTafel.csv', newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=';', quotechar='|')
        for row in reader:
            print(', '.join(row))


def OX1(parent1: Tuple[Any], parent2: Tuple[Any]) -> Tuple[Any]:
    child1 = [0 for _ in range(len(parent1))]

    # - Choose a number of random genes
    start = randint(0, len(parent1)-1)
    tmp = parent1[start:len(parent1)] + parent1[:start]
    genes = tmp[0:(len(parent1)// 2)]

    # - Place the random genes
    child1[:len(genes)] = genes
    child1 = deque(child1)
    child1.rotate(start)
    child1 = list(child1)

    index = start - len(parent1) + (len(parent1) // 2)
    
    counter = len(genes)
    parent_index = index
    # - Fill the empty genes
    while(counter > 0):
        if parent2[parent_index] not in genes:
            child1[index] = parent2[parent_index]
        else:
            parent_index += 1
            continue
        index += 1
        parent_index += 1
        if index > len(parent1) -1:
            index = 0
        if parent_index > len(parent1) -1:
            parent_index = 0
        counter -= 1
    return tuple(child1)


def get_offsprings(parent1: Tuple[Any], parent2: Tuple[Any]) -> List[Tuple[Any]]:
    offsprings = []
    offsprings.append(OX1(parent1, parent2))
    offsprings.append(OX1(parent2, parent1))
    return offsprings