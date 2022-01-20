import csv
from typing import Any, Dict, List
from random import randint, seed, shuffle
from collections import deque
from unittest import result


'''
    TODO:
    - Create populations
    - Generate new generations
'''


def read_csv(filename: str) -> Dict[str, List[float]]:
    affinity = dict()
    with open(filename, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=';', quotechar='|')
        for row in list(reader)[1:]:
            affinity[row[0]] = row[1:]
    return affinity


def OX1(parent1: List[Any], parent2: List[Any]) -> List[Any]:
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
        if index > len(parent1) -1:
            index = 0
        if parent_index > len(parent1) -1:
            parent_index = 0

        if parent2[parent_index] not in genes:
            child1[index] = parent2[parent_index]
        else:
            parent_index += 1
            continue
        index += 1
        parent_index += 1
        counter -= 1
    return child1


def get_offsprings(parent1: List[Any], parent2: List[Any]) -> List[List[Any]]:
    offsprings = []
    offsprings.append(OX1(parent1, parent2))
    offsprings.append(OX1(parent2, parent1))
    offsprings.append(OX1(parent1, parent2))
    offsprings.append(OX1(parent2, parent1))
    return offsprings


def mutate(offspring: List[Any]) -> List[Any]:
    index1 = randint(0, len(offspring)-1)
    index2 = randint(0, len(offspring)-1)

    tmp = offspring[index1]
    offspring[index1] = offspring[index2]
    offspring[index2] = tmp
    return offspring


def fitness(table: List[str], dictionary: Dict[str, List[float]]) -> float:
    indexes = dictionary[' ']
    result = 0.0
    
    for knight_index in range(len(table)):
        affinity_list = dictionary[table[knight_index]]
        neighbour_index = knight_index+1
        if neighbour_index >= len(table):
            neighbour_index = 0
        
        neighbour = table[neighbour_index]
        result += float(affinity_list[indexes.index(neighbour)])
        


    return result


def generate_population(dictionary: Dict[str, List[str]], old_population: List[List[Any]]=None, amount: int=0) -> List[List[Any]]:
    population = []

    # If no older population exists, create a new one
    if not old_population:
        names = dictionary[' ']
        for i in range(amount):
            tmp = names.copy()
            shuffle(tmp)
            population.append(tmp)
        return population
    
    # Create the next generation
    tmp = []
    for index in range(len(old_population)):
        tmp.append([old_population[index], fitness(old_population[index], dictionary)])
    tmp.sort(reverse=True, key=lambda x: x[1])
    
    old_pop_sorted = list(map(lambda x: x[0], tmp))
    old_pop_sorted = old_pop_sorted[:(len(old_pop_sorted) // 2)]
    length = len(old_pop_sorted)

    for i in range(length // 2):
        # Select random pair, generate offsprings and remove them from the sorted list
        parent1 = None
        parent2 = None
        index = None
        while parent1 is None:
            index = randint(0, len(old_pop_sorted)-1)
            parent1 = old_pop_sorted[index]
        old_pop_sorted[index] = None

        while parent2 is None:
            index = randint(0, len(old_pop_sorted)-1)
            parent2 = old_pop_sorted[index]
        old_pop_sorted[index] = None
        
        population += get_offsprings(parent1, parent2)

    return population



def main():
    affinity = read_csv("./Evolutionary Algorithms/RondeTafel.csv")
    # p1 = ["Arthur", "Lancelot", "Gawain", "Geraint", "Percival", "Bors the Younger", "Lamorak", "Kay Sir Gareth", "Bedivere", "Gaheris", "Galahad", "Tristan"]
    # p2 = ["Bors the Younger", "Tristan", "Arthur", "Bedivere", "Gaheris", "Galahad","Lancelot", "Percival", "Gawain", "Geraint",  "Lamorak", "Kay Sir Gareth"]

    # offsprings = get_offsprings(p1, p2)
    # c1 = mutate(offsprings[0])
    # c2 = mutate(offsprings[1])

    # print("Fitness C1:", fitness(c1, affinity))
    # print("Fitness C2:", fitness(c2, affinity))

    n_generations = 100
    counter = 0
    current_gen = None
    while(counter <= n_generations):
        if not current_gen:
            current_gen = generate_population(affinity, amount=20)
        else:
            current_gen = generate_population(affinity, current_gen)

        print("Generation", counter)
        results = [fitness(current_gen[i], affinity) for i in range(len(current_gen))]
        results.sort(reverse=True)
        print("Top 3:")
        print("\t", results[0])
        print("\t", results[1])
        print("\t", results[2])
        print()
        counter += 1




if __name__ == "__main__":
    seed(0)
    main()