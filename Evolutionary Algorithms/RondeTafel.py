import csv
from typing import Any, Dict, List
from random import randint, seed, shuffle
from collections import deque
import matplotlib.pyplot as plt


'''
    Deze functie leest het hele csv-file uit, op de eerste regel na.
    De output is een Dictionary met als key de eerste waarde van de rij
    en als value een List van de overige waarden van de rij.
    In het geval van de opdracht, is de naam van de ridder de key, en de
    de lijst van affiniteiten de value.
'''
def read_csv(filename: str) -> Dict[str, List[str]]:
    affinity = dict()
    with open(filename, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=';', quotechar='|')
        for row in list(reader)[1:]:
            affinity[row[0]] = row[1:]
    return affinity


'''
    Deze functie genereert een nakomeling van de opgegeven ouders. Dit wordt gedaan door
    eerst op een willekeurige index van de eerste ouder een aantal genen door te geven
    naar de nakomeling. Daarna worden de genen van de tweede ouder in de nakomeling
    gezet, echter alleen als deze NIET hetzelfde zijn als de gekozen genen van de eerste
    ouder.
'''
def OX1(parent1: List[Any], parent2: List[Any]) -> List[Any]:
    offspring = [0 for _ in range(len(parent1))]

    # - Choose a number of random genes
    start = randint(0, len(parent1)-1)
    tmp = parent1[start:len(parent1)] + parent1[:start]
    genes = tmp[0:(len(parent1)// 2)]

    # - Place the random genes
    offspring[:len(genes)] = genes
    offspring = deque(offspring)
    offspring.rotate(start)
    offspring = list(offspring)

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
            offspring[index] = parent2[parent_index]
        else:
            parent_index += 1
            continue
        index += 1
        parent_index += 1
        counter -= 1
    return offspring


'''
    Deze functie creëert een lijst van twee nakomelingen van de twee megegeven ouders.
    Hier wordt dus twee keer de OX1 crossover functie aangeroepen, waarna de nakomeling
    gemuteerd wordt en wordt toegevoegd aan de lijst.
    In de tweede aanroep worden de ouders omgedraaid, zodat de OX1-functie genen kiest
    van de andere ouder.
'''
def get_offsprings(parent1: List[Any], parent2: List[Any]) -> List[List[Any]]:
    offsprings = []
    offsprings.append(mutate(OX1(parent1, parent2)))
    offsprings.append(mutate(OX1(parent2, parent1)))
    return offsprings


'''
    Deze functie muteert een gegeven individu door eerst twee willekeurige indexes te
    kiezen. Daarna worden hiermee twee waarden met elkaar verwisseld en tot slot wordt
    dit nieuwe individu gereturnt.
'''
def mutate(offspring: List[Any]) -> List[Any]:
    index1 = randint(0, len(offspring)-1)
    index2 = randint(0, len(offspring)-1)

    tmp = offspring[index1]
    offspring[index1] = offspring[index2]
    offspring[index2] = tmp
    return offspring


'''
    Deze functie berekent de fitness van een individu. De fitness wordt berekend door 
    voor elke ridder zijn affiniteit met zijn linker- en rechterbuurman bij elkaar op
    te tellen. Dit wordt bij het resultaat opgeteld en uiteindelijk gereturnt.
'''
def fitness(individual: List[str], dictionary: Dict[str, List[str]]) -> float:
    indexes = dictionary[' ']
    result = 0.0
    
    for knight_index in range(len(individual)):
        affinity_list = dictionary[individual[knight_index]]
        neighbour_index = knight_index+1
        if neighbour_index >= len(individual):
            neighbour_index = 0
        individual
        neighbour = individual[neighbour_index]
        result += float(affinity_list[indexes.index(neighbour)])

        neighbour_index = knight_index-1
        if neighbour_index < 0:
            neighbour_index = len(individual)-1
        
        neighbour = individual[neighbour_index]
        result += float(affinity_list[indexes.index(neighbour)])
    return result


'''
    Deze functie voert een toernooi uit met k aantal ridders van de meegegeven populatie
    om op die manier de beste ridder van die populatie te bepalen.
'''
def tournament_selection(population: List[List[Any]], k: int, dictionary: Dict[str, List[str]]) -> List[Any]:
    best = None
    for i in range(k):
        individual = population[randint(0, len(population)-1)]
        if (best == None) or fitness(individual, dictionary) > fitness(best, dictionary):
            best = individual
    return best


'''
    Deze functie creëert een generatie van ridder tafelvolgordes.
    Als er nog geen oudere generatie bestaat, wordt er een nieuwe generatie gemaakt.
    Als er wel een oude generatie is, wordt vanuit de oude generatie de volgende generatie gemaakt.
    Dit gebeurt door eerst de beste 10% van de oude generatie te muteren om daarna mee te
    nemen naar de volgende generatie. Vervolgens wordt tournament selectie toegepast om
    de rest van de nieuwe generatie te vullen, totdat de nieuwe generatie net zo groot is
    als de oude.   
'''
def generate_population(dictionary: Dict[str, List[str]], old_population: List[List[Any]]=None, population_size: int=0) -> List[List[Any]]:
    # If no older population exists, create a new one
    if not old_population:
        population = []
        names = dictionary[' ']
        for i in range(population_size):
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

    # Choose 10% of the best 
    offsprings = [] + [mutate(elite) for elite in old_pop_sorted[:int(len(old_pop_sorted) * 0.1)]]

    while len(offsprings) < len(old_population):
        parent1 = tournament_selection(old_pop_sorted, 2, dictionary)
        parent2 = tournament_selection(old_pop_sorted, 2, dictionary)

        offsprings += get_offsprings(parent1, parent2)
    return offsprings


'''
    Deze functie plot een grafiek waarin de Top Drie van elke epoch wordt weergegeven.
    Ook wordt de meest optimale individu weergegeven. Op de x-as wordt de epoch
    weergegeven en op de y-as wordt de fitness.
'''
def plot(fitness_values: List[List[float]], n_generation: int):
    epochs = [i for i in range(n_generation+1)]

    fig, ax = plt.subplots()
    ax.plot(epochs, fitness_values[0], label="Nr1.")
    ax.plot(epochs, fitness_values[1], label="Nr2.")
    ax.plot(epochs, fitness_values[2], label="Nr3.")
    ax.plot(epochs, fitness_values[3], label="Best.")
    ax.legend()
    plt.xlabel("Epoch")
    plt.ylabel("Fitness Value")
    plt.show()


'''
    De main functie. Deze creëert eerst de affiniteiten dictionary en wordt de
    populatiegrootte en het aantal epochs gedefinieerd. In de while loop wordt de huidige
    populatie gegenereerd en worden de fitness van de Top Drie en de meest optimale
    individu in hun eigen Lists opgeslagen. Deze Lists worden later gebruikt om de grafiek
    te plotten. Tot slot worden de gewichtsproducten van de meest optimale individu
    aangemaakt en worden deze geprint.
'''
def main():
    affinity = read_csv("./Evolutionary Algorithms/RondeTafel.csv")
    epochs = 500
    population_size = 100
    counter = 0
    current_gen = None
    results = None
    best_table = None

    nr1_values = []
    nr2_values = []
    nr3_values = []
    best_values = []

    while(counter <= epochs):
        if not current_gen:
            current_gen = generate_population(affinity, population_size=population_size)
        else:
            current_gen = generate_population(affinity, current_gen)

        print("Generation", counter)
        results = [[i, fitness(current_gen[i], affinity)] for i in range(len(current_gen))]
        results.sort(reverse=True, key=lambda x: x[1])

        if best_table is None or (results[0][1] > best_table[0]):
            best_table = (results[0][1], current_gen[results[0][0]])

        nr1_values.append(results[0][1])
        nr2_values.append(results[1][1])
        nr3_values.append(results[2][1])
        best_values.append(best_table[0])

        print("Top 3:")
        print("\t", nr1_values[-1])
        print("\t", nr2_values[-1])
        print("\t", nr3_values[-1])
        print("Best solution:")
        print("\t", best_table[0])
        print("\t", best_table[1])
        print()

        counter += 1
    
    plot([nr1_values, nr2_values, nr3_values, best_values], epochs)

    weights = ""
    for knight_index in range(len(best_table[1])):
        indexes = affinity[' ']
        knight_name = best_table[1][knight_index]
        neighbour_index = knight_index+1

        if neighbour_index >= len(best_table[1]):
            neighbour_index = 0
        neighbour_name = best_table[1][neighbour_index]

        affinity_list_knight = affinity[knight_name]
        affinity_list_neighbour = affinity[neighbour_name]

        affinity_knight = affinity_list_knight[indexes.index(neighbour_name)]
        affinity_neighbour = affinity_list_neighbour[indexes.index(knight_name)]

        weights += knight_name + " (" + str(affinity_knight) + " x " + str(affinity_neighbour) +  ") "
    print("Best Solution Weight Product:")
    print(weights)


if __name__ == "__main__":
    seed(0)
    main()