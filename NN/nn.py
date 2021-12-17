#!/usr/bin/python
from __future__ import annotations
import math
import numpy as np
from random import uniform, seed, sample
from typing import List, Any, Optional


'''
    Deze functie berekent de sigmoïde van de parameter 'z'.
'''
def sigmoid(z: float):
    return (1.0 / (1.0 + math.exp(-z)))


'''
    Deze functie berekent de afgeleide sigmoïde van de parameter 'z'.
'''
def der_sigmoid(z: float):
    return sigmoid(z) * (1.0-sigmoid(z))


class SigmoidNeuron():
    '''

    '''
    def __init__(self, id: int, prev_layer: Optional[List[SigmoidNeuron]]):
        self.id = id
        
        self.prev_layer = prev_layer
        self.next_layer = None

        self.weights = [uniform(-1.0, 1.0) for _ in range(len(prev_layer))] if self.prev_layer is not None else None
        self.bias = uniform(-1.0, 1.0)

        self.a = 0.0
        self.z = 0.0
        self.delta = 0.0


    '''
        Deze functie 'verbindt' het huidige neuron met de volgende laag 'next_layer'. 
    '''
    def add_next_layer(self, next_layer: List[SigmoidNeuron]):
        self.next_layer = next_layer
    

    '''
        Deze functie updatet de weights van het neuron. Voor het berekenen, moet de delta
        geüpdatet worden. De berekening van de delta is afhankelijk van in welke laag het
        neuron zit. Als het in de laatste laag zit, dan wordt de delta berekent door de
        afgeleide sigmoïde te nemen van 'z' en dat te vermenigvuldigen met het verschil
        van de gewenste output en de 'a'.
        Als het neuron in een tussenlaag zit, dan wordt eerst de totale som van de vorige
        laag zijn delta te vermenigvuldigen met de weigths berekent. Deze som wordt
        vervolgens vermenigvuldigt met de afgeleide sigmoïde van 'z' waar de delta
        uitkomt.
        Als de delta bekent is, wordt iedere weight geüpdatet. Eerst wordt de learning_factor
        met de delta en de vorige laag zijn 'a' te vermenigvuld. Dan wordt deze uitkomt
        bij de weight opgeteld.
        Tot lot wortd de bias geüpdatet door de learning_factor te vermenigvuldigen met de
        delta en vervolgens dit bij de bias op te tellen.
    '''
    def update_weights(self, learning_factor: float, desired_output: List[float]):
        # Update weights
        tmp = 0.0

        if self.prev_layer is not None:
            # If current layer is the last one...
            if self.next_layer is None:
                self.delta = der_sigmoid(self.z) * (desired_output[self.id] - self.a)
            
            else:
                for neuron in self.next_layer:
                    tmp += (neuron.delta * neuron.weights[self.id])
                self.delta = der_sigmoid(self.z) * tmp

            for node in range(len(self.weights)):
                self.weights[node] = self.weights[node] + learning_factor * self.delta * self.prev_layer[node].a
            
        # Update bias
        self.bias += learning_factor * self.delta


    '''
        Deze functie verwerkt de inputs van een neuron naar de z en a van die neuron.
    '''
    def process_input(self):
        tmp = 0.0
        for neuron in self.prev_layer:
            tmp += (neuron.a * self.weights[neuron.id])
        self.z = tmp + self.bias
        self.a = sigmoid(self.z)


'''
    Deze functie train het neural netwerk. Dit doet de functie door een mee gegeven 
    aantal iteraties te draaien waarbij de inputs voor iedere node verwerkt worden 
    en waarna vervolgens voor iedere output node de weights geupdated worden.
'''
def train_network(iterations: int, l_factor: float, data_inputs: List[List[Any]], desired_outputs: List[List[Any]], input_neuron_vector: List[SigmoidNeuron], output_neuron_vector: List[SigmoidNeuron]):
    for iteration in range(iterations):
        if iteration % 100 == 0:
            print("Iteration:", iteration+1)
        for data in range(len(data_inputs)):
            # input_vector 
            for input_n in range(len(input_neuron_vector)):
                input_neuron_vector[input_n].a = data_inputs[data][input_n]

            for output_n in range(len(output_neuron_vector)):
                output_neuron_vector[output_n].process_input()
                output_neuron_vector[output_n].update_weights(l_factor, desired_outputs[data])         
            
    #Debug print
            if iteration % 100 == 0:
                print([_input.a for _input in input_neuron_vector], desired_outputs[data], "\t", [_output.a for _output in output_neuron_vector])

'''
    Deze functie test het getrainde neural netwerk. Dit doet de functie door de 
    inputs te verwerken voor alle nodes en vervolgens de outputs terug te geven.
'''
def test_network(data_inputs: List[List[Any]], expected_outputs: List[List[Any]], input_neuron_vector: List[SigmoidNeuron], output_neuron_vector: List[SigmoidNeuron]) -> List[List[float], int, int]:
    results = []
    for data in range(len(data_inputs)):
        # input_vector 
        for input_n in range(len(input_neuron_vector)):
            input_neuron_vector[input_n].a = data_inputs[data][input_n]

        for output_n in range(len(output_neuron_vector)):
            output_neuron_vector[output_n].process_input()
        
        print([_input.a for _input in input_neuron_vector], expected_outputs[data], "\t", [_output.a for _output in output_neuron_vector])

        test_output = [_output.a for _output in output_neuron_vector]
        highest_out = np.argmax(test_output)
        expected_out = np.argmax(expected_outputs[data])
        results.append([test_output, highest_out, expected_out])
    return results


def exercise_NOR():
    p0 = SigmoidNeuron(0, None)
    p1 = SigmoidNeuron(1, None)
    p2 = SigmoidNeuron(2, None)

    # NOR:  0 0 0 = 1
    #       x x 1 = 0
    #       x 1 x = 0
    #       1 x x = 0

    pNOR = SigmoidNeuron(0, [p0, p1, p2])

    p0.add_next_layer([pNOR])
    p1.add_next_layer([pNOR])
    p2.add_next_layer([pNOR])


    inputs = [
        (0,0,0),
        (0,0,1),
        (0,1,0),
        (0,1,1),
        (1,0,0),
        (1,0,1),
        (1,1,0),
        (1,1,1),
    ]
    desired_output = [[1],[0],[0],[0],[0],[0],[0],[0]]
    
    l_factor = 0.1

    train_network(5000, l_factor, inputs, desired_output, [p0,p1,p2], [pNOR])


'''
    Deze functie voert de bloemenopdracht uit. Eerst wordt de trainingsset ingeladen
    door de eerste vier kolommen in te lezen. De laatste vijfde kolom wordt apart
    in een list in geladen. Omdat het neurale netwerk met getallen werkt krijgt elke
    bloem een nummer:
        1.  Iris Setosa
        2.  Iris Versicolour
        3.  Iris Virginica
    Daarom worded de namen in de list vervangen met de juiste getallen. Hierna worden
    de inputneuronen gedefinieerd en wordt de inputvector aangemaakt. Daarna worden 
    de outputneuronen gedefineerd en wordt de outputvector aangemaakt. Vervolgen wordt
    de inputvector met de outputvector verbonden en is het neurale netwerk klaar.
    Na de initialisatie begint het trainen van het netwerk. Er wordt een learning factor
    van 0.1 gekozen en het aantal iteraties wordt op 5000 gezet.
    Nadat het netwerk getraint is, wordt het getest door 25 willekeurige bloemen uit de
    dataset te kiezen en deze als input door het netwerk te laten verwerken.
    De uitkomst van de test wordt geprint, waarna het foutpercentage wordt geprint.

'''
def exercise_D_Iris():
    training_set = np.genfromtxt("./nn/trainingset.csv", delimiter=',', usecols=[0,1,2,3])
    training_names = list(np.genfromtxt("./nn/trainingset.csv", delimiter=',', usecols=[4], dtype=str))
    
    for name in range(len(training_names)):
        if training_names[name] == "Iris-setosa":
            training_names[name] = [1, 0, 0]
        elif training_names[name] == "Iris-versicolor":
            training_names[name] = [0, 1, 0]
        elif training_names[name] == "Iris-virginica":
            training_names[name] = [0, 0, 1]
    
    test_set = np.genfromtxt("./nn/testset.csv", delimiter=',', usecols=[0,1,2,3])
    test_names = list(np.genfromtxt("./nn/testset.csv", delimiter=',', usecols=[4], dtype=str))
    
    for name in range(len(test_names)):
        if test_names[name] == "Iris-setosa":
            test_names[name] = [1, 0, 0]
        elif test_names[name] == "Iris-versicolor":
            test_names[name] = [0, 1, 0]
        elif test_names[name] == "Iris-virginica":
            test_names[name] = [0, 0, 1]
    
    '''
        4 Inputs:
            - sepal length
            - sepal width
            - petal length
            - petal width

        3 Outputs:
            - Iris Setosa
            - Iris Versicolour
            - Iris Virginica
    
    '''

    s_length = SigmoidNeuron(0, None)
    s_width = SigmoidNeuron(1, None)
    p_lenght = SigmoidNeuron(2, None)
    p_width = SigmoidNeuron(3, None)

    input_vector = [s_length, s_width, p_lenght, p_width]

    setosa = SigmoidNeuron(0, input_vector)
    versicolour = SigmoidNeuron(1, input_vector)
    virginica  =SigmoidNeuron(2, input_vector)

    output_vector = [setosa, versicolour, virginica]

    for _input in input_vector:
        _input.add_next_layer(output_vector)

    l_factor = 0.1

    # Train the network...
    train_network(5000, l_factor, list(training_set), list(training_names), input_vector, output_vector)
    print("\nDone Training!\nTesting...\n")

    # ...and test the network with the test dataset
    results = test_network(list(test_set), list(test_names), input_vector, output_vector)
    flowers = ["Iris-setosa", "Iris-versicolor", "Iris-virginica"]

    print("\n...Done!\n")

    # results = [test_output, index_of_highest_out_value, index_of_expected_out]
    error_counter = 0
    for _out in results:
        print("Expected Flower: {:<15}\t Output Flower: {:<15}\t Output Value: {}".format(flowers[_out[2]], flowers[_out[1]],_out[0][_out[1]]))
        if flowers[_out[2]] != flowers[_out[1]]:
            error_counter += 1

    # Calculate wrong percentage
    print("\nError Percentage: {:%}". format(error_counter / len(test_names)))
    


def main():
    exercise_D_Iris()
    return


if __name__ == "__main__":
    seed(0)
    main()
