#!/usr/bin/python
from __future__ import annotations
import math
import numpy as np
from random import uniform, seed, sample
from typing import List, Any, Optional


def sigmoid(z: float):
    return (1.0 / (1.0 + math.exp(-z)))


def der_sigmoid(z: float):
    return sigmoid(z) * (1.0-sigmoid(z))


class SigmoidNeuron():
    def __init__(self, id: int, prev_layer: Optional[List[SigmoidNeuron]]):
        self.id = id
        
        self.prev_layer = prev_layer
        self.next_layer = None

        self.weights = [uniform(-1.0, 1.0) for _ in range(len(prev_layer))] if self.prev_layer is not None else None
        self.bias = uniform(-1.0, 1.0)

        self.a = 0.0
        self.z = 0.0
        self.delta = 0.0


    def add_next_layer(self, next_layer: List[SigmoidNeuron]):
        self.next_layer = next_layer
    

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
        

    def process_input(self):
        tmp = 0.0
        for neuron in self.prev_layer:
            tmp += (neuron.a * self.weights[neuron.id])
        self.z = tmp + self.bias
        self.a = sigmoid(self.z)


def train_network(l_factor: float, data_inputs: List[List[Any]], desired_outputs: List[List[Any]], input_neuron_vector: List[SigmoidNeuron], output_neuron_vector: List[SigmoidNeuron]):
    for iteration in range(5000):
        if iteration % 100 == 0:
            print("Iteration:", iteration+1)
        for data in range(len(data_inputs)):
            # input_vector 
            for input_n in range(len(input_neuron_vector)):
                input_neuron_vector[input_n].a = data_inputs[data][input_n]

            for output_n in range(len(output_neuron_vector)):
                output_neuron_vector[output_n].process_input()
                output_neuron_vector[output_n].update_weights(l_factor, desired_outputs[data])         
            
            if iteration % 100 == 0:
                print([_input.a for _input in input_neuron_vector], desired_outputs[data], "\t", [_output.a for _output in output_neuron_vector])


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

    train_network(l_factor, inputs, desired_output, [p0,p1,p2], [pNOR])


def exercise_D_Iris():
    training_set = np.genfromtxt("./nn/iris.data", delimiter=',', usecols=[0,1,2,3])
    output_names = list(np.genfromtxt("./nn/iris.data", delimiter=',', usecols=[4], dtype=str))
    for name in range(len(output_names)):
        if output_names[name] == "Iris-setosa":
            output_names[name] = [1, 0, 0]
        elif output_names[name] == "Iris-versicolor":
            output_names[name] = [0, 1, 0]
        elif output_names[name] == "Iris-virginica":
            output_names[name] = [0, 0, 1]
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
    train_network(l_factor, list(training_set), list(output_names), input_vector, output_vector)

    # ...and test the network with the test dataset
    # Pick random flower samples from dataset
    sample_nr = 25
    test_indeces = sample(range(0, len(training_set)-1), sample_nr)
    test_samples = [training_set[_sample] for _sample in test_indeces]
    expected_output = [output_names[_sample] for _sample in test_indeces]

    print("\nDone Training!\nTesting...\n")

    results = test_network(test_samples, expected_output, input_vector, output_vector)
    flowers = ["Iris-setosa", "Iris-versicolor", "Iris-virginica"]

    print("\n...Done!\n")

    # results = [test_output, index_of_highest_out_value, index_of_expected_out]
    error_counter = 0
    for _out in results:
        print("Expected Flower: {:<15}\t Output Flower: {:<15}\t Output Value: {}".format(flowers[_out[2]], flowers[_out[1]],_out[0][_out[1]]))
        if flowers[_out[2]] != flowers[_out[1]]:
            error_counter += 1

    # Calculate wrong percentage
    print("\nError Percentage: {:%}". format(error_counter / sample_nr))
    


def main():
    exercise_D_Iris()
    return


if __name__ == "__main__":
    seed(0)
    main()
