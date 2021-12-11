#!/usr/bin/python
import math
import numpy as np
from random import uniform, seed, sample
from typing import List, Any


def sigmoid(z):
    return (1.0 / (1.0 + math.exp(-z)))


def der_sigmoid(z):
    return sigmoid(z) * (1.0-sigmoid(z))



class SigmoidNeuron():
    def __init__(self, id: int, prev_layer: List):
        self.id = id
        
        self.prev_layer = prev_layer
        self.next_layer = None

        self.weights = [uniform(-1.0, 1.0) for _ in range(len(prev_layer))] if self.prev_layer is not None else None
        self.bias = uniform(-1.0, 1.0)

        self.a = 0.0
        self.z = 0.0
        self.delta = 0.0

    def add_next_layer(self, next_layer):
        self.next_layer = next_layer
    
    def connect_to(self, other_id):
        self.next_layer[other_id].inputs[self.id] = self
    

    def update_weights(self, learning_factor, desired_output: List[float]):
        # Update weights
        tmp = 0.0

        if self.prev_layer is not None:
            # If current layer is the last one...
            if self.next_layer is None:
                self.delta = der_sigmoid(self.z) * (desired_output[self.id] - self.a)
                # print(der_sigmoid(self.z), desired_output[self.id], self.a, self.delta)
            
            else:
                for i in self.next_layer:
                    tmp += (i.delta * i.weights[self.id])
                self.delta = der_sigmoid(self.z) * tmp
                #print("\t\t\t", self.id, self.delta)

            for node in range(len(self.weights)):
               # print("BEFORE:\t", self.id, self.weights[node], self.delta)
                self.weights[node] = self.weights[node] + learning_factor * self.delta * self.prev_layer[node].a
              #  print("AFTER:\t", self.id, self.weights[node], self.delta)
            
        # Update bias
        self.bias += learning_factor * self.delta
        


    def process_input(self, input: int=None):
        tmp = 0.0
        for i in self.prev_layer:
            tmp += (i.a * self.weights[i.id])     # i = tuple: (Neuron, weight)
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
                
        #d 
        # if x % 1000 == 0:
            # print("debug print line")


def test_network(data_inputs: List[List[Any]], expected_outputs: List[List[Any]], input_neuron_vector: List[SigmoidNeuron], output_neuron_vector: List[SigmoidNeuron]):
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
        


class Perceptron():
    def __init__(self, n_inputs, treshold, isInput=False):
        self.isInput = isInput
        if self.isInput:
            self.input = None
            self.inputs = []
        else:
            self.inputs = [None for _ in range(n_inputs)]
        self.output = 0
        self.treshold = treshold
        
    
    def connect_to(self, other, i_input, weight):
        other.inputs[i_input] = (self, weight)

    
    def process_input(self):
        if self.isInput:
            self.output = self.input
        else:
            tmp = 0
            for i in self.inputs:
                tmp += (i[0].output * i[1])     # i = tuple: (Perceptron, weight)
            
            if tmp >= self.treshold:
                self.output = 1
            else:
                self.output = 0


    def update(self, value):

        pass


class NeuralNetwork():
    def __init__(self, n_layers):
        self.layers = [[] for _ in range(n_layers)]

    def addToLayer(self, layer, neuron: SigmoidNeuron):
        self.layers[layer].append(neuron)

    def cost(self, input_samples):
        input_layer = self.layers[0]
        output_layer = self.layers[-1]
        n = len(input_layer)
        tmp_sum = 0

        # trainings sample, ([INPUT], [EXPECTED OUTPUT])
        for sample in input_samples:
            for i in range(n):
                input_layer[i].input = (sample[0][i])
            tmp = [0 for _ in range(n)]
            for j in range(len(output_layer)):
                output_layer[j].process_input()
                tmp[j] = (sample[1][j] - output_layer[j].output)**2 / 2
        return tmp


    def delta(self, input_samples, l_factor):
        input_layer = self.layers[0]
        output_layer = self.layers[-1]
        
        
        for j in range(len(input_layer)):
            input_layer[j][1] = input_layer[j][1] * l_factor * der_sigmoid() * (self.layers[1][j])
            pass



    def process(self):
        for layer in self.layers:
            for neuron in layer:
                neuron.process_input()

def basic_logic_test():
    p0 = Perceptron(1, 1)           # Maak een Perceptron aan met 1 inputs, treshold 1
    p1 = Perceptron(1, 1)           # Maak een Perceptron aan met 1 inputs, treshold 1
    pAND = Perceptron(2, 1)         # Maak een Perceptron aan met 2 inputs, treshold 1
    pOR = Perceptron(2, 0.5)        # Maak een Perceptron aan met 2 inputs, treshold 0.5
    pNOT = Perceptron(1, -0.5)      # Maak een Perceptron aan met 1 inputs, treshold -0.5
    pNAND = Perceptron(2, -0.5)

    p0.connect_to(pAND, 0, 0.5)     # Verbind p0 met pAND op ingang 0, weight = 0.5
    p1.connect_to(pAND, 1, 0.5)     # Verbind p1 met pAND op ingang 1, weight = 0.5
    p0.connect_to(pOR, 0, 0.5)      # Verbind p0 met pOR op ingang 0, weight = 0.5
    p1.connect_to(pOR, 1, 0.5)      # Verbind p1 met pOR op ingang 1, weight = 0.5
    p0.connect_to(pNOT, 0, -1)      # Verbind p0 met pNOT op ingang 0, weight = -1
    p0.connect_to(pNAND, 0, -0.5)
    p1.connect_to(pNAND, 1, -0.5)

    # AND-gate test
    for i in range(0,2):
        for j in range(0,2):
            p0.output, p1.output = i, j
            pAND.process_input()
            print(i, j, "=", pAND.output)
    print()

    # OR-gate test
    for i in range(0,2):
        for j in range(0,2):
            p0.output, p1.output = i, j
            pOR.process_input()
            print(i, j, "=", pOR.output)
    print()

    # NOT-gate test
    for i in range(0,2):
        p0.output = i
        pNOT.process_input()
        print(i, "=", pNOT.output)
    print()

    # NAND-gate test
    for i in range(0,2):
        for j in range(0,2):
            p0.output, p1.output = i, j
            pNAND.process_input()
            print(i, j, "=", pNAND.output)
    print()


def exercise_B_NOR():
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

    # TODO TODO urgent, please do
    # - Make function

    train_network(l_factor, inputs, desired_output, [p0,p1,p2], [pNOR])

    # for x in range(20000):
    #     if x % 100 == 0:
    #         print("Iteration:", x+1)
    #     for i in range(len(inputs)):
    #         p0.a, p1.a, p2.a = inputs[i][0], inputs[i][1], inputs[i][2]
    #         pNOR.process_input()            
    #         pNOR.update_weights(l_factor, desired_output[i])
    #         if x % 100 == 0:
                
    #             print(inputs[i][0], inputs[i][1], inputs[i][2], "=", pNOR.a)
    #     if x % 1000 == 0:
    #         print(",ksjfhkjsf")


def exercise_A_Neuron():
    treshold = 0
    weight = -1
    p0 = Perceptron(0, 0)
    p1 = Perceptron(0, 0)
    p2 = Perceptron(0, 0)

    # NOR:  0 0 0 = 1
    #       x x 1 = 0
    #       x 1 x = 0
    #       1 x x = 0

    pNOR = Perceptron(3, treshold)
    p0.connect_to(pNOR, 0, weight)
    p1.connect_to(pNOR, 1, weight)
    p2.connect_to(pNOR, 2, weight)
    
    print("3-input NOR-gate with treshold:{} and input weight:{}".format(treshold, weight))
    for i in range(0,2):
        for j in range(0,2):
            for k in range(0,2):
                p0.output, p1.output, p2.output = i, j, k
                pNOR.process_input()
                print(i, j, k, "=", pNOR.output)
    print()
    
    # Adder:    a b = x C
    #           0 0 = 0 0
    #           0 1 = 1 0
    #           1 0 = 1 0
    #           1 1 = 0 1

    treshold = -0.5
    weight_NAND = -0.5
    weight_NOT = -1

    pNAND_L0 = Perceptron(2, treshold)
    pNAND_L1_0 = Perceptron(2, treshold)
    pNAND_L1_1 = Perceptron(2, treshold)
    pNAND_L2 = Perceptron(2, treshold)
    pNOT_L2 = Perceptron(1, treshold)

    p0.connect_to(pNAND_L0, 0, weight_NAND)
    p1.connect_to(pNAND_L0, 1, weight_NAND)

    pNAND_L0.connect_to(pNAND_L1_0, 0, weight_NAND)
    p0.connect_to(pNAND_L1_0, 1, weight_NAND)

    p1.connect_to(pNAND_L1_1, 0, weight_NAND)
    pNAND_L0.connect_to(pNAND_L1_1, 1, weight_NAND)

    pNAND_L1_0.connect_to(pNAND_L2, 0, weight_NAND)
    pNAND_L1_1.connect_to(pNAND_L2, 1, weight_NAND)

    pNAND_L0.connect_to(pNOT_L2, 0, weight_NOT)

    print("1-bit Half Adder with treshold:{}, NAND-input weights:{} and NOT-input weight:{}".format(treshold, weight_NAND, weight_NOT))
    for i in range(0,2):
        for j in range(0,2):
            p0.output, p1.output = i, j
            pNAND_L0.process_input()
            pNAND_L1_0.process_input()
            pNAND_L1_1.process_input()
            pNAND_L2.process_input()
            pNOT_L2.process_input()
            print(i, j, "= sum:{} carry:{}".format(pNAND_L2.output, pNOT_L2.output))
    

def exercise_C_XOR():

    pass


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
