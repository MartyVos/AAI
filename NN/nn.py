#!/usr/bin/python
import math
from random import uniform, seed
from typing import List


def sigmoid(z):
    return (1 / (1 + math.exp(-z)))

def der_sigmoid(z):
    return sigmoid(1-sigmoid(z))



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
    
    '''
    def connect_to(self, other_id):
        self.next_layer[other_id].inputs[self.id] = self
    '''
    

    def update_weights(self, learning_factor):
        # Update weights
        tmp = 0.0

        if self.prev_layer is not None:
            for i in self.prev_layer:
                tmp += (i.delta * self.weights[i.id])
            self.delta = der_sigmoid(self.z) * tmp

            for node in range(len(self.weights)):
                self.weights[node] += learning_factor * self.delta * self.prev_layer[node].a
            
        # Update bias
        self.bias = learning_factor * self.delta
        


    def process_input(self):
        tmp = 0.0
        for i in self.prev_layer:
            tmp += (i.a * self.weights[i.id])     # i = tuple: (Neuron, weight)
        self.a = sigmoid(tmp + self.bias)


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
    

def main():
    weight = -1
    p0 = SigmoidNeuron(0, None)
    p1 = SigmoidNeuron(1, None)
    p2 = SigmoidNeuron(2, None)

    # NOR:  0 0 0 = 1
    #       x x 1 = 0
    #       x 1 x = 0
    #       1 x x = 0

    pNOR = SigmoidNeuron(0, [p0, p1, p2])


    
    print("3-input NOR-gate with input weight:{}".format(weight))
    for x in range(10):
        print("Iteration", x)
        for i in range(0,2):
            for j in range(0,2):
                for k in range(0,2):
                    p0.a, p1.a, p2.a = i, j, k
                    pNOR.process_input()
                    print(i, j, k, "=", pNOR.a)
        pNOR.update_weights(math.pi)


    return
     
#region Zooi
    # basic_logic_test()
    # exercise_A_Neuron()
    p0 = SigmoidNeuron(0, 1, True)           # Maak een SigmoidNeuron aan met 0 inputs, bias 1, is een input neuron
    p1 = SigmoidNeuron(0, 1, True)           # Maak een SigmoidNeuron aan met 0 inputs, bias 1, is een input neuron
    pAND = SigmoidNeuron(2, 1, False)        # Maak een SigmoidNeuron aan met 2 inputs, bias 1, is géén input neuron
    
    p0.connect_to(pAND, 0, 1)
    p1.connect_to(pAND, 1, 1)


    N = NeuralNetwork(2)

    N.addToLayer(0, p0)
    N.addToLayer(0, p1)
    N.addToLayer(1, pAND)

    p0.input, p1.input = 1, 1
    N.process()

    inputs = [([0, 0], [0]),
              ([0, 1], [0]),
              ([1, 0], [0]),
              ([1, 1], [1])
    ]


    # for i in range(0,2):
    #     for j in range(0,2):
    #         p0.input, p1.input = i, j
    #         p0.process_input()
    #         p1.process_input()
    #         pAND.process_input()
    #         print(i, j, "=", pAND.output)
    # print()      
#endregion

if __name__ == "__main__":
    seed(0)
    main()
