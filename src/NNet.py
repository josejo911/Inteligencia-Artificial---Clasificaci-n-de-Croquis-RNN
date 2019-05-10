'''
JOSE JAVIER JO ESCOBAR
14343
RED NEURAL
PROYECTO 2 IA
'''


'''
REFERENCIAS 
BLOG https://towardsdatascience.com/how-to-build-your-own-neural-network-from-scratch-in-python-68998a08e4f6
BLOG2 https://medium.com/analytics-vidhya/neural-networks-for-digits-recognition-e11d9dff00d5
GIT https://gist.github.com/jamesloyys/ff7a7bb1540384f709856f9cdcdee70d#file-neural_network_backprop-py
DATA https://quickdraw.withgoogle.com/data

'''

#### LIBRERIAS
import numpy as np
import random


class NNet(object):

    def __init__(self, sizes):

        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x)
                        for x, y in zip(sizes[:-1], sizes[1:])]

    def showinfo(self):
        print(len(self.weights))
        print(len(self.biases))

    def feedforward(self, a):

        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a)+b)
        return a

    def ff(self, input):
        z2 = np.dot(self.weights[0], input) + self.biases[0]
        a2 = sigmoid(z2)
        z3 = np.dot(self.weights[1], a2) + self.biases[1]
        a3 = sigmoid(z3)

        return a3

    def SGD(self, training_data, corrida, mini_batch_size, eta,
            test_data=None):

        if test_data: n_test = len(test_data)
        n = len(training_data)
        for j in range(corrida):
            random.shuffle(training_data)
            mini_batches = [
                training_data[k:k+mini_batch_size]
                for k in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.batch(mini_batch, eta)
            if test_data:
                print("EPOCA {0}: {1} / {2}".format(
                    j, self.evaluate(test_data), n_test))
            else:
                print("EPOCA {0} completada".format(j))

    def batch(self, mini_batch, eta):

        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backpropagation(x, y)
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.weights = [w-(eta/len(mini_batch))*nw
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b-(eta/len(mini_batch))*nb
                       for b, nb in zip(self.biases, nabla_b)]

    def backpropagation(self, x, y):

        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        activation = x
        activations = [x] 
        zs = [] 

        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation)+b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        delta = self.cost_derivative(activations[-1], y) * \
            sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())


        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return (nabla_b, nabla_w)

    def evaluate(self, test_data):

        test_results = [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

    def cost_derivative(self, output_activations, y):

        return (output_activations-y)

def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
    return sigmoid(z)*(1-sigmoid(z))