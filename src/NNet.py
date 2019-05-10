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
        """
        En la lista 'Sizes" podemos ver el numero de neuronas en cada capa respectiva de la red.
        Por ejemplo
        si la lista era de [2,3,1] seria una red de tres capas, con la primer capa con 2 neuronas , 3 neuronas en la segunda capa
        y 1 neurona de output

        La red es iniciada aleatoriamente  usando la distribucion gaussiana con media 0  y varianza de 1
        En la primer capa se asume que es la capa de entrada y por facilidad no vamos a establecer ningun sesgo
        para las neuronas de entrada ya que unicamente en la ultima capa se calcularian los sesgos.
        """
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x)
                        for x, y in zip(sizes[:-1], sizes[1:])]

    def showinfo(self):
        print(len(self.weights))
        print(len(self.biases))

    def feedforward(self, a):
        """
        Devolvemos el output de la red al ingresarlo en 'a'
         """
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
        """
        Se entrana la red neuronal con el desesnso de gradiente estocastico ya que es mas rapido su entrenamiento
        El 'training data' es la lista de tuplas (x,y) que representa las entradas de entrenamiento
        y el output deseado.
        Los otros parametros no opcionales se explican solos, Si damos el 'test data' la red evaluara
        contra los datos de prueba despues de cada corrida y proceso parcial que se muestra
        Es util para ver el proceso pero vuelve lento un poco.
        """
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
        """
        Se actualizan los pesos y bias de la red aplicando el desenso del gradiente y el backprop
        en una pequena porcion de los datos de aprendizaje

        Basicamente es una lista de tuplas  *x,y) , y, eta --> que es la tasa de aprendizaje
        """
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
        """
        Devolvemos una tupla (nabla_b,nabla_v) que seria el gradiente de la funcion de costo C_x
        nabla_b y nabla_v ---> son listas capa por capa de matrices de numpy, son donde se guardan los cambios
        a los pesos y los biases por capa
        """
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        activation = x
        activations = [x] # Lista que almacena todas las activaciones capa por capa
        zs = [] # Lista que almacena todos los vectores z, capa por capa

        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation)+b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        # BACKWARD PASS
        delta = self.cost_derivative(activations[-1], y) * \
            sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        '''
        Se toma en cuenta que la variable 1 es el bucle de abajo que es utilizado casi directamente 
        a la notacion que lleva el backprop
        1 =1 es la ultima capa de neuronas
        1 =2 es la segunda capa y asi
        
        Son al final los indices negativos de las listas creadas por python
        '''

        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return (nabla_b, nabla_w)

    def evaluate(self, test_data):
        """
        Devolvemos el numero de inputs de prueba que las neuronas de la red produce para el resultado correcto
        El output de la red es el indice de cualquera de las neuronas en la capa final que tiene la activacion mas alta.
        """
        test_results = [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

    def cost_derivative(self, output_activations, y):
        """
        Damos el vector de la derivada parcial ---> parcial C_x que es la parcial para las activaciones del output
        """
        return (output_activations-y)

def sigmoid(z):
    """ La función sigmoidea. """
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
    """Derivada de la función sigmoidea."""
    return sigmoid(z)*(1-sigmoid(z))