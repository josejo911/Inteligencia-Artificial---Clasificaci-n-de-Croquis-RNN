'''
    JOSE JAVIER JO ESCOBAR
    14343
    MAIN
'''
import numpy as np
from NNet import NNet
import DatasetReader as DR

NN = NNet([784, 60, 10])
Training_data, Verification_data, Testing_data = DR.data_load()
Training_data = list(map(lambda x, y : (x, y), Training_data[0], Training_data[1]))
test = list(map(lambda x, y : (x, y), Testing_data[0], Testing_data[1]))

NN.SGD(Training_data, 5, 40, 3.0, test_data=test) # 30,10,3.0 best

x = np.array(DR.image_read())
# for i in range(28):
#     print([x[i*28 + j] for j in range(28)])
# print(x)

res = NN.ff(np.array([x]).reshape(784,1))
#print(res.shape)
res = res * 100

print(' \n 0 ---> CIRCULO\n'
      ' 1 ---> HUEVO\n'
      ' 2 ---> CASA\n'
      ' 3 ---> INTERROGACION\n'
      ' 4 ---> CARA TRISTE\n'
      ' 5 ---> CARA FELIZ\n'
      ' 6 ---> CUADRADO\n'
      ' 7 ---> ARBOL\n'
      ' 8 ---> TRIANGULO\n'
      ' 9 ---> MICKEY MOUSE\n')

#print(np.argmax(res))
print('\nEstoy a un  ' + '{:.5}%'.format(str(res[np.argmax(res)])), 'de seguridad que es el numero ', np.argmax(res))

#print(res)