'''
JOSE JAVIER JO ESCOBAR
14343
MANIPULACION DE DATOS

'''
import numpy as np
from PIL import Image
import math

def vectorized_result(j):
    """Return a 10-dimensional unit vector with a 1.0 in the jth
    position and zeroes elsewhere.  This is used to convert a digit
    (0...9) into a corresponding desired output from the neural
    network."""
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e

def data_split(data):

    data_len = len(data[0])
    # print(data_len)
    dataset = data[0]
    results = data[1]
    print(dataset.shape)
    #print(results.shape)

    indices = np.random.permutation(data_len)
    split1 = math.trunc(data_len*0.7)
    split2 = math.trunc(data_len*0.85)
    tr_d = dataset[indices[:split1]]
    va_d = dataset[indices[split1:split2]]
    te_d = dataset[indices[split2:]]
    tr_d_res = results[indices[:split1]]
    va_d_res = results[indices[split1:split2]]
    te_d_res = results[indices[split2:]]

    train_inputs = [np.reshape(x, (784,1)) for x in tr_d]
    train_results = [vectorized_result(y) for y in tr_d_res]
    train_data = (train_inputs,train_results)

    val_inputs = [np.reshape(x, (784, 1)) for x in va_d]
    val_data = (val_inputs, va_d_res)

    test_inputs = [np.reshape(x, (784, 1)) for x in te_d]
    test_data = (test_inputs, te_d_res)

    return (train_data, val_data, test_data)

def data_load():

    circle = np.load('../Datasets/circle.npy')
    eggs = np.load('../Datasets/eggs.npy')
    house = np.load('../Datasets/house.npy')
    question_mark = np.load('../Datasets/question_mark.npy')
    sad_face = np.load('../Datasets/sad_face.npy')
    smiley_face = np.load('../Datasets/smiley_face.npy')
    square = np.load('../Datasets/square.npy')
    tree = np.load('../Datasets/tree.npy')
    triangle = np.load('../Datasets/triangle.npy')
    mickey = np.load('../Datasets/mickeymouse.npy')
    # print(len(triangle))
    # print(house.shape)
    # print(triangle.shape)

    circle = circle[:4000]
    eggs = eggs[:4000]
    house = house[:4000]
    question_mark= question_mark[:4000]
    sad_face = sad_face[:4000]
    smiley_face = smiley_face[:4000]
    square = square[:4000]
    tree = tree[:4000]
    triangle = triangle[:4000]
    mickey = mickey[:4000]

    results = np.concatenate((
                            np.repeat(0, len(circle)),
                            np.repeat(1, len(eggs)),
                            np.repeat(2, len(house)),
                            np.repeat(3, len(question_mark)),
                            np.repeat(4, len(sad_face)),
                            np.repeat(5, len(smiley_face)),
                            np.repeat(6, len(square)),
                            np.repeat(7, len(tree)),
                            np.repeat(8, len(triangle)),
                            np.repeat(9, len(mickey))
                        ))
    input = np.concatenate((
                            circle,
                            eggs,
                            house,
                            question_mark,
                            sad_face,
                            smiley_face,
                            square,
                            tree,
                            triangle,
                            mickey
                        ))
    input[input > 1] = 1
    # print(input.shape)
    # print(results.shape)
    dataset = (input, results)

    train_data, val_data, testing_data = data_split(dataset)
    return (train_data,val_data,testing_data)


def image_read():

    input = Image.open("../input/Untitled.bmp")
    # input.show()
    img_array = np.array(input)
    res = [0 if img_array[i][j][0] == 255 else 1 for i in range(28) for j in range(28)]
    return res


