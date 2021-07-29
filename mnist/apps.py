from django.apps import AppConfig
from django.conf import settings
from mnist.NeuralNetwork import NeuralNetwork
import numpy as np
import os

class MnistConfig(AppConfig):
    
    # Train Data Func
    def test_data(NN):
        test_data = open(os.path.join(settings.TRAIN_DATA, 'mnist_test.csv'), 'r')
        test_data_list = test_data.readlines()
        test_data.close()
        
        total = len(test_data_list)
        total_corr = 0

        for rec in test_data_list:
            val = rec.split(',')
            correct = int(val[0])
            inputs = (np.asfarray(val[1:]) / 255.0 * 0.99) + 0.01
            outputs = NN.query(inputs)
            label = np.argmax(outputs)
            total_corr += 1 if label == correct else 0

        print("Accuracy is {}%".format((total_corr / total) * 100))

    # Load Mnist Data
    data = open(os.path.join(settings.TRAIN_DATA, 'mnist_train.csv'), 'r')
    data_list = data.readlines()
    data.close()
    
    NN = NeuralNetwork(784, 200, 10, 0.3)

    for record in data_list:
        values = record.split(',')
        inputval = (np.asfarray(values[1:]) / 255.0 * 0.99) + 0.01
        
        target = np.zeros(10) + 0.01
        target[int(values[0])] = .99

        NN.train(inputval, target)

    print("MODEL TRAINED!!")
    test_data(NN)