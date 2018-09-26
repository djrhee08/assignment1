from three_layer_neural_network import NeuralNetwork, generate_data, plot_decision_boundary
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
import sys

class DeepNeuralNetwork(NeuralNetwork):

    def __init__(self, num_layers, layer_sizes, actFun_type='relu', reg_lambda=0.01, seed=2):
        '''
        :param num_layers : the number of layers
        :param layer_sizes : layers sizes in list
        :param actFun_type: type of activation function. 3 options: 'tanh', 'sigmoid', 'relu'
        :param reg_lambda: regularization coefficient
        :param seed: random seed
        '''
        self.num_layers = num_layers
        self.layer_sizes = layer_sizes
        self.actFun_type = actFun_type
        self.reg_lambda = reg_lambda

        if num_layers != len(layer_sizes):
            print(layer_sizes, " and the length ", num_layers-1, " not matched!")
            sys.exit()

        self.W = []
        self.b = []

        # initialize the weights and biases in the network
        np.random.seed(seed)
        for i in range(num_layers-1):
            self.W.append(np.random.rand(self.layer_sizes[i], self.layer_sizes[i+1]) / np.sqrt(self.layer_sizes[i]))
            self.b.append(np.zeros((1, self.layer_sizes[i+1])))

        #for i in range(len(self.W)):
        #    print("W shape : ", self.W[i].shape)

    def feedforward(self, X, actFun):
        '''
        feedforward builds a 3-layer neural network and computes the two probabilities,
        one for class 0 and one for class 1
        :param X: input data
        :param actFun: activation function
        :return:
        '''

        # YOU IMPLEMENT YOUR feedforward HERE
        self.z = []
        self.a = []

        for i in range(self.num_layers - 1):
            if i == 0:
                self.z.append(np.dot(X, self.W[i]) + self.b[i])
            else:
                self.z.append(np.dot(self.a[i-1], self.W[i]) + self.b[i])

            self.a.append(actFun(self.z[i]))

        exp_scores = np.exp(self.z[-1])
        self.probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

        return None

    def backprop(self, X, y):
        '''
        backprop implements backpropagation to compute the gradients used to update the parameters in the backward step
        :param X: input data
        :param y: given labels
        :return: dL/dW, dL/db
        '''

        dW = []
        db = []

        # IMPLEMENT YOUR BACKPROP HERE
        num_examples = len(X)
        delta3 = self.probs
        delta3[range(num_examples), y] -= 1

        # Calculate dW and db in backwards, [dW[-1], dW[-2] ... , dW[1], dW[0]]
        for i in range(self.num_layers - 2, -1, -1):
            if i == self.num_layers - 2:
                dW.append(np.dot(self.a[i-1].T, delta3))
                db.append(np.sum(delta3, axis=0, keepdims=True))
                dhidden = np.dot(delta3, self.W[i].T) * self.diff_actFun(self.a[i-1], self.actFun_type)
            elif i == 0:
                dW.append(np.dot(X.T, dhidden))
                db.append(np.sum(dhidden, axis=0, keepdims=True))
            else:
                dW.append(np.dot(self.a[i-1].T, dhidden))
                db.append(np.sum(dhidden, axis=0, keepdims=True))
                dhidden = np.dot(dhidden, self.W[i].T) * self.diff_actFun(self.a[i-1], self.actFun_type)

        # Reverse the order of dW and db to match it with W and b
        # Now, it is in order of [dW[0], dW[1] ... , dW[-2], dW[-1]]
        dW.reverse()
        db.reverse()

        #for i in range(len(dW)):
        #    print("dW shape : ", dW[i].shape)

        return dW, db

    def calculate_loss(self, X, y):
        '''
        calculate_loss computes the loss for prediction
        :param X: input data
        :param y: given labels
        :return: the loss for prediction
        '''

        num_examples = len(X)
        self.feedforward(X, lambda x: self.actFun(x, type=self.actFun_type))
        # Calculating the loss

        # YOU IMPLEMENT YOUR CALCULATION OF THE LOSS HERE
        y_encoding = np.zeros([num_examples, self.layer_sizes[-1]])
        y_encoding[range(num_examples), y] += 1

        data_loss = -np.sum((np.log(self.probs) * y_encoding).flatten())

        # Add regulatization term to loss (optional)
        sum_W = 0
        for i in range(len(self.W)):
            sum_W += np.sum(self.W[i])

        data_loss += self.reg_lambda / 2 * (np.sum(np.square(sum_W)))

        return (1. / num_examples) * data_loss

    def fit_model(self, X, y, epsilon=0.001, num_passes=50000, print_loss=True):
        '''
        fit_model uses backpropagation to train the network
        :param X: input data
        :param y: given labels
        :param num_passes: the number of times that the algorithm runs through the whole dataset
        :param print_loss: print the loss or not
        :return:
        '''
        # Gradient descent.
        for i in range(0, num_passes):
            # Forward propagation
            self.feedforward(X, lambda x: self.actFun(x, type=self.actFun_type))
            # Backpropagation
            dW, db = self.backprop(X, y)

            # Add regularization terms (b1 and b2 don't have regularization terms)
            for j in range(len(self.W)):
                dW[j] += self.reg_lambda * self.W[j]

            # Gradient descent parameter update
            for j in range(len(self.W)):
                self.W[j] += -epsilon * dW[j]
                self.b[j] += -epsilon * db[j]

            # Optionally print the loss.
            # This is expensive because it uses the whole dataset, so we don't want to do it too often.
            if print_loss and i % 1000 == 0:
                print("Loss after iteration %i: %f" % (i, self.calculate_loss(X, y)))

#class Layer(DeepNeuralNetwork):
#    def __init__(self, num_layers, layer_sizes, actFun_type='tanh', reg_lambda=0.01, seed=2):


def main():
    # generate and visualize Make-Moons dataset
    X, y = generate_data()
    print(X.shape, y.shape)
    #plt.scatter(X[:, 0], X[:, 1], s=40, c=y, cmap=plt.cm.Spectral)
    #plt.show()

    layer_sizes = [2,4,6,6,4,2]
    num_layers = len(layer_sizes)
    model = DeepNeuralNetwork(num_layers=num_layers, layer_sizes=layer_sizes, actFun_type='sigmoid', reg_lambda=0.01, seed=2)
    model.fit_model(X,y)
    model.visualize_decision_boundary(X,y)

if __name__ == "__main__":
    main()