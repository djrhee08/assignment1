from three_layer_neural_network import NeuralNetwork
import numpy as np

class DeepNeuralNetwork(NeuralNetwork):

    def __init__(self, num_layers, layer_sizes, actFun_type='tanh', reg_lambda=0.01, seed=2):
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

        self.W = []
        self.b = []

        # initialize the weights and biases in the network
        np.random.seed(seed)
        for i in range(num_layers):
            self.W.append(np.random.rand(self.layer_sizes[i], self.layer_sizes[i+1]) / np.sqrt(self.layer_sizes[i]))
            self.b.append(np.zeros((1, self.layer_sizes[i+1])))

            print(self.W[i].shape, self.b[i].shape)


    def feedforward(self, X):
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

        #self.z1 = np.dot(X, self.W1) + self.b1
        #self.a1 = actFun(self.z1)
        #self.z2 = np.dot(self.a1, self.W2) + self.b2

        #exp_scores = np.exp(self.z2)
        #self.probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

        return None


    def Layer(self):
    """


def main():
    model = DeepNeuralNetwork(num_layers=4, layer_sizes=[2,3,3,3,2], actFun_type='tanh', reg_lambda=0.01, seed=2)


if __name__ == "__main__":
    main()