import numpy as np
from Functions import sigmoid, derivative_sigmoid


class FeedForwardNeuralNetwork:
    def __init__(self, *args, **kwargs):
        self.__w = []
        self.__b = []
        self.__delta = []

        for i in range(0, len(args) - 1):
            self.__w.append(np.random.random_sample((args[i], args[i + 1])))
            self.__b.append(np.zeros((args[i + 1]), ))
            self.__delta.append(np.zeros((args[i], args[i + 1])))

        self.EPOCHS = kwargs['epochs']
        self.LEARN_RATE = kwargs['learn_rate']
        self.ALPHA = kwargs['alpha']

    def feedforward(self, input_values):
        """
        Feedforwarding input values
        :param input_values: input values
        :return: outputs on each layer, actual output
        """
        __in = []
        __out = []

        for layer in range(len(self.__w)):
            __in.append(np.dot(input_values, self.__w[layer]) + self.__b[layer])
            __out.append(sigmoid(__in[layer]))
            input_values = __out[layer]

        return __in, __out, input_values

    @staticmethod
    def output_delta(expected, actual):
        """
        Calculating output deltas
        :param expected: expected output
        :param actual: actual output
        :return: output deltas
        """
        return expected - actual

    @staticmethod
    def hidden_grad(learn_rate, grad):
        """
        Calculating delta weights for hidden layer
        :param learn_rate: learnability of neural network
        :param grad: gradient of layer
        :return: delta weights of layer
        """
        return learn_rate * grad

    @staticmethod
    def hidden_delta(weight, delta, output):
        """
        Calculating hidden layer deltas
        :param weight: weights of current layer
        :param delta: delta of next layer
        :param output: outputs of previous layer
        :return: hidden layer deltas
        """
        return np.dot(delta, weight.T) * derivative_sigmoid(output)

    def train(self, inputs, expectation):
        """
        Training Neural Network
        :param inputs: np.array of inputs data
        :param expectation: np.array of predictions for inputs data
        """
        for _ in range(self.EPOCHS):
            for x, y in zip(inputs, expectation):
                x = x[np.newaxis, :]
                y = y[:, np.newaxis]

                _in, _out, prediction = self.feedforward(x)
                dl = -2 * self.output_delta(y, prediction)

                for layer in range(len(self.__w) - 1, 0, -1):
                    grad = np.dot(_out[layer - 1].T, dl)
                    dl = self.hidden_delta(self.__w[layer], dl, _out[layer - 1])

                    tri = self.hidden_grad(self.LEARN_RATE, grad)
                    self.__w[layer] -= tri + self.ALPHA * self.__delta[layer]
                    self.__b[layer] -= np.nansum(tri)
                    self.__delta[layer] = tri

                grad = np.dot(x.T, dl)
                tri = self.hidden_grad(self.LEARN_RATE, grad)
                self.__w[0] -= tri + self.ALPHA * self.__delta[0]
                self.__b[0] -= np.nansum(tri)
                self.__delta[0] = tri

    def predict(self, x):
        return np.max(self.feedforward(x)[-1])


data_set = np.array([
    [1],
    [3],
    [5],
    [7],
])

data_y = np.array([
    [3],
    [2],
    [5],
    [3],
])


model = FeedForward(1, 4, 1, alpha=0.0, learn_rate=0.0003, epochs=10)
model.train(data_set, data_y)


print(model.predict(np.array([[2]])))
print(model.predict(np.array([[4]])))
print(model.predict(np.array([[6]])))
