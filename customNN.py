import numpy as np
from IPython.display import clear_output
import matplotlib.pyplot as plt


def softmax_crossentropy_with_logits(logits, reference_answers, regularise=False, alpha=0.01, weights=None):
    """Compute crossentropy from logits[batch,n_classes] and ids of correct answers"""

    logits_for_answers = logits[np.arange(len(logits)), reference_answers]

    xentropy = - logits_for_answers + np.log(np.sum(np.exp(logits), axis=1))

    if regularise:  # add l2 regularisation. Note adding this to gradient w.r.t last activattion propagates it across all net
        assert weights is not None, 'Missing weights'
        xentropy = xentropy + alpha * np.sum(weights**2, axis=None)  #first across all columns, then all rows

    return xentropy


def grad_softmax_crossentropy_with_logits(logits,reference_answers,regularised=False, alpha=0.01, weights=None):
    """Compute crossentropy gradient from logits[batch,n_classes] and ids of correct answers"""

    ones_for_answers = np.zeros_like(logits)
    ones_for_answers[np.arange(len(logits)), reference_answers] = 1

    softmax = np.exp(logits) / np.exp(logits).sum(axis=-1, keepdims=True)
    grad = (- ones_for_answers + softmax ) / logits.shape[0]

    if regularised:
        assert weights is not None, 'Missing weights'
        grad += alpha * 2 * np.sum(weights, axis=None) / logits.shape[0]

    return grad


class Layer:
    """
    A building block. Each layer is capable of performing two things:

    - Process input to get output:  output = layer.forward(input)

    - Propagate gradients through itself:    grad_input = layer.backward(input, grad_output)

    Some layers also have learnable parameters which they update during layer.backward.
    """
    def __init__(self):
        """Here you can initialize layer parameters (if any) and auxiliary stuff."""
        # A dummy layer does nothing
        pass

    def forward(self, input):
        """
        Takes input data of shape [batch, input_units], returns output data [batch, output_units]
        """
        # A dummy layer just returns whatever it gets as input.
        return input

    def backward(self, input, grad_output):
        """
        Performs a backpropagation step through the layer, with respect to the given input.

        To compute loss gradients w.r.t input, you need to apply chain rule (backprop):

        d loss / d x  = (d loss / d layer) * (d layer / d x)

        Luckily, you already receive d loss / d layer as input, so you only need to multiply it by d layer / d x.

        If your layer has parameters (e.g. dense layer), you also need to update them here using d loss / d layer
        """
        # The gradient of a dummy layer is precisely grad_output, but we'll write it more explicitly
        num_units = input.shape[1]

        d_layer_d_input = np.eye(num_units)

        return np.dot(grad_output, d_layer_d_input) # chain rule


class ReLU(Layer):
    """
    Regularised linear unit. Just takes the value if positive, else returns 0.
    This class inherits from layer.

    """
    def __init__(self):
        Layer.__init__(self)
        """ReLU layer simply applies elementwise rectified linear unit to all inputs"""
        pass

    def forward(self, intake):
        """Apply element-wise ReLU to [batch, input_units] matrix"""
        return np.maximum(intake, 0)  # maximum for element-wise maxima

    def backward(self, intake, grad_output):
        """Compute gradient of loss w.r.t. ReLU input"""
        relu_grad = intake > 0
        return grad_output * relu_grad


class Dense(Layer):
    def __init__(self, input_units, output_units, learning_rate=0.1, random_seed=20):
        """

        :param input_units: number of features (columns) of the input to the layer. In NN jargon they are units
        :param output_units: number of features (columns) of the output to the layer
        :param learning_rate: for each step of back-propagation, multiply gradient by
        """
        Layer.__init__(self)  # when a class inherits, init of super-class has to be initialised.
        """
        A dense layer is a layer which performs a learned affine transformation:
        f(x) = <W*x> + b
        """
        self.learning_rate = learning_rate
        self.output_units = output_units
        self.input_units = input_units
        np.random.seed(20)

        # initialize weights with small random numbers. We use normal initialization,
        # but surely there is something better. Try this once you got it working: http://bit.ly/2vTlmaJ
        self.weights = np.random.randn(input_units, output_units) * 0.01
        self.biases = np.zeros(output_units)

    def forward(self, intake):
        """
        Perform an affine transformation:
        f(x) = <W*x> + b

        input shape: [batch, input_units]
        output shape: [batch, output units]
        """
        assert intake.shape[1] == self.weights.shape[0], \
            'Input feats are %s and Layer feats are %s' % (intake.shape[1], self.weights.shape[0] )
        #dot_product = np.dot(input, self.weights) + self.biases  # why dot and not matmul?
        return np.dot(intake, self.weights) + self.biases  # preferred to matmul or @,
        #  see https://numpy.org/devdocs/reference/generated/numpy.matmul.html#numpy.matmul

    def backward(self, intake, grad_output):
        """
        input: X values, or result from a previous layer
        grad_output: incoming gradient, used for chain rule
        """

        # compute d f / d x = d f / d dense * d dense / d x
        # where d dense/ d x = weights transposed
        grad_input = np.dot(grad_output, self.weights.T)  # dZ.dot(weights transpose)
        # compute gradient w.r.t. weights and biases
        grad_weights = np.dot(intake.T, grad_output)
        grad_biases = grad_output.sum(axis=0)  # as the derivative is 1.

        if not grad_weights.shape == self.weights.shape and grad_biases.shape == self.biases.shape:
            print('%s & %s' % (grad_biases.shape, self.biases.shape))
            raise ValueError('Cannot perform dot product with these dimensions')
        # Here we perform a stochastic gradient descent step.
        self.weights = np.subtract(self.weights, self.learning_rate * grad_weights)
        self.biases = np.subtract(self.biases,  self.learning_rate * grad_biases)
        return grad_input

    def get_weights(self):  # add a method for implementing regularisation
        return self.weights


class NeuralNetwork(list):
    def __init__(self, X=None, y=None, random_seed=20, l2_regularisation=False, alpha=0.01):
        super().__init__()
        self.X = X
        self.y = y
        self.l2_regularisation = l2_regularisation
        self.alpha = alpha
        self.random_seed = random_seed

    def forward(self, intake=None):
        """

        :param intake: a given set with same features as training set. If not, use training set.
        :return: outputs of each layer

        Compute activations of all network layers by applying them sequentially.
        Return a list of activations for each layer.
        Make sure last activation corresponds to network logits.
        """
        activations = [self.X if intake is None else intake]  # depending if it is training set or validation one
        assert activations[0] is not None, 'Please give parameter X to method'

        for n in range(len(self)):  # forward pass across all layers
            layer = self[n]
            activations.append(layer.forward(intake=activations[-1]))

        assert len(activations[1:]) == len(self)
        return activations[1:]

    def predict(self, X=None):
        """
        Compute network predictions.
        """
        X = self.X if X is None else X
        logits = self.forward(intake=X)[-1]
        return np.argmax(logits, axis=1)

    def training_step(self, X=None, y=None, regularise=False):
        """
        Train  network on a given batch of X and y.
        All layer activations are run forwards
        Then they are run layer.backward going from last to first layer.
        After you called backward for all layers, all Dense layers have already made one gradient step.
        """
        # X = self.X if X is None else X
        # y = self.y if y is None else y
        self.X = X if X is not None else self.X
        self.y = y if y is not None else self.y
        assert self.X is not None and self.y is not None, 'Please specify parameters X and y'

        # Get the layer activations
        layer_activations = self.forward(intake=None)  # None as self.X is used
        layer_inputs = [self.X]+layer_activations  # layer_input[i] is an input for network[i]
        logits = layer_activations[-1]  # we are assuming last layer is dense
        # Compute the loss and the initial gradient
        loss = softmax_crossentropy_with_logits(logits, self.y, regularise=regularise, weights=self[-1].get_weights())
        loss_grad = grad_softmax_crossentropy_with_logits(logits, self.y,
                                                          regularised=regularise, weights=self[-1].get_weights())

        #  propagate gradients through the network
        backward_grads = [loss_grad]
        for n in reversed(range(len(self))):
            layer = self[n]
            grads_input = layer.backward(intake=layer_inputs[n], grad_output=backward_grads[-1])
            backward_grads.append(grads_input)

        return np.mean(loss)

    def train(self, epochs, batchsize=32, shuffle=False, X=None, y=None, X_valid=None, y_valid=None, visualise=False):

        self.X = X if X is not None else self.X
        self.y = y if y is not None else self.y

        assert self.X is not None and self.y is not None, 'Missing X and y'
        x_rows = self.X.shape[0]

        # shuffle
        if shuffle:
            indexes = np.random.permutation(list(range(self.X.shape[0])))
            self.X = self.X[indexes]

        mean_loss_array = []
        mean_batch_array = []
        for epoch in range(epochs):  # when all training samples are looked at, epoch finishes

            for batch_start in range(0, x_rows, batchsize):
                mean_loss = self.training_step(X=X[batch_start: batch_start+batchsize, :],
                                               y=y[batch_start: batch_start+batchsize], regularise=False)
                mean_loss_array += [mean_loss]
                mean_batch_array += [np.mean(X[batch_start: batch_start+batchsize, :])]

            if visualise:

                clear_output()
                print("Epoch", epoch)
                print("Loss:", np.mean(mean_loss_array))
                plt.grid()
                plt.plot(mean_loss_array, label='batch mean loss', color='firebrick')
                plt.xlabel('Iteration')
                plt.ylabel('Loss')
                plt.show()

        self.X = X
        self.y = y
        return mean_batch_array


