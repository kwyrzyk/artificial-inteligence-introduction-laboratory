class MultiLevelPerceptron:
    def __init__(self, layer_sizes):
        """
        layer_sizes: List of integers specifying the number of neurons in each layer.
                     Example: [2, 4, 3, 1] -> 2 inputs, 4 neurons in the first hidden layer,
                     3 neurons in the second hidden layer, 1 output.
        """
        self.num_layers = len(layer_sizes) - 1  # Number of weight matrices = number of layers - 1
        self.layer_sizes = layer_sizes
        self.weights = []  # List to store weight matrices
        self.biases = []   # List to store biases

        self.initialise_weights()

    def initialise_weights(self):
        weights = []
        biases = []
        for i in range(self.num_layers):
            weights.append(np.random.uniform(-1, 1, (self.layer_sizes[i], self.layer_sizes[i+1])))
            biases.append(np.random.uniform(-1, 1, (1, self.layer_sizes[i+1])))
        self.weights = weights
        self.biases = biases


    def forward(self, inputs, activation=np.sin):
        """
        Perform forward propagation through the network.
        inputs: Input data, shape (num_samples, num_inputs)
        """
        self.activations = [inputs]  # Store all layer activations, starting with the input
        current_input = inputs

        for i in range(self.num_layers):
            # Compute activation for the current layer
            current_input = activation(np.dot(current_input, self.weights[i]) + self.biases[i])
            self.activations.append(current_input)
        return current_input
    
    def backward(self, targets, learning_rate, activation_derivative=sin_derivative):
        """
        Perform backpropagation to update weights and biases.
        targets: True output values, shape (num_samples, num_outputs)
        learning_rate: Learning rate for weight updates
        """
        # Compute the error at the output layer
        output_error = self.activations[-1] - targets
        deltas = [output_error * activation_derivative(self.activations[-1])]

        # Backpropagate the error through the hidden layers
        for i in range(self.num_layers - 1, 0, -1):
            delta = np.dot(deltas[0], self.weights[i].T) * activation_derivative(self.activations[i])
            deltas.insert(0, delta)  # Insert at the beginning to reverse the order

        # Update weights and biases
        for i in range(self.num_layers):
            self.weights[i] -= np.dot(self.activations[i].T, deltas[i]) * learning_rate
            self.biases[i] -= np.sum(deltas[i], axis=0, keepdims=True) * learning_rate