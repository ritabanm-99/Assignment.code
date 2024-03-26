import numpy as np
from pathlib import Path

def load_data_small():
    """ Load small training and validation dataset

        Returns a tuple of length 4 with the following objects:
        X_train: An N_train-x-M ndarray containing the training data (N_train examples, M features each)
        y_train: An N_train-x-1 ndarray contraining the labels
        X_val: An N_val-x-M ndarray containing the training data (N_val examples, M features each)
        y_val: An N_val-x-1 ndarray contraining the labels
    """
    script_dir = Path(__file__).parent
    train_all = np.loadtxt(f'{script_dir}/data/smallTrain.csv', dtype=int, delimiter=',')
    valid_all = np.loadtxt(f'{script_dir}/data/smallValidation.csv', dtype=int, delimiter=',')

    X_train = train_all[:, 1:]
    y_train = train_all[:, 0]
    X_val = valid_all[:, 1:]
    y_val = valid_all[:, 0]

    return (X_train, y_train, X_val, y_val)


def load_data_medium():
    """ Load medium training and validation dataset

        Returns a tuple of length 4 with the following objects:
        X_train: An N_train-x-M ndarray containing the training data (N_train examples, M features each)
        y_train: An N_train-x-1 ndarray contraining the labels
        X_val: An N_val-x-M ndarray containing the training data (N_val examples, M features each)
        y_val: An N_val-x-1 ndarray contraining the labels
    """
    script_dir = Path(__file__).parent
    train_all = np.loadtxt(f'{script_dir}/data/mediumTrain.csv', dtype=int, delimiter=',')
    valid_all = np.loadtxt(f'{script_dir}/data/mediumValidation.csv', dtype=int, delimiter=',')

    X_train = train_all[:, 1:]
    y_train = train_all[:, 0]
    X_val = valid_all[:, 1:]
    y_val = valid_all[:, 0]

    return (X_train, y_train, X_val, y_val)


def load_data_large():
    """ Load large training and validation dataset

        Returns a tuple of length 4 with the following objects:
        X_train: An N_train-x-M ndarray containing the training data (N_train examples, M features each)
        y_train: An N_train-x-1 ndarray contraining the labels
        X_val: An N_val-x-M ndarray containing the training data (N_val examples, M features each)
        y_val: An N_val-x-1 ndarray contraining the labels
    """
    script_dir = Path(__file__).parent
    train_all = np.loadtxt(f'{script_dir}/data/largeTrain.csv', dtype=int, delimiter=',')
    valid_all = np.loadtxt(f'{script_dir}/data/largeValidation.csv', dtype=int, delimiter=',')

    X_train = train_all[:, 1:]
    y_train = train_all[:, 0]
    X_val = valid_all[:, 1:]
    y_val = valid_all[:, 0]

    return (X_train, y_train, X_val, y_val)


def linearForward(input, p):
    """
    :param input: input vector (column vector) WITH bias feature added
    :param p: parameter matrix (alpha/beta) WITH bias parameter added
    :return: output vector
    """
    assert input.shape[0] == p.shape[1]
    output = np.dot(p, input)
    return output


def sigmoidForward(a):
    """
    :param a: input vector WITH bias feature added
    """
    sigmoid_output = 1 / (1 + np.exp(-a))
    return sigmoid_output


def softmaxForward(b):
    """
    :param b: input vector WITH bias feature added
    """
    e_b = np.exp(b - np.max(b, axis=0))
    return e_b / np.sum(e_b, axis=0, keepdims=True)


def crossEntropyForward(hot_y, y_hat):
    """
    :param hot_y: 1-hot vector for true label
    :param y_hat: vector of probabilistic distribution for predicted label
    :return: float
    """
    return -np.sum(hot_y * np.log(y_hat + 1e-9)) / hot_y.shape[1]


def NNForward(x, y, alpha, beta):
    """
    Assumes biases are already included in `x` and does not add them again.
    :param x: input data WITH bias feature already added.
    :param y: input (true) labels.
    :param alpha: alpha matrix WITH bias parameter included.
    :param beta: beta matrix WITH bias parameter included.
    :return: Intermediate steps and cross-entropy loss.
    """
    # Ensure x is 2D
    #x = np.atleast_2d(x)
    
    # One-hot encoding of y
    y_one_hot = np.zeros((beta.shape[0],))
    y_one_hot[y] = 1

    # Apply linear transformation (first layer)
    a = linearForward(x, alpha)

    # Apply sigmoid activation (hidden layer output)
    z = sigmoidForward(a)
    
    # Add the bias term to z. Since the shape of beta is (10,4), and the first column of beta
    # represents the weights for the bias term, we add a row of ones to z to account for the bias.
    z_with_bias = np.vstack((np.ones((1, z.shape[1])), z))
    
    # Apply linear transformation (second layer)
    b = linearForward(z_with_bias, beta)

    # Apply softmax to get probabilities (output layer)
    y_hat = softmaxForward(b)

    # Compute the cross-entropy loss
    J = crossEntropyForward(y_one_hot, y_hat)

    # Return all intermediate quantities and the loss
    return x, a, z_with_bias, b, y_hat, J


def softmaxBackward(hot_y, y_hat):
    """
    :param hot_y: 1-hot vector for true label
    :param y_hat: vector of probabilistic distribution for predicted label
    """
    output = y_hat - hot_y
    return output


def linearBackward(prev, p, grad_curr):
    """
    Compute the gradient w.r.t. the parameters of a linear layer.
    :param prev: activations from the previous layer (with bias added as the first row).
    :param p: weights of the current layer (with bias weights as the first column).
    :param grad_curr: gradient of the loss w.r.t. the current layer's output.
    :return: 
        - grad_param: gradient of the loss w.r.t. the current layer's weights.
        - grad_prev: gradient of the loss w.r.t. the previous layer's output.
    """
    grad_param = np.dot(grad_curr, prev.T)
    # Exclude the bias gradient from the previous layer
    grad_prev = np.dot(p[:, 1:].T, grad_curr)
    return grad_param, grad_prev

def sigmoidBackward(curr, grad_curr):
    """
    Compute the gradient w.r.t. the input of the sigmoid function.
    :param curr: output of the sigmoid function.
    :param grad_curr: gradient of the loss w.r.t. the current layer's output.
    :return: grad_prev: gradient of the loss w.r.t. the sigmoid input.
    """
    # Note: curr should not include the bias.
    sigmoid_derivative = curr * (1 - curr)
    grad_prev = sigmoid_derivative * grad_curr
    return grad_prev

def NNBackward(x, y, alpha, beta, z, y_hat):
    """
    Perform the backward pass.
    """
    # One-hot encoding for y
    
    y_one_hot = np.zeros_like(y_hat)
    y_one_hot[y, np.arange(y_hat.shape[1])] = 1
    
    # Softmax and cross-entropy gradient
    g_b = softmaxBackward(y_one_hot, y_hat)
    
    # Gradients for beta and activation from sigmoid layer
    grad_beta, g_z = linearBackward(z, beta, g_b)
    
    # Sigmoid layer does not include bias term in gradient calculation
    # We only need the derivative for the non-bias terms
    g_a = sigmoidBackward(z[1:], g_z)
    
    # Gradients for alpha and input layer
    # The input x already includes a bias term, so no need to modify
    grad_alpha, _ = linearBackward(x, alpha, g_a)
    
    return grad_alpha, grad_beta, g_b, g_z, g_a


def initialize_weights(input_size, output_size, init_flag):
    if init_flag:
        # Initialize weights to random values in Uniform[-0.1, 0.1]
        weights = np.random.uniform(-0.1, 0.1, (output_size, input_size))
    else:
        # Initialize weights to zeros
        weights = np.zeros((output_size, input_size))
    # Initialize bias to zero
    bias = np.zeros((output_size, 1))
    return np.hstack((bias, weights))

def SGD(tr_x, tr_y, valid_x, valid_y, hidden_units, num_epoch, init_flag, learning_rate):
    """
    Train a neural network using stochastic gradient descent.
    """
    # Initialize weights
    alpha = initialize_weights(tr_x.shape[1] + 1, hidden_units, init_flag)
    beta = initialize_weights(hidden_units + 1, len(np.unique(tr_y)), init_flag)

    train_entropy = []  # List to store training cross-entropy loss per epoch
    valid_entropy = []  # List to store validation cross-entropy loss per epoch

    for epoch in range(num_epoch):
        # Variables to accumulate loss over the epoch
        epoch_train_loss = 0
        epoch_valid_loss = 0
        
        # Loop over each training example
        for i in range(tr_x.shape[0]):
            # Forward pass
            # NNForward should return the activation 'a', the output 'z', the predicted 'y_hat', and the loss 'J'
            x, y = tr_x[i, :], tr_y[i]
            a, z, y_hat, J = NNForward(x, y, alpha, beta)
            epoch_train_loss += J  # Accumulate loss
            
            # Backward pass to compute gradients
            grad_alpha, grad_beta = NNBackward(x, y, alpha, beta, z, y_hat)
            
            # Update parameters
            alpha -= learning_rate * grad_alpha
            beta -= learning_rate * grad_beta
        
        # Store the mean training loss for this epoch
        train_entropy.append(epoch_train_loss / tr_x.shape[0])
        
        # Forward pass on the entire validation set to compute validation loss
        for i in range(valid_x.shape[0]):
            x, y = valid_x[i, :], valid_y[i]
            _, _, _, J_val = NNForward(x, y, alpha, beta)
            epoch_valid_loss += J_val
        
        # Store the mean validation loss for this epoch
        valid_entropy.append(epoch_valid_loss / valid_x.shape[0])
        
    return alpha, beta, train_entropy, valid_entropy

# Helper functions NNForward, NNBackward, and initialize_weights need to be defined


# Helper function to initialize weights
def initialize_weights(input_dim, output_dim, init_flag):
    if init_flag:
        # Initialize weights to random values in Uniform[-0.1, 0.1]
        weights = np.random.uniform(-0.1, 0.1, (output_dim, input_dim + 1))  # +1 for the bias term
    else:
        # Initialize weights to zeros
        weights = np.zeros((output_dim, input_dim + 1))  # +1 for the bias term
    return weights

def prediction(tr_x, tr_y, valid_x, valid_y, tr_alpha, tr_beta):
    """
    :param tr_x: Training data input (size N_train x M)
    :param tr_y: Training labels (size N_train x 1)
    :param valid_x: Validation data input (size N_valid x M)
    :param valid_y: Validation labels (size N-valid x 1)
    :param tr_alpha: Alpha weights WITH bias
    :param tr_beta: Beta weights WITH bias
    :return:
        - train_error: training error rate (float)
        - valid_error: validation error rate (float)
        - y_hat_train: predicted labels for training data
        - y_hat_valid: predicted labels for validation data
    """
    def forward_pass(x, alpha, beta):
        # Add bias to input data
        x_with_bias = np.hstack([np.ones((x.shape[0], 1)), x])
        # First linear step
        a = np.dot(x_with_bias, alpha.T)
        # Sigmoid activation
        z = 1 / (1 + np.exp(-a))
        # Add bias to hidden layer
        z_with_bias = np.hstack([np.ones((z.shape[0], 1)), z])
        # Second linear step
        b = np.dot(z_with_bias, beta.T)
        # Softmax
        y_hat = np.exp(b) / np.sum(np.exp(b), axis=1, keepdims=True)
        return y_hat
    
    # Forward pass for training and validation sets
    y_hat_train = forward_pass(tr_x, tr_alpha, tr_beta)
    y_hat_valid = forward_pass(valid_x, tr_alpha, tr_beta)

    # Convert probabilities to predicted labels
    predictions_train = np.argmax(y_hat_train, axis=1)
    predictions_valid = np.argmax(y_hat_valid, axis=1)

    # Calculate error rates
    train_error = np.mean(predictions_train != tr_y)
    valid_error = np.mean(predictions_valid != valid_y)

    return train_error, valid_error, predictions_train, predictions_valid

### FEEL FREE TO WRITE ANY HELPER FUNCTIONS

def train_and_valid(X_train, y_train, X_val, y_val, num_epoch, num_hidden, init_rand, learning_rate):
    """ Main function to train and validate your neural network implementation.

        X_train: Training input in N_train-x-M numpy nd array. Each value is binary, in {0,1}.
        y_train: Training labels in N_train-x-1 numpy nd array. Each value is in {0,1,...,K-1},
            where K is the number of classes.
        X_val: Validation input in N_val-x-M numpy nd array. Each value is binary, in {0,1}.
        y_val: Validation labels in N_val-x-1 numpy nd array. Each value is in {0,1,...,K-1},
            where K is the number of classes.
        num_epoch: Positive integer representing the number of epochs to train (i.e. number of
            loops through the training data).
        num_hidden: Positive integer representing the number of hidden units.
        init_flag: Boolean value of True/False
        - True: Initialize weights to random values in Uniform[-0.1, 0.1], bias to 0
        - False: Initialize weights and bias to 0
        learning_rate: Float value specifying the learning rate for SGD.

        RETURNS: a tuple of the following six objects, in order:
        loss_per_epoch_train (length num_epochs): A list of float values containing the mean cross entropy on training data after each SGD epoch
        loss_per_epoch_val (length num_epochs): A list of float values containing the mean cross entropy on validation data after each SGD epoch
        err_train: Float value containing the training error after training (equivalent to 1.0 - accuracy rate)
        err_val: Float value containing the validation error after training (equivalent to 1.0 - accuracy rate)
        y_hat_train: A list of integers representing the predicted labels for training data
        y_hat_val: A list of integers representing the predicted labels for validation data
    """
    ### YOUR CODE HERE

    alpha = initialize_weights(X_train.shape[1] + 1, num_hidden, init_rand)
    beta = initialize_weights(num_hidden + 1, len(np.unique(y_train)), init_rand)

    train_entropy = []
    valid_entropy = []

    for epoch in range(num_epoch):
        for i in range(X_train.shape[0]):
            # Forward and backward passes
            # Update weights (alpha, beta)
            # This part requires implementations of NNForward, NNBackward, and weight update logic
            pass  # Implement the logic as per SGD function

        # After each epoch, compute and store the mean cross-entropy over the entire training and validation sets
        # This requires a full pass of forward propagation for all training and validation examples

    # After training, use the prediction function to get predictions and error rates
    train_error, valid_error, y_hat_train, y_hat_valid = prediction(X_train, y_train, X_val, y_val, alpha, beta)

    # Return the required information
    return train_entropy, valid_entropy, train_error, valid_error, y_hat_train, y_hat_valid