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
    output = np.dot(p, input)
    if output.ndim == 1:
        output = output.reshape(-1, 1)
    return output


def sigmoidForward(a):
    """
    :param a: input vector WITH bias feature added
    """
    output = 1 / (1 + np.exp(-a))
    if output.ndim == 1:
        output = output.reshape(-1, 1)
    return output


def softmaxForward(b):
    """
    :param b: input vector WITH bias feature added
    """
    e_b = np.exp(b - np.max(b, axis=0, keepdims=True))
    return e_b / np.sum(e_b, axis=0, keepdims=True)


def crossEntropyForward(hot_y, y_hat):
    """
    :param hot_y: 1-hot vector for true label
    :param y_hat: vector of probabilistic distribution for predicted label
    :return: float
    """
    y_hat = np.clip(y_hat, 1e-9, 1 - 1e-9)
    cross_entropy = -np.sum(hot_y * np.log(y_hat))  # Removed normalization by number of classes
    return cross_entropy

def NNForward(x, y, alpha, beta):
    """
    :param x: input data (column vector) WITH bias feature added
    :param y: input (true) labels
    :param alpha: alpha WITH bias parameter added
    :param beta: alpha WITH bias parameter added
    :return: all intermediate quantities x, a, z, b, y, J #refer to writeup for details
    TIP: Check on your dimensions. Did you make sure all bias features are added?
    """
    #  Convert y to one-hot encoding
    # a = # Apply linear transformation
    # z = # Apply sigmoid activation

    # Add bias term to hidden layer output before passing to output layer
    # z_with_bias

    # b = # Forward Pass through output layer using linearForward with augmented z
    # y_hat = # Apply softmax to get probabilities

    # Compute the cross-entropy loss
    # J = 
    
    # return x, a, z_with_bias, b, y_hat, J
    if np.ndim(y) == 0:
        y_one_hot = np.zeros((beta.shape[0], 1))
        y_one_hot[y] = 1
    else:
        y_one_hot = y

    a = linearForward(x, alpha)
    z = sigmoidForward(a)
    z_with_bias = np.vstack((np.ones((1, z.shape[1])), z))  # Add bias term
    b = linearForward(z_with_bias, beta)
    y_hat = softmaxForward(b)
    J = crossEntropyForward(y_one_hot, y_hat)

    return x, a, z_with_bias, b, y_hat, J



def softmaxBackward(hot_y, y_hat):
    """
    :param hot_y: 1-hot vector for true label
    :param y_hat: vector of probabilistic distribution for predicted label
    """
    return y_hat - hot_y


def linearBackward(prev, p, grad_curr):
    """
    :param prev: previous layer WITH bias feature
    :param p: parameter matrix (alpha/beta) WITH bias parameter
    :param grad_curr: gradients for current layer
    :return:
        - grad_param: gradients for parameter matrix (alpha/beta)
        - grad_prevl: gradients for previous layer
    TIP: Check your dimensions.
    """
    grad_param = np.dot(grad_curr, prev.T)
    # Exclude the bias gradient from the previous layer
    grad_prev = np.dot(p[:, 1:].T, grad_curr)
    return grad_param, grad_prev


def sigmoidBackward(curr, grad_curr):
    """
    :param curr: current layer WITH bias feature
    :param grad_curr: gradients for current layer
    :return: grad_prevl: gradients for previous layer
    TIP: Check your dimensions
    """
    sigmoid_derivative = curr * (1 - curr)
    grad_prev = grad_curr * sigmoid_derivative
    return grad_prev


def NNBackward(x, y, alpha, beta, z, y_hat):
    """
    :param x: input data (column vector) WITH bias feature added
    :param y: input (true) labels
    :param alpha: alpha WITH bias parameter added
    :param beta: alpha WITH bias parameter added
    :param z: z as per writeup
    :param y_hat: vector of probabilistic distribution for predicted label
    :return:
        - grad_alpha: gradients for alpha
        - grad_beta: gradients for beta
        - g_b: gradients for layer b (softmaxBackward)
        - g_z: gradients for layer z (linearBackward)
        - g_a: gradients for layer a (sigmoidBackward)
    """
    # Convert y to one-hot encoding
    # y_one_hot =
    
    # Gradient of Cross Entropy Loss w.r.t. y_hat
    # g_y_hat =
    
    # Gradient of Loss w.r.t. beta (Weights from hidden to output layer)
    # grad_beta, g_b = 
    
    # Gradient of Loss w.r.t. activation before sigmoid (a)
    # g_a =
    
    # Gradient of Loss w.r.t. alpha (Weights from input to hidden layer)
    # grad_alpha, g_x =
    
    # return grad_alpha, grad_beta, g_y_hat, g_b_no_bias, g_a
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

def SGD(tr_x, tr_y, valid_x, valid_y, hidden_units, num_epoch, init_flag, learning_rate):
    N_train, M = tr_x.shape
    num_classes = len(np.unique(tr_y))
    
    # Add a column of ones to the training and validation data to account for the input bias term
    tr_x = np.hstack((tr_x, np.ones((N_train, 1))))
    valid_x = np.hstack((valid_x, np.ones((valid_x.shape[0], 1))))
    M += 1  # Adjust for the input bias term

    # Initialize weights
    if init_flag:
        alpha = np.random.uniform(-0.1, 0.1, (hidden_units, M))
    else:
        alpha = np.zeros((hidden_units, M))
    # Initialize beta with an additional bias unit for the hidden layer
    if init_flag:
        beta = np.random.uniform(-0.1, 0.1, (num_classes, hidden_units + 1))
    else:
        beta = np.zeros((num_classes, hidden_units + 1))

    train_entropy = []
    valid_entropy = []

    for epoch in range(num_epoch):
        epoch_train_losses = []

        for i in range(N_train):
            x = tr_x[i, :][None, :]  # Include bias input

            y = np.zeros((num_classes, 1))
            y[tr_y[i], 0] = 1  # One-hot encoding

            # Forward pass
            z = np.dot(alpha, x.T)
            a = 1 / (1 + np.exp(-z))  # Sigmoid activation
            a_bias = np.vstack((a, np.ones((1, 1))))  # Add bias unit for hidden layer output
            o = np.dot(beta, a_bias)
            y_hat = np.exp(o) / np.sum(np.exp(o), axis=0)  # Softmax

            # Compute loss
            loss = -np.sum(y * np.log(y_hat))
            epoch_train_losses.append(loss)

            # Backward pass
            d_o = y_hat - y
            d_beta = np.dot(d_o, a_bias.T)

            d_a = np.dot(beta[:, :-1].T, d_o)  # Exclude bias weight
            d_z = d_a * a * (1 - a)
            d_alpha = np.dot(d_z, x)

            # Update weights
            beta -= learning_rate * d_beta
            alpha -= learning_rate * d_alpha

        # Calculate mean loss for the epoch
        train_entropy.append(np.mean(epoch_train_losses))

        # Validation phase
        epoch_valid_losses = []
        for i in range(valid_x.shape[0]):
            x = valid_x[i, :][None, :]  # Include bias input

            y = np.zeros((num_classes, 1))
            y[valid_y[i], 0] = 1  # One-hot encoding

            # Forward pass
            z = np.dot(alpha, x.T)
            a = 1 / (1 + np.exp(-z))
            a_bias = np.vstack((a, np.ones((1, 1))))  # Add bias unit for hidden layer output
            o = np.dot(beta, a_bias)
            y_hat = np.exp(o) / np.sum(np.exp(o), axis=0)

            # Compute loss
            loss = -np.sum(y * np.log(y_hat))
            epoch_valid_losses.append(loss)

        # Calculate mean validation loss for the epoch
        valid_entropy.append(np.mean(epoch_valid_losses))

    return alpha, beta, train_entropy, valid_entropy


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
    N_train, M = X_train.shape
    num_classes = len(np.unique(np.concatenate((y_train, y_val))))
    
    # Initialize weights and biases
    W1, b1 = initialize_params(M, num_hidden, init_rand)
    W2, b2 = initialize_params(num_hidden, num_classes, init_rand)
    
    # Placeholder for the return values
    metrics = {
        'loss_per_epoch_train': [],
        'loss_per_epoch_val': [],
        'y_hat_train': [],
        'y_hat_val': [],
        'err_train': 0,
        'err_val': 0
    }

    for epoch in range(num_epoch):
        update_params(X_train, y_train, W1, b1, W2, b2, learning_rate, num_classes)
        update_metrics(X_train, y_train, W1, b1, W2, b2, metrics, 'train')
        update_metrics(X_val, y_val, W1, b1, W2, b2, metrics, 'val')

    return (
        metrics['loss_per_epoch_train'], 
        metrics['loss_per_epoch_val'], 
        metrics['err_train'], 
        metrics['err_val'], 
        metrics['y_hat_train'], 
        metrics['y_hat_val']
    )

def initialize_params(n_in, n_out, init_rand):
    if init_rand:
        return (np.random.uniform(-0.1, 0.1, (n_out, n_in)), np.zeros(n_out))
    else:
        return (np.zeros((n_out, n_in)), np.zeros(n_out))

def forward_backward_pass(x, y, W1, b1, W2, b2, num_classes):
    # Forward pass
    z1 = np.dot(W1, x) + b1
    a1 = 1 / (1 + np.exp(-z1))
    z2 = np.dot(W2, a1) + b2
    a2 = np.exp(z2 - np.max(z2))
    y_hat = a2 / np.sum(a2)
    
    # Backward pass
    y_true = np.zeros(num_classes)
    y_true[y] = 1
    d2 = y_hat - y_true
    dW2 = np.outer(d2, a1)
    db2 = d2
    da1 = np.dot(W2.T, d2)
    dz1 = da1 * a1 * (1 - a1)
    dW1 = np.outer(dz1, x)
    db1 = dz1
    
    return dW1, db1, dW2, db2, y_hat

def update_params(X, y, W1, b1, W2, b2, learning_rate, num_classes):
    for i in range(X.shape[0]):
        dW1, db1, dW2, db2, _ = forward_backward_pass(X[i], y[i], W1, b1, W2, b2, num_classes)
        W1 -= learning_rate * dW1
        b1 -= learning_rate * db1
        W2 -= learning_rate * dW2
        b2 -= learning_rate * db2

def update_metrics(X, y, W1, b1, W2, b2, metrics, dataset_type):
    loss = 0
    correct_preds = 0
    predictions = []
    for i in range(X.shape[0]):
        _, _, _, _, y_hat = forward_backward_pass(X[i], y[i], W1, b1, W2, b2, len(np.unique(y)))
        predictions.append(np.argmax(y_hat))
        loss += -np.log(y_hat[y[i]])
        correct_preds += y[i] == predictions[-1]
    
    loss /= X.shape[0]
    error = 1 - correct_preds / X.shape[0]
    metrics[f'loss_per_epoch_{dataset_type}'].append(loss)
    metrics[f'y_hat_{dataset_type}'] = predictions
    metrics[f'err_{dataset_type}'] = error
