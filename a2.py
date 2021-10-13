# I tried to make this as general as possible to work with 
#   vector labels, more hidden layers, variable sizes, etc
# Also tried to squeeze some efficiency with the matrix representations


# General description:
#   goal: move towards a matrix represenation of neural nets
#   forward propagation
#     - generalize each list outputs in the standard neural net code as a vector
#     - weights now behave as matrices and matrix multiplation supplants
#       inner products between previous levels
#     - the input layer can be seen as output level 0 (y[0])
#     - forward propagation now consists of matrix operations and applying
#       activation function on vector outputs
#
#   assemble "tensors" combining each layer's weights into an indexible collection
#       (for example the input is y[0], output is y[k], the i'th W matrix is W[i])
#
#   backpropagation
#     - track delta values in vectors in the delta tensor
#     - delta[k] is the base case, calculate separately
#     - generally, a delta value is the sum of all above units' weight connection
#       to the current unit, times that above unit's delta value, times the current
#       unit's z value passed through the derivative of the activation function:
#           delta[i] = SUM_(all j above) (delta[j] * W[j][i] * s'(z[i]))
#     - with some matrix fiddling, this can be rewritten as the matrix multiplication
#       between the transpose of the weight matrix times the delta[j] vector times
#       the derivative of the activation of z[i]:
#           delta[i] = (W[j]T * delta[j]) o s'(z[i])
#                                         ^- elem-wise multiplication
import numpy as np

# [-1, +1] -> boolean
# determine which of two outputs is larger
def sign(x):
    if x < 0:
        return -1
    return 1

# tanh(z/2)
def sigmoid(z):
    return np.tanh(z * 0.5)

# derivative of sigmoid
def sigmoid_prime(z):
    sig_z = sigmoid(z)
    return (1 - (sig_z * sig_z))/2


# supply data/label paths
def read_data(data):
    # read info
    file = open(data)
    points = []

    for line in file:
        # parse the information as doubles into np.arrays 
        point = np.array(line.strip().split(' '), dtype=np.float64)
        points.append(point)
    
    return np.array(points)


# tensors for tracking matrices and vectors
def create_tensors(layers):
    # weight, gradient matrices
    W = [np.array([])] * len(layers)

    # populate each layer of calculations so they can be indexed 
    for i in range(1, len(layers)):
        # create weight matrix
        # add room for a bias dimension in the input vector        -v
        weights = np.random.rand(layers[i], layers[i - 1] + 1)

        # adjust weights to be centered around zero and normalized by input size
        weights = (weights - 0.5)/(layers[i])
        W[i] = weights

    # initialize gradient to zero
    grad = [w * 0 for w in W]

    # z, y, delta vectors
    z = [np.array([])] * len(layers)
    y = [np.array([])] * len(layers)
    delta = [np.array([])] * len(layers)

    return W, y, z, delta, grad


# feed forward propagation
def propagate(W, y, z, k):
    # look at layer j=[1..k]
    for j in range(1, k):
        # vector of dot products b/w weights and respective inputs
        z[j] = np.matmul(W[j], y[j-1])
        # run the above vector through the activation function
        if j != k - 1:
            # add bias if not output
            y[j] = np.concatenate((sigmoid(z[j]), [1]))
        else:
            # no bias if output
            y[j] = sigmoid(z[j])

# backpropagation
def backprop(W, y, z, delta, grad, layers, learning_rate=0.1, batch_size=1):
    # go backwards j=[k-1..1]
    for j in range(len(layers) - 1, 0, -1):
        # delta_j = partial(error)/partial(z_j)
        if j == len(layers) - 1:
            # base case for delta[k]
            # take the partial derivative of error w.r.t. z[j]
            delta[j] = (label - y[j]) * sigmoid_prime(z[j])
        else:
            # multiply the delta vector by the transpose of the weight matrix
            # this results in a vector of dot products between deltas of above units
            #   and the weights connecting them (the first half of the standard summation)
            # element-wise multiply that vector with the z[j] values passed through 
            #   the derivative of our sigmoid
            # we also need to drop the last row of the vector resulting from the multiplication
            #   because it corresponds to a bias and not a z value
            delta[j] = np.matmul(W[j+1].T, delta[j+1])[:-1] * sigmoid_prime(z[j])

        # creating a new matrix representing the gradient of error w.r.t. W[j]
        g = []
        for r in range(layers[j]):
            # multiple the value of delta[j] at each row by the input vector y[j-1]
            row = (learning_rate * delta[j][r]) * y[j-1]
            g.append(row)

        # add sample's gradient to the batch gradient
        grad[j] += np.array(g)

        # correct the matrix by the gradient
        if i % batch_size == batch_size - 1:
            W[j] = W[j] + grad[j]
            # reset the gradient after batch
            grad[j] = W[j] * 0


## data
# read training data
xs, ys = read_data('train/data.txt'), read_data('train/labels.txt')
# read testing data
test_xs, test_ys = read_data('test/data.txt'), read_data('test/labels.txt')

# xs = np.concatenate((xs, test_xs))
# ys = np.concatenate((ys, test_ys))

# maximum value of training data
max_val = 1 / max(abs(xs.max()), abs(xs.min()))


## model setup
# layer shape: 1000 inputs, 10 hidden nodes, 2 output
layers = [1000, 10, 1]
W, z, y, delta, grad = create_tensors(layers)


## training
# concat points to train over multiple times
xs = np.concatenate(([xs] * 8))
ys = np.concatenate(([ys] * 8))

# loop over all points
correct = []
for i, sample in enumerate(xs):
    # set the target label
    label = ys[i]

    # the 0th layer's outputs is the sample
    #   normalize by 1/max_value of data
    y[0] = np.concatenate((sample * max_val, [1]))

    propagate(W, y, z, len(layers))

    # output
    # print(y[-1], sign(y[k]), sign(label), label)
    if sign(y[-1]) == sign(label):
        correct.append(1)
    else:
        correct.append(0)

    # backpropagation
    backprop(W, y, z, delta, grad, layers, learning_rate=0.1, batch_size=10)

print('Training:')
print(f'    #correct:               {sum(correct)}')
print(f'    #trained:               {len(correct)}')
print(f'    %correct (full sample): {100*sum(correct)/len(correct)}%')
amt = 1000
print(f'    %correct (last {amt}):   {100*sum(correct[-amt:])/len(correct[-amt:])}%')


## testing
correct = []
for i, sample in enumerate(test_xs):
    label = test_ys[i]

    # the 0th layer's outputs is the sample
    y[0] = np.concatenate((sample * max_val, [1]))

    propagate(W, y, z, len(layers))

    # output
    if sign(y[-1]) == sign(label):
        correct.append(1)
    else:
        correct.append(0)

print()
print('Testing:')
print(f'    #correct:               {sum(correct)}')
print(f'    #tested:                {len(correct)}')
print(f'    %correct (full sample): {100*sum(correct)/len(correct)}%')
