import math
import numpy
N_BATCH_SIZE = 4
N_INPUT = 2
N_NODE_LAYER_1 = 2
N_NODE_LAYER_2 = 1
N_OUTPUT = N_NODE_LAYER_2


def create_1d_array(shape):
    return [0.0] * shape


def create_2d_array(row, col):
    return [[0.0] * col for i in range(row)]


def step_function(net):
    if net >= 0:
        return 1
    else:
        return 0


def forward(X):

    # net_1
    for k in range(N_BATCH_SIZE):
        for i in range(N_NODE_LAYER_1):
            net_1[i][k] = 0.0
            for j in range(N_INPUT):
                net_1[i][k] += weight_1[i][j] * X[k][j]
            net_1[i][k] += bias_1[i]

    # batch norm.
    for i in range(N_NODE_LAYER_1):
        mean = numpy.mean(net_1[i])
        variance = numpy.var(net_1[i])
        for k in range(N_BATCH_SIZE):
            net_1[i][k] = (net_1[i][k] - mean) / math.sqrt(variance)

    for k in range(N_BATCH_SIZE):
        # Layer 1
        for i in range(N_NODE_LAYER_1):
            y_di = (gamma_1[i] * net_1[i][k]) + beta_1[i]
            hidden_1[i] = step_function(y_di)

        # Layer 2
        for i in range(N_OUTPUT):
            net_2 = 0.0
            for j in range(N_NODE_LAYER_1):
                net_2 += weight_2[i][j] * hidden_1[j]
            net_2 += bias_2[i]
            output[i] = step_function(net_2)
        print('X: {}, Y^: {}'.format(X[k], output[0]))


X = [[1.0, -1.0], [2.0, 0.0], [-2.0, 3.0], [1.0, 1.0]]
Y = [[1.0], [1.0], [0.0], [0.0]]

weight_1 = create_2d_array(row=N_NODE_LAYER_1, col=N_INPUT)
weight_1[0][0] = 0.4
weight_1[0][1] = 0.4
weight_1[1][0] = 0.5
weight_1[1][1] = 0.5
bias_1 = create_1d_array(N_NODE_LAYER_1)
bias_1[0] = 0.4
bias_1[1] = 1.0
gamma_1 = create_1d_array(N_NODE_LAYER_1)
gamma_1[0] = 1.0
gamma_1[1] = -0.5
beta_1 = create_1d_array(N_NODE_LAYER_1)
beta_1[0] = -0.5
beta_1[1] = -1.0
net_1 = create_2d_array(N_NODE_LAYER_1, N_BATCH_SIZE)
hidden_1 = create_1d_array(N_NODE_LAYER_1)

weight_2 = create_2d_array(row=N_OUTPUT, col=N_NODE_LAYER_1)
weight_2[0][0] = 1.0
weight_2[0][1] = 1.0
bias_2 = create_1d_array(N_OUTPUT)
bias_2[0] = -1.0
output = create_1d_array(N_OUTPUT)

forward(X)
