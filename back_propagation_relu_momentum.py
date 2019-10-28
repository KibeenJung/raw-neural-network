LEARNING_RATE = 1.0
MOMENTUM_RATE = 1.0

N_TRAIN_DATA = 1
N_INPUT = 2
N_NODE_LAYER_1 = 1
N_NODE_LAYER_2 = 1
N_OUTPUT = N_NODE_LAYER_2
N_EPISODE = 2


def create_1d_array(shape):
    return [0.0] * shape


def create_2d_array(row, col):
    return [[0.0] * col for i in range(row)]


def ReLU(net):
    return max(0, net)


def init_weight():
    # Layer 1
    for i in range(N_NODE_LAYER_1):
        for j in range(N_INPUT):
            weight_1[i][j] = 1.0
    # Layer 2
    for i in range(N_OUTPUT):
        for j in range(N_NODE_LAYER_1):
            weight_2[i][j] = 1.0


def print_weight():
    print('[Layer 1]', end='')
    for i in range(N_NODE_LAYER_1):
        for j in range(N_INPUT):
            print('weight_{0}{1}: {2}'.format(i, j, weight_1[i][j]), end='')
            if j != (N_INPUT - 1):
                print(', ', end='')
            else:
                print('')
    print('[Layer 2]', end='')
    for i in range(N_OUTPUT):
        for j in range(N_NODE_LAYER_1):
            print('weight_{0}{1}: {2}'.format(i, j, weight_2[i][j]), end='')
            if j != (N_NODE_LAYER_1 - 1):
                print(', ', end='')
            else:
                print('')
    print('')


def reset_gradient():
    # Layer 1
    for i in range(N_NODE_LAYER_1):
        for j in range(N_INPUT):
            gradient_weight_1[i][j] = 0.0
    # Layer 2
    for i in range(N_OUTPUT):
        for j in range(N_NODE_LAYER_1):
            gradient_weight_2[i][j] = 0.0


def reset_momentum():
    # Layer 1
    for i in range(N_NODE_LAYER_1):
        for j in range(N_INPUT):
            momentum_weight_1[i][j] = 0.0
    # Layer 2
    for i in range(N_OUTPUT):
        for j in range(N_NODE_LAYER_1):
            momentum_weight_2[i][j] = 0.0


def calculate_error(y):
    error = 0.0
    for i in range(N_OUTPUT):
        error += (y[i] - output[i]) ** 2
    return error / 2.0


def backward(x, y):
    # Layer 2
    for i in range(N_OUTPUT):
        for j in range(N_NODE_LAYER_1):
            if net_2[i] > 0:
                gradient_weight_2[i][j] = -hidden_1[j] * (y[i] - output[i])
            else:
                gradient_weight_2[i][j] = 0
    # Layer 1
    for i in range(N_NODE_LAYER_1):
        for j in range(N_INPUT):
            temp = 0.0
            for k in range(N_OUTPUT):
                if net_2[k] > 0:
                    temp += -weight_2[k][i] * (y[k] - output[k])
                else:
                    temp += 0
            if net_1[i] > 0:
                gradient_weight_1[i][j] = temp * x[j]
            else:
                gradient_weight_1[i][j] = 0


def update_weight():
    # Layer 1
    for i in range(N_NODE_LAYER_1):
        for j in range(N_INPUT):
            momentum_weight_1[i][j] = MOMENTUM_RATE * momentum_weight_1[i][j] + \
                                      LEARNING_RATE * gradient_weight_1[i][j]
            weight_1[i][j] -= momentum_weight_1[i][j]
    # Layer 2
    for i in range(N_OUTPUT):
        for j in range(N_NODE_LAYER_1):
            momentum_weight_2[i][j] = MOMENTUM_RATE * momentum_weight_2[i][j] + \
                                      LEARNING_RATE * gradient_weight_2[i][j]
            weight_2[i][j] -= momentum_weight_2[i][j]


def forward(x):
    # Layer 1
    for i in range(N_NODE_LAYER_1):
        net = 0.0
        for j in range(N_INPUT):
            net += weight_1[i][j] * x[j]
        net_1[i] = net
        hidden_1[i] = ReLU(net)
    # Layer 2
    for i in range(N_OUTPUT):
        net = 0.0
        for j in range(N_NODE_LAYER_1):
            net += weight_2[i][j] * hidden_1[j]
        net_2[i] = net
        output[i] = ReLU(net)


X = [[1.0, -0.5]]
Y = [[0.0]]

weight_1 = create_2d_array(row=N_NODE_LAYER_1, col=N_INPUT)
gradient_weight_1 = create_2d_array(row=N_NODE_LAYER_1, col=N_INPUT)
momentum_weight_1 = create_2d_array(row=N_NODE_LAYER_1, col=N_INPUT)
net_1 = create_1d_array(N_NODE_LAYER_1)
hidden_1 = create_1d_array(N_NODE_LAYER_1)

weight_2 = create_2d_array(row=N_OUTPUT, col=N_NODE_LAYER_1)
gradient_weight_2 = create_2d_array(row=N_OUTPUT, col=N_NODE_LAYER_1)
momentum_weight_2 = create_2d_array(row=N_OUTPUT, col=N_NODE_LAYER_1)
net_2 = create_1d_array(N_OUTPUT)
output = create_1d_array(N_OUTPUT)

init_weight()
reset_momentum()

for epoch in range(N_EPISODE):
    reset_gradient()
    error = 0.0
    for d in range(N_TRAIN_DATA):
        forward(X[d])
        error += calculate_error(Y[d])
        backward(X[d], Y[d])
    update_weight()

    print('[epoch {}] error: {}'.format(epoch + 1, error))
    print_weight()
    epoch += 1

