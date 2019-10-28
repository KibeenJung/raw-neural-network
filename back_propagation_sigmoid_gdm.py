import random
import math
LEARNING_RATE = 0.5
N_TRAIN_DATA = 4
N_INPUT = 2
N_NODE_LAYER_1 = 100
N_NODE_LAYER_2 = 1
N_OUTPUT = N_NODE_LAYER_2
N_EPISODE = 10000000
TARGET_ERROR = 0.0001


def create_1d_array(shape):
    return [0.0] * shape


def create_2d_array(row, col):
    return [[0.0] * col for i in range(row)]


def sigmoid(net):
    return 1.0 / (1.0 + math.exp(-net))


def init_weight():
    # Layer 1
    for i in range(N_NODE_LAYER_1):
        for j in range(N_INPUT):
            weight_1[i][j] = random.gauss(0.0, 0.5)

    for i in range(N_NODE_LAYER_1):
        bias_1[i] = random.gauss(0.0, 0.5)

    # Layer 2
    for i in range(N_OUTPUT):
        for j in range(N_NODE_LAYER_1):
            weight_2[i][j] = random.gauss(0.0, 0.5)

    for i in range(N_OUTPUT):
        bias_2[i] = random.gauss(0.0, 0.5)


def print_weight():
    print('--------------- weight --------------')
    print('[Layer 1]')
    for i in range(N_NODE_LAYER_1):
        for j in range(N_INPUT):
            print('weight_{0}{1}: {2}'.format(i, j, weight_1[i][j]), end='')
            if j != (N_INPUT - 1):
                print(', ', end='')
            else:
                print('')

    for i in range(N_NODE_LAYER_1):
        print('bias_{0}: {1}'.format(i, bias_1[i]), end='')
        if i != (N_NODE_LAYER_1 - 1):
            print(', ', end='')
        else:
            print('')

    print('[Layer 2]')
    for i in range(N_OUTPUT):
        for j in range(N_NODE_LAYER_1):
            print('weight_{0}{1}: {2}'.format(i, j, weight_2[i][j]), end='')
            if j != (N_NODE_LAYER_1 - 1):
                print(', ', end='')
            else:
                print('')

    for i in range(N_OUTPUT):
        print('bias_{0}: {1}'.format(i, bias_2[i]), end='')
        if i != (N_OUTPUT - 1):
            print(', ', end='')
        else:
            print('')
    print('')


def reset_gradient():
    # Layer 1
    for i in range(N_NODE_LAYER_1):
        for j in range(N_INPUT):
            gradient_weight_1[i][j] = 0.0
    for i in range(N_NODE_LAYER_1):
        gradient_bias_1[i] = 0.0
    # Layer 2
    for i in range(N_OUTPUT):
        for j in range(N_NODE_LAYER_1):
            gradient_weight_2[i][j] = 0.0
    for i in range(N_OUTPUT):
        gradient_bias_2[i] = 0.0


def forward(x):
    # Layer 1
    for i in range(N_NODE_LAYER_1):
        net_1 = 0.0
        for j in range(N_INPUT):
            net_1 += weight_1[i][j] * x[j]
        net_1 += bias_1[i]
        hidden_1[i] = sigmoid(net_1)

    # Layer 2
    for i in range(N_OUTPUT):
        net_2 = 0.0
        for j in range(N_NODE_LAYER_1):
            net_2 += weight_2[i][j] * hidden_1[j]
        net_2 += bias_2[i]
        output[i] = sigmoid(net_2)


def calculate_error(y):
    error = 0.0
    for i in range(N_OUTPUT):
        error += (y[i] - output[i]) ** 2
    return error / 2.0


def backward(x, y):
    # Layer 2
    for i in range(N_OUTPUT):
        for j in range(N_NODE_LAYER_1):
            gradient_weight_2[i][j] += -output[i] * (y[i] - output[i]) * (1 - output[i]) * hidden_1[j]

    for i in range(N_OUTPUT):
        gradient_bias_2[i] += -output[i] * (y[i] - output[i]) * (1 - output[i])

    # Layer 1
    for i in range(N_NODE_LAYER_1):
        for j in range(N_INPUT):
            temp = 0.0
            for k in range(N_OUTPUT):
                temp += gradient_weight_2[k][i] * (y[k] - output[k]) * output[k] * (1 - output[k])
            gradient_weight_1[i][j] += -x[j] * hidden_1[i] * (1 - hidden_1[i]) * temp

    for i in range(N_NODE_LAYER_1):
        temp = 0.0
        for k in range(N_OUTPUT):
            temp += gradient_weight_2[k][i] * (y[k] - output[k]) * output[k] * (1 - output[k])
        gradient_bias_1[i] += -hidden_1[i] * (1 - hidden_1[i]) * temp


def update_weight():
    # Layer 1
    for i in range(N_NODE_LAYER_1):
        for j in range(N_INPUT):
            weight_1[i][j] -= LEARNING_RATE * gradient_weight_1[i][j]

    for i in range(N_NODE_LAYER_1):
        bias_1[i] -= LEARNING_RATE * gradient_bias_1[i]

    # Layer 2
    for i in range(N_OUTPUT):
        for j in range(N_NODE_LAYER_1):
            weight_2[i][j] -= LEARNING_RATE * gradient_weight_2[i][j]

    for i in range(N_OUTPUT):
        bias_2[i] -= LEARNING_RATE * gradient_bias_2[i]


def early_stopping(error, epoch=20):
    if error > early_stopping.previous_error:
        early_stopping.n_degradation += 1
    else:
        early_stopping.n_degradation = 0

    early_stopping.previous_error = error

    if early_stopping.n_degradation == epoch:
        return True
    else:
        return False


early_stopping.previous_error = 0.0
early_stopping.n_degradation = 0


def print_answer():
    print('----------- XOR Table ----------')
    print('x_n1  x_n2  y_n1  o_n1')
    for d in range(N_TRAIN_DATA):
        forward(X[d])
        print('{0:4d}  {1:4d}  {2:4d}  {3}'.format(int(X[d][0]), int(X[d][1]), int(Y[d][0]), output[0]))


X = [[1.0, 1.0], [1.0, 0.0], [0.0, 1.0], [0.0, 0.0]]
Y = [[0.0], [1.0], [1.0], [0.0]]

weight_1 = create_2d_array(row=N_NODE_LAYER_1, col=N_INPUT)
bias_1 = create_1d_array(N_NODE_LAYER_1)
gradient_weight_1 = create_2d_array(row=N_NODE_LAYER_1, col=N_INPUT)
gradient_bias_1 = create_1d_array(N_NODE_LAYER_1)
hidden_1 = create_1d_array(N_NODE_LAYER_1)

weight_2 = create_2d_array(row=N_OUTPUT, col=N_NODE_LAYER_1)
bias_2 = create_1d_array(N_OUTPUT)
gradient_weight_2 = create_2d_array(row=N_OUTPUT, col=N_NODE_LAYER_1)
gradient_bias_2 = create_1d_array(N_OUTPUT)
output = create_1d_array(N_OUTPUT)

init_weight()

exit_cond = False

for epoch in range(N_EPISODE):
    reset_gradient()
    error = 0.0
    for d in range(N_TRAIN_DATA):
        forward(X[d])
        error += calculate_error(Y[d])
        if early_stopping(error) or error < TARGET_ERROR:
            exit_cond = True
        backward(X[d], Y[d])
    update_weight()
    if epoch % 100 == 0:
        print(epoch, error)
    if exit_cond:
        print(epoch, error)
        break
    epoch += 1

print_weight()
print_answer()
