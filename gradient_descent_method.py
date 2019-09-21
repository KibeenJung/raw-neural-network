import random

NUM_TRAIN_DATA = 4
NUM_WEIGHT = 3
LEARNING_RATE = 0.01

W = []
delta = [None] * NUM_WEIGHT
X = [0.0, 1.0, 1.0, 2.0]
Y = [0.0, 1.0, 2.0, 1.0]


def get_model(x):
    return (W[2] * (x ** 2)) + (W[1] * x) + W[0]


def get_partial_differentiation(x, y, mode):
    partial_diff = -2 * (y + ((-1) * get_model(x)))
    if mode == 0:
        partial_diff = partial_diff * 1
    elif mode == 1:
        partial_diff = partial_diff * x
    elif mode == 2:
        partial_diff = partial_diff * (2 * W[2] * x)
    return partial_diff


def update_gradient():
    for g_idx in range(NUM_WEIGHT):
        delta[g_idx] = 0.0
        for d_idx in range(NUM_TRAIN_DATA):
            delta[g_idx] += get_partial_differentiation(x=X[d_idx], y=Y[d_idx], mode=g_idx)


def update_weight():
    for w_idx in range(NUM_WEIGHT):
        W[w_idx] = W[w_idx] - (LEARNING_RATE * delta[w_idx])


def init_weight():
    # Initialize weight based on gaussian distribution whose mean is zero, and std is 0.01
    for w_idx in range(NUM_WEIGHT):
        W.append(random.gauss(0.0, 0.1))
    print_weight('[init]')


def print_weight(prefix):
    print(prefix, end=' ')
    for w_idx in range(NUM_WEIGHT):
        print('W{}: {}'.format(w_idx, W[w_idx]), end=', ')
    print()


init_weight()

for episode in range(500):
    update_gradient()
    update_weight()
    if episode % 10 == 0:
        print_weight('[ep.{}]'.format(episode))
    episode += 1
print_weight('[final ep.{}]'.format(episode))
