import numpy as np
import h5py
import matplotlib.pyplot as plt

rng = np.random.default_rng()


def load_data(filename):
    with h5py.File(filename, 'r') as f:
        X = np.array(f['X']['value'][:]).T
        y = np.array(f['y']['value'][:], np.int).T
    return X, y


def prepare_Y(y):
    Y = np.zeros((len(y), y.max()+1))
    Y[np.arange(len(y)), y] = 1
    return Y.T


def g(A):
    return 1 / (1 + np.exp(-A))


def init_theta(n):
    Theta = [0.2 * (rng.random((n[l+1], n[l]+1)) - 0.5) for l in range(0, len(n)-1)]
    return Theta


def forward_propagate(Theta, X):
    L = len(Theta)+1
    A = [X.T]
    for l in range(0, L-1):
        Atemp = np.vstack((np.ones(A[l].shape[1]), A[l]))
        Anext = g(Theta[l] @ Atemp)
        A.append(Anext)
    return A


def eln(x):
    tmp = np.nan_to_num(np.log(x), nan=np.log(1e-300))
    result = np.maximum(tmp, np.log(1e-300))
    return result


def J(Theta, A, Y, lmbda):
    m = A.shape[1]
    acc = 0.0
    for theta_current in Theta:
        acc += np.sum(np.square(theta_current[:, 1:]))
    reg_term = acc * (lmbda / (2*m))
    unreg_term = (-eln(A) * Y) - (eln(1-A) * (1-Y))
    sum_term = np.sum(unreg_term) / m
    cost = sum_term + reg_term
    return cost


def back_propagate(Theta, A, Y, lmbda):
    L = len(A)
    m = Y.shape[1]
    D = [np.zeros(Theta[l].shape) for l in range(0, len(Theta))]
    Delta = [np.zeros(A[l].shape) for l in range(0, len(A))]
    Delta[-1] = A[-1] - Y
    for l in range(L-2, 0, -1):
        tCurr = Theta[l][:, 1:]
        dCost_dA = np.dot(tCurr.T, Delta[l+1])
        dA_dZ = A[l] * 1-A[l]
        Delta[l] = dCost_dA * dA_dZ
    for l in range(0, L-1):
        tempA = np.vstack((np.ones(A[l].shape[1]), A[l]))
        term = np.dot(Delta[l+1], tempA.T)
        D[l] += term
        D[l][:, 1:] += lmbda * Theta[l][:, 1:]
        D[l] /= m
    return D


def gradient_descent(Theta, X, Y, lmbda, alpa, numIter):
    costs = []
    for i in range(0, numIter):
        A = forward_propagate(Theta, X)
        costs.append(J(Theta, A[-1], Y, lmbda))
        print("Running gradient descent ({}). Cost = {}".format(i, costs[-1]))
        D = back_propagate(Theta, A, Y, lmbda)
        for l in range(0, len(Theta)):
            dif = alpa * D[l]
            Theta[l] = Theta[l] - dif
    costs.append(J(Theta, forward_propagate(Theta, X)[-1], Y, lmbda))
    print("Running gradient descent ({}). Cost = {}".format(i, costs[-1]))
    return Theta, costs


def plot_cost(costs):
    plt.plot(costs)
    plt.show()
