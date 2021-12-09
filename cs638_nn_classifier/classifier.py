from functions import *


# NOTE: This classifier doesn't yet work. I couldn't get the translation
#       from octave to python quite right, and it does't perform well.

X, y = load_data("digit_data.h5")
sel = rng.permutation(len(X))

X_test = X[sel[4000:], :]
X = X[sel[0:4000], :]
y_test = y[sel[4000:], :].flatten()-1
y = y[sel[0:4000], :].flatten()-1

m = X.shape[0]
n1 = X.shape[1]
n = [n1, 30, np.max(y)+1]
Y = prepare_Y(y)
Theta = init_theta(n)

lmbda = 1
alpa = 2
maxIter = 500

Theta, costs = gradient_descent(Theta, X, Y, lmbda, alpa, maxIter)

plot_cost(costs)

estimate = forward_propagate(Theta, X)[-1]
# print("estimate: {}".format(estimate))
pred = np.transpose(np.argmax(estimate, axis=0))
# print("argmax: {}".format(pred))

correct = np.sum(pred == y)
accuracy = correct / y.shape[0]
# print("correct: {}, yShape[1]: {}".format(correct, y.shape[0]))
print("Training accuracy: {}".format(accuracy))

est_test = forward_propagate(Theta, X_test)[-1]
# print("estimateShape: {}".format(est_test.shape))
pred_test = np.transpose(np.argmax(est_test, axis=0))
# print("argmax: {}".format(pred_test))
# print("est_test: {}".format(est_test))

correct_test = np.sum(pred_test == y_test)
accuracy_test = correct_test / y_test.shape[0]
print("Test_correct: {}, Testing accuracy: {}".format(correct_test, accuracy_test))


