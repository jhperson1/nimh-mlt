from sklearn.linear_model import LogisticRegression
import numpy as np

ET = "random"

filename = '_normed_standardized_Actual_EV'
gamma = 0.5
suffix = "_"+ET

LAMBDAs = [0.1, 1.0, 10.0]

x = np.genfromtxt("x" + suffix + filename + str(gamma) + "_interactions.csv", delimiter=",")
y = np.genfromtxt("y" + suffix + filename + str(gamma) + ".csv", delimiter=",")


for LAMBDA in LAMBDAs:
    xs = x[1:,1:]
    ys = y[1:,1:]
    print xs.shape
    print ys.shape

    clf = LogisticRegression(penalty = 'l1', C = 1/LAMBDA, solver='liblinear')
    clf.fit(xs, ys.ravel())
    w = clf.coef_.ravel()

    np.save(ET +"_lambda" + str(LAMBDA) +  "_weights", w)
    print ET +"_lambda" + str(LAMBDA) +  "_weights"