## visualize weights

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import math
import seaborn as sns
import csv

LAMBDA = 1.0


ET1 = "random"

wnames = np.load('x_random_normed_standardized_Actual_EV0.5_interactions_names.npy')
w1 = np.load(ET1 +"_lambda" + str(LAMBDA) + "_weights.npy")
print wnames.shape
print w1.shape

wm = [np.absolute(i) for i in w1]
print sorted(zip(wm,w1), reverse=True)

w1 = [x for _,x in sorted(zip(wm,w1), reverse=True)]
wnames = [x for _,x in sorted(zip(wm,wnames), reverse=True)]

with open('model_weights_'+ET1+'.csv', 'w') as writeFile:
    writer = csv.writer(writeFile)
    for i in range(len(wnames)):
        writer.writerow([wnames[i], w1[i]])
