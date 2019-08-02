import numpy as np
import csv

ET = "random"
it = "with interactions"
filename = '_normed_standardized_Actual_EV'
# filename = '_normed_EV'
gamma = 0.5

suffix = "_"+ET

ratios = np.load("lr_sids_ratios_for_alllambdas_of_" + ET + suffix + filename + str(gamma) + ".npy")
print ratios.shape
LAMBDAS = np.load("lr_sids_lambdas_for_alllambdas_of_" + ET + suffix + filename + str(gamma) + ".npy")
print LAMBDAS.shape

# filename example: "results.csv"
# return the columnnames of csv file
def getcolumnnames(filename):
    with open(filename, "rb") as f:
        reader = csv.reader(f)
        i = next(reader)
    return i

if it == "no interactions":
    i = getcolumnnames("x_" + ET + filename + str(gamma) + ".csv")
    names = i[1:]
elif it == "with interactions":
    print "file : " + 'x' + suffix + filename + str(gamma) + '_interactions_names.npy'
    names = np.load('x' + suffix + filename + str(gamma) + '_interactions_names.npy')
    print names
else:
    print "it parameter was not recognized"

# compute area
# x is a list of increments along the x axis
# y is a list of highs along each increment
# left trapezoidal rules
def ca(x,y):
    area = 0.0
    for i in range(len(x) - 1):
        dx = x[i+1] - x[i]
        dy = y[i+1] - y[i]
        area = area + dx * y[i+1] + dx * dy / 2
    maxarea = max(y) * (x[-1] - x[0]) # use this if you want to normalize
    area = area
    return area

areas = np.zeros(shape=len(names))
maxes = np.zeros(shape=len(names))
with open('areas_' + ET + suffix + filename + str(gamma) + '.csv', 'a') as file:
    filewriter = csv.writer(file, delimiter=',')
    filewriter.writerow(['index', 'name','area','max'])
    counter = 0
    for f in range(len(names)):
        fname = names[f]
        fratios = ratios[:,f]
        fa = ca(LAMBDAS, fratios)
        fm = np.max(fratios)
        filewriter.writerow([counter, fname, fa, fm])
        areas[counter] = fa
        maxes[counter] = fm
        counter = counter + 1
    np.save("areas_areas_" + ET + suffix + filename + str(gamma) + ".npy", areas)
    np.save("areas_maxes_" + ET + suffix + filename + str(gamma) + ".npy", maxes)
    file.close()
