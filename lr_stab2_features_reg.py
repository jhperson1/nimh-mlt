from sklearn.linear_model import LogisticRegression
import numpy as np
#import tensorflow as tf
import sklearn as sk
import matplotlib.pyplot as plt
import math
import time
import csv

overlaps = True

epsilon=0.0
tolerance = 0.0005
it = "with interactions"

# where to get features from
data = "random"
NFEATS = 7

# where to build model
ET = "1block"
suffix = '_' + ET
filename = '_normed_standardized_Actual_EV'
# filename = '_normed_EV'
gamma = 0.5

# if ET == "random":
#     suffix = "_random"
# elif ET == "1block":
#     suffix = ""
# elif ET == "3block":
#     suffix = "_3block"
# else:
#     print "experiment type (ET) parameter not recognized"

# x has a subject_id column
# l is a 0-indexed list of desired column numbers
# return x with only the columns in l
# plus the 0th column should still be subject ids
def subsetx(x, l):
    xnew = np.zeros(shape=(x.shape[0], len(l) + 1))
    xnew[:,0] = x[:,0]
    for col in range(len(l)):
        colind = l[col]
        xnew[:,col+1] = x[:,colind]
    return xnew

# filename example: "results.csv"
# return the columnnames of csv file
def getcolumnnames(filename):
    with open(filename, "rb") as f:
        reader = csv.reader(f)
        i = next(reader)
    return i

if it == "no interactions":
    x = np.genfromtxt("x" + suffix + filename + str(gamma) + ".csv", delimiter=",")
    i = getcolumnnames("x" + suffix + filename + str(gamma) + ".csv")
    names = i[1:]
    print names
    nf = len(names)
elif it == "with interactions":
    if overlaps:
        x = np.genfromtxt('x' + suffix + filename + str(gamma) + '_interactions_stdz_overlap.csv', delimiter=",")
    else:
        x = np.genfromtxt('x' + suffix + filename + str(gamma) + '_interactions_stdz.csv', delimiter=",")
    names = np.load('x' + suffix + filename + str(gamma) + '_interactions_names.npy')
    nf = len(names)
else:
    print "it parameter was not recognized"

if overlaps:
    y = np.genfromtxt("y" + suffix + filename + str(gamma) + "_overlap.csv", delimiter=",")
else:
    y = np.genfromtxt("y" + suffix + filename + str(gamma) + ".csv", delimiter=",")
# select gps
if ET == "1block":
    gps = 'gps.csv'
elif ET == "3block":
    gps = 'gps_3block.csv'
elif ET == "random":
    gps = 'gps_random.csv'
else:
    print "error"
sf = np.genfromtxt(gps, delimiter=",", usecols=np.arange(0,2)) # subject file
ORIGsids = np.array(sf[1:,1]) # subject ids
n = len(ORIGsids)

# number of bootstraps
start = 0
T = 100

LAMBDAS = map(lambda x : math.exp(float(x) / 2) , range(-8,20,1))
# LAMBDAS = LAMBDAS[2:21:1]
#LAMBDAS = LAMBDAS[1:10:4]

nonzero = lambda x : not (x < tolerance and x > - tolerance)

# return features and targets given a subject ID, x, y
# [features, targets] = subjXY (sids[0], x, y)
def subjXY(sid, x, y):
    sidscol = np.array(x[:,0])
    inds = np.where(sidscol == sid)
    features = np.squeeze(x[inds,1:])
    targets = y[inds,1:]
    size = features.shape[0]
    return [features, targets, size]

def getXY(sids, x, y):
    xs = np.zeros(shape=[x.shape[0]-1, x.shape[1]-1])
    ys = np.zeros(shape=[y.shape[0]-1, 1])
    xbool = np.zeros(shape=y.shape[0]-1)

    counter = 0
    for i in range(len(sids)):
        [sx, sy, size] = subjXY(sids[i], x, y)
        xs[range(counter, counter+size), :] = sx
        ys[range(counter, counter+size)] = sy
        counter = counter + size
    return [xs, ys]

# return a vector of feature weights
def fitmodel(xs, ys, LAMBDA):
    clf = LogisticRegression(penalty = 'l1', C = 1/LAMBDA, solver='liblinear')
    clf.fit(xs, ys.ravel())
    w = clf.coef_.ravel()
    return w

# return a list with a vector of feature weights and the intercept
def fitmodelwi(xs, ys, LAMBDA):
    clf = LogisticRegression(penalty = 'l1', C = 1/LAMBDA, solver='liblinear')
    clf.fit(xs, ys.ravel())
    w = clf.coef_.ravel()
    i = clf.intercept_.ravel()
    return [w,i]

def reggraph(x, y):
    XY = getXY(ORIGsids, x, y)
    xs = XY[0]
    ys = XY[1]
    tj = time.time()
    for LAMBDA in LAMBDAS:
        ti = time.time()
        print "time taken : ", ti - tj
        tj = ti
        print "lambda : {}".format(LAMBDA)
        w = fitmodel(xs, ys, LAMBDA)
        np.save("lr_origsids_lambda"+str(LAMBDA)+"_epsilon"+str(epsilon)+ suffix + filename + str(gamma)+"_weights.npy", w)
    return

def reggraphinteractions(x, y, NFEATS):
    # get x, only some features
    ind_a = np.load("areas_areas_indices_" + data + ".npy") # 0 indexed
    names = np.load('x_' + ET + filename + str(gamma) + '_interactions_names.npy')
    featrank = [ind + 1 for ind in ind_a] # 1 indexed
    featranknames = [names[f-1] for f in featrank]
    ml = featrank[0: NFEATS]
    xnew = subsetx(x,ml)
    # get x, y
    XY = getXY(ORIGsids, xnew, y)
    xs = XY[0]
    ys = XY[1]

    tj = time.time()
    for LAMBDA in LAMBDAS:
        ti = time.time()
        print "time taken : ", ti - tj
        tj = ti
        print "lambda : {}".format(LAMBDA)
        [w,i] = fitmodelwi(xs, ys, LAMBDA)
        np.save("lr_origsids_interactions_NFEATS" + str(NFEATS) + "_lambda"+str(LAMBDA)+"_epsilon"+str(epsilon)+ suffix + filename + str(gamma) + "_overlaps"+ str(overlaps)+  "_weights.npy", w)
        np.save("lr_origsids_interactions_NFEATS" + str(NFEATS) + "_lambda"+str(LAMBDA)+"_epsilon"+str(epsilon)+ suffix + filename + str(gamma) + "_overlaps"+ str(overlaps)+"_intercept.npy", i)
    return

def reggraphsaveinteractions(ET1, ET2, NFEATS):
    ind_a = np.load("areas_areas_indices_" + data + ".npy") # 0 indexed
    names = np.load('x_' + ET + filename + str(gamma) + '_interactions_names.npy')
    featrank = [ind + 1 for ind in ind_a] # 1 indexed
    featranknames = [names[f-1] for f in featrank]
    ml = featrank[0: NFEATS]
    fn = featranknames[0: NFEATS]

    weights1 = np.zeros(shape=(len(LAMBDAS), NFEATS))
    weights2 = np.zeros(shape=(len(LAMBDAS), NFEATS))
    counter = 0
    for LAMBDA in LAMBDAS:
        i1 = np.load("lr_origsids_interactions_NFEATS" + str(NFEATS) + "_lambda"+str(LAMBDA)+"_epsilon"+str(epsilon)+ "_" + ET1 + filename + str(gamma)+ "_overlaps"+ str(overlaps)+"_intercept.npy")
        i2 = np.load("lr_origsids_interactions_NFEATS" + str(NFEATS) + "_lambda"+str(LAMBDA)+"_epsilon"+str(epsilon)+ "_" + ET2 + filename + str(gamma)+ "_overlaps"+ str(overlaps)+"_intercept.npy")
        w1 = np.load("lr_origsids_interactions_NFEATS" + str(NFEATS) + "_lambda"+str(LAMBDA)+"_epsilon"+str(epsilon)+ "_" + ET1 + filename + str(gamma)+ "_overlaps"+ str(overlaps)+"_weights.npy")
        w2 = np.load("lr_origsids_interactions_NFEATS" + str(NFEATS) + "_lambda"+str(LAMBDA)+"_epsilon"+str(epsilon)+ "_" + ET2 + filename + str(gamma)+ "_overlaps"+ str(overlaps)+"_weights.npy")
        print w1
        print w2
        weights1[counter, :] = w1
        weights2[counter, :] = w2
        counter = counter + 1
    cm = plt.get_cmap('gist_rainbow')
    fig = plt.figure()
    ax = fig.add_subplot(111)
    NUM_COLORS = NFEATS
    originalcycle = [cm(1.*i/NUM_COLORS) for i in range(NUM_COLORS)]
    doublescycle = []
    for e in originalcycle:
        doublescycle.append(e)
        doublescycle.append(e)
    markerdoublecycle = []
    for _i in range(NFEATS):
        markerdoublecycle.append('.')
        markerdoublecycle.append('+')
    doublefn = []
    for featurename in fn:
        doublefn.append(featurename + ' (' + ET1 + ')')
        doublefn.append(featurename + ' (' + ET2 + ')')
    ax.set_prop_cycle(color = doublescycle, marker = markerdoublecycle)
    for f in range(NFEATS):
        ax.plot(LAMBDAS, weights1[:,f]) # 'o'
        ax.plot(LAMBDAS, weights2[:,f]) # '+'
    ax.axhline(y=0, color='k')
    # for f in range(len(l)):
    #     plt.plot(LAMBDAS, weights[:,f])
    plt.xscale('log')
    plt.legend(doublefn, loc='upper right')
    plt.title("Regularization Plot")
    plt.savefig("regplotsave.png")
    plt.show()
    return

def reggraphsaveinteractionsintercept(ET1, ET2, NFEATS):
    ind_a = np.load("areas_areas_indices_" + data + ".npy") # 0 indexed
    names = np.load('x_' + ET + filename + str(gamma) + '_interactions_names.npy')
    featrank = [ind + 1 for ind in ind_a] # 1 indexed
    featranknames = [names[f-1] for f in featrank]
    ml = featrank[0: NFEATS]
    fn = featranknames[0: NFEATS]

    weights1 = np.zeros(shape=(len(LAMBDAS), NFEATS+1))
    weights2 = np.zeros(shape=(len(LAMBDAS), NFEATS+1))
    counter = 0
    for LAMBDA in LAMBDAS:
        i1 = np.load("lr_origsids_interactions_NFEATS" + str(NFEATS) + "_lambda"+str(LAMBDA)+"_epsilon"+str(epsilon)+ "_" + ET1 + filename + str(gamma)+"_intercept.npy")
        i2 = np.load("lr_origsids_interactions_NFEATS" + str(NFEATS) + "_lambda"+str(LAMBDA)+"_epsilon"+str(epsilon)+ "_" + ET2 + filename + str(gamma)+"_intercept.npy")
        w1 = np.load("lr_origsids_interactions_NFEATS" + str(NFEATS) + "_lambda"+str(LAMBDA)+"_epsilon"+str(epsilon)+ "_" + ET1 + filename + str(gamma)+"_weights.npy")
        w2 = np.load("lr_origsids_interactions_NFEATS" + str(NFEATS) + "_lambda"+str(LAMBDA)+"_epsilon"+str(epsilon)+ "_" + ET2 + filename + str(gamma)+"_weights.npy")
        print w1
        print w2
        weights1[counter, :] = np.concatenate((w1, i1))
        weights2[counter, :] = np.concatenate((w2, i2))
        counter = counter + 1
    cm = plt.get_cmap('gist_rainbow')
    fig = plt.figure()
    ax = fig.add_subplot(111)
    NUM_COLORS = NFEATS+1
    originalcycle = [cm(1.*i/NUM_COLORS) for i in range(NUM_COLORS)]
    doublescycle = []
    for e in originalcycle:
        doublescycle.append(e)
        doublescycle.append(e)
    markerdoublecycle = []
    for _i in range(NFEATS+1):
        markerdoublecycle.append('.')
        markerdoublecycle.append('+')
    doublefn = []
    for featurename in fn:
        doublefn.append(featurename + ' (' + ET1 + ')')
        doublefn.append(featurename + ' (' + ET2 + ')')
    doublefn.append('intercept (' + ET1 + ')')
    doublefn.append('intercept (' + ET2 + ')')
    ax.set_prop_cycle(color = doublescycle, marker = markerdoublecycle)
    for f in range(NFEATS+1):
        ax.plot(LAMBDAS, weights1[:,f]) # 'o'
        ax.plot(LAMBDAS, weights2[:,f]) # '+'
    ax.axhline(y=0, color='k')
    # for f in range(len(l)):
    #     plt.plot(LAMBDAS, weights[:,f])
    plt.xscale('log')
    plt.legend(doublefn, loc='upper right')
    plt.title("Regularization Plot")
    plt.savefig("regplotsave.png")
    plt.show()
    return

def reggraphsave(x, y):
    XY = getXY(ORIGsids, x, y)
    xs = XY[0]
    ys = XY[1]
    weights = np.zeros(shape=(len(LAMBDAS), nf))
    counter = 0
    for LAMBDA in LAMBDAS:
        w = np.load("lr_origsids_lambda"+str(LAMBDA)+"_epsilon"+str(epsilon)+ suffix + filename + str(gamma)+"_weights.npy")
        weights[counter, :] = w
        counter = counter + 1
    for f in range(len(names)):
        plt.plot(LAMBDAS, weights[:,f])

    plt.xscale('log')
    plt.legend(names, loc='upper right')
    plt.title("Regularization Plot")
    plt.savefig("regplotsave.png")
    plt.show()
    return

def reggraphsavefeat(x, y, l):
    XY = getXY(ORIGsids, x, y)
    xs = XY[0]
    ys = XY[1]
    weights = np.zeros(shape=(len(LAMBDAS), nf))
    counter = 0
    for LAMBDA in LAMBDAS:
        w = np.load("lr_origsids_lambda"+str(LAMBDA)+"_epsilon"+str(epsilon)+ suffix + filename + str(gamma)+"_weights.npy")
        weights[counter, :] = w
        counter = counter + 1
    cm = plt.get_cmap('gist_rainbow')
    fig = plt.figure()
    ax = fig.add_subplot(111)
    NUM_COLORS = len(l)
    ax.set_color_cycle([cm(1.*i/NUM_COLORS) for i in range(NUM_COLORS)])
    for f in l:
        ax.plot(LAMBDAS, weights[:,f])
    # for f in range(len(l)):
    #     plt.plot(LAMBDAS, weights[:,f])

    plt.xscale('log')
    plt.legend([names[l[i]] for i in range(len(l))], loc='lower right')
    plt.title("Regularization Plot")
    plt.savefig("regplotsave.png")
    plt.show()
    return

def stabgraph(x, y):
    index = 0
    tj = time.time()
    for LAMBDA in LAMBDAS:
        counts = np.zeros(8)
        for t in range(start, start + T):
            ti = time.time()
            print "time taken : ", ti - tj
            tj = ti
            index = index + 1
            print "lambda : {} - Subsample : {}".format(LAMBDA, t)
            ns = len(ORIGsids) / 2
            np.random.seed(t) ; sids = np.random.choice(ORIGsids, ns, replace=False)
            XY = getXY(sids, x, y)
            xs = XY[0]
            ys = XY[1]
            w = fitmodel(xs, ys, LAMBDA)
            np.save("lr2_sids_subsample"+str(t)+"_lambda"+str(LAMBDA)+"_epsilon"+str(epsilon)+ suffix + filename + str(gamma)+"_weights.npy", w)
    return

def stabgraphhist(i):
    LAMBDA = LAMBDAS[i]
    print "Lambda : ", LAMBDA
    weights = np.zeros(shape=(T, nf))
    bools = np.zeros(shape=(T, nf))
    index = 0
    for t in range(start, start+T):
        w = np.load("lr2_sids_subsample"+str(t)+"_lambda"+str(LAMBDA)+"_epsilon"+str(epsilon)+ suffix + filename + str(gamma)+"_weights.npy")
        weights[index, :] = w
        b = map(lambda x : not (x < tolerance and x > - tolerance), w)
        bools[index, :] = b
        index = index + 1
    fig, axes = plt.subplots(nf, 1, sharex=True, sharey=True)
    print weights
    for f in range(nf):
        gi = weights[:,f]
        axes[f].hist(gi)
        axes[f].axvline(gi.mean(), color='b', linestyle='dashed', linewidth=1)
        axes[f].text(gi.mean() + gi.mean()/8, 10, 'NN Mean: {:.3f}'.format(gi.mean()), color='b')
        axes[f].set_title(names[f] + " Weights in T : " + str(T) + " Subsamples")
        # print "mean: ", gi.mean()
        # print "sd : ", gi.std()
    # plt.savefig("grads"+"_"+REG+"_layers"+str(LAYERS)+"_lambda"+str(LAMBDA)+"_epsilon"+str(epsilon)+"_noise"+NOISE+"_feature"+names[f]+'.png')
    plt.show()

    return

def stabgraphsave(x, y):
    ratios = np.zeros(shape=(len(LAMBDAS), nf))
    counter = 0
    for LAMBDA in LAMBDAS:
        all_signs = np.zeros(shape=(T, nf))
        counts = np.zeros(nf)
        for t in range(start, start+T):
            w = np.load("lr2_sids_subsample"+str(t)+"_lambda"+str(LAMBDA)+"_epsilon"+str(epsilon)+ suffix + filename + str(gamma)+"_weights.npy")
            #print "Weights : ", w
            # b = map(nonzero, w)
            # #print "Bools : ", b
            # counts = counts + b
            all_signs[t - start, :] = map(np.sign, w)
        pos_sum = np.sum(all_signs == 1, 0)
        neg_sum = np.sum(all_signs == -1, 0)
        counts = np.maximum(pos_sum, neg_sum)
        #print "Lambda : ", LAMBDA, " Counts : ", counts
        ratios[counter, :] = map(lambda x : float(x) / float(T), counts)
        #print "Lambda : ", LAMBDA, " Ratios : ", ratios[counter, :]
        counter = counter + 1
    plt.figure()
    for f in range(len(names)):
        plt.plot(LAMBDAS, ratios[:,f])
    np.save("lr_sids_ratios_for_alllambdas_of_" + ET + suffix + filename + str(gamma) + ".npy", ratios)
    np.save("lr_sids_lambdas_for_alllambdas_of_" + ET + suffix + filename + str(gamma) + ".npy", LAMBDAS)
    plt.legend(names, loc='lower left')
    plt.title("Stability Plot")
    plt.xscale('log')
    plt.savefig("stabgplotsave.png")
    plt.show()
    return

def stabgraphsavenolegend(x, y):
    ratios = np.zeros(shape=(len(LAMBDAS), nf))
    counter = 0
    for LAMBDA in LAMBDAS:
        all_signs = np.zeros(shape=(T, nf))
        counts = np.zeros(nf)
        for t in range(start, start+T):
            w = np.load("lr2_sids_subsample"+str(t)+"_lambda"+str(LAMBDA)+"_epsilon"+str(epsilon)+ suffix + filename + str(gamma)+"_weights.npy")
            #print "Weights : ", w
            # b = map(nonzero, w)
            # #print "Bools : ", b
            # counts = counts + b
            all_signs[t - start, :] = map(np.sign, w)
        pos_sum = np.sum(all_signs == 1, 0)
        neg_sum = np.sum(all_signs == -1, 0)
        counts = np.maximum(pos_sum, neg_sum)
        #print "Lambda : ", LAMBDA, " Counts : ", counts
        ratios[counter, :] = map(lambda x : float(x) / float(T), counts)
        #print "Lambda : ", LAMBDA, " Ratios : ", ratios[counter, :]
        counter = counter + 1
    plt.figure()
    for f in range(len(names)):
        plt.plot(LAMBDAS, ratios[:,f])
    np.save("lr_sids_ratios_for_alllambdas_of_" + ET + suffix + filename + str(gamma)+ ".npy", ratios)
    np.save("lr_sids_lambdas_for_alllambdas_of_" + ET + suffix + filename + str(gamma)+ ".npy", LAMBDAS)
    #plt.legend(names, loc='lower left')
    plt.title("Stability Plot")
    plt.xscale('log')
    plt.savefig("stabgplotsave.png")
    plt.show()
    return

# plot specific features with indices in list l
# l should have 0-indexed indices
def stabgraphsavefeat(x, y, l):
    ratios = np.zeros(shape=(len(LAMBDAS), nf))
    counter = 0
    for LAMBDA in LAMBDAS:
        all_signs = np.zeros(shape=(T, nf))
        counts = np.zeros(nf)
        for t in range(start, start+T):
            w = np.load("lr2_sids_subsample"+str(t)+"_lambda"+str(LAMBDA)+"_epsilon"+str(epsilon)+ suffix + filename + str(gamma)+"_weights.npy")
            #print "Weights : ", w
            # b = map(nonzero, w)
            # #print "Bools : ", b
            # counts = counts + b
            all_signs[t - start, :] = map(np.sign, w)
        pos_sum = np.sum(all_signs == 1, 0)
        neg_sum = np.sum(all_signs == -1, 0)
        counts = np.maximum(pos_sum, neg_sum)
        #print "Lambda : ", LAMBDA, " Counts : ", counts
        ratios[counter, :] = map(lambda x : float(x) / float(T), counts)
        #print "Lambda : ", LAMBDA, " Ratios : ", ratios[counter, :]
        counter = counter + 1
    cm = plt.get_cmap('gist_rainbow')
    fig = plt.figure()
    ax = fig.add_subplot(111)
    NUM_COLORS = len(l)
    ax.set_color_cycle([cm(1.*i/NUM_COLORS) for i in range(NUM_COLORS)])
    for f in l:
        ax.plot(LAMBDAS, ratios[:,f])
    namesl = [names[l[i]] for i in range(len(l))]
    np.save("lr_sids_ratios_for_alllambdas_of_" + ET + suffix + filename + str(gamma)+ ".npy", ratios)
    np.save("lr_sids_lambdas_for_alllambdas_of_" + ET + suffix + filename + str(gamma)+ ".npy", LAMBDAS)
    plt.legend(namesl, loc='lower left')
    plt.title("Stability Plot")
    plt.xscale('log')
    plt.savefig("stabgplotsave.png")
    plt.show()

    return



# stabgraph(x, y)
# stabgraphsavenolegend(x, y)
# reggraph(x,y)

# NFEATS = 7
# ind_a = np.load("areas_areas_indices_" + ET + suffix + filename + str(gamma)+ ".npy")
# ind_m = np.load("areas_maxes_indices_" + ET + suffix + filename + str(gamma)+ ".npy")
# print ind_a
# print ind_m
# l = ind_a[:NFEATS]
# print len(l)
# stabgraphsavefeat(x, y, l)
# reggraphsavefeat(x, y, l)

reggraphinteractions(x, y, NFEATS)
# reggraphsaveinteractions("random", "1block", NFEATS)
reggraphsaveinteractionsintercept("random", "1block", NFEATS)
