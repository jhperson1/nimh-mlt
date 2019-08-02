from sklearn.linear_model import LogisticRegression
import numpy as np
import matplotlib.pyplot as plt
import sklearn
import csv

it = "with interactions"

ET = "1block"
suffix = '_' + ET
filename = '_normed_standardized_Actual_EV'
# filename = '_normed_EV'
gamma = 0.5

# x and y both have a subject_id column
y = np.genfromtxt("y" + suffix + filename + str(gamma) + ".csv", delimiter=",")
sf = np.genfromtxt('gps' + suffix + '.csv', delimiter=",", usecols=np.arange(0,2)) # subject file
ORIGsids = np.array(sf[1:,1]) # subject ids
n = len(ORIGsids)

print ORIGsids
print n
# nTRIALS = int(float(y.shape[0] - 1) / float(n))

# if ET == "3block":
#     nTRIALS = 78

# print n
# print y.shape[0] - 1
# print nTRIALS

# number of bootstraps
T = 100
graphtype = "m" # all

# obsvs is binary
# preds is probabilities
def balancedacc(obsvs, preds, threshold):
    preds_bool = preds > threshold
    conditionpositive = sum(obsvs)
    tn, fp, fn, tp = sklearn.metrics.confusion_matrix(obsvs, preds_bool).ravel()
    tpr = float(tp) / float(tp + fn)
    fpr = float(tn) / float(tn + fp)
    bacc = (tpr + fpr) / 2
    return bacc

# obsvs is binary
# preds is probabilities
def accthreshold(obsvs, preds, threshold):
    preds_bool = preds > threshold
    acc = sklearn.metrics.accuracy_score(obsvs, preds_bool)
    return acc

def get_indices(x,xs):
    return [i for (i,y) in zip(range(len(xs)), xs) if y == x]

# obsvs is binary
# preds is probabilities
def maxacc(obsvs, preds):
    thresholds = [float(i) / float(100) for i in range(1,100,1)]
    temp = 0
    print "the accs are "
    for threshold in thresholds:
        temp2 = accuracythreshold(obsvs, preds, threshold)
        print temp2
        temp = max(temp, temp2)
    return temp

# obsvs is binary
# preds is probabilities
def maxbacc(obsvs, preds):
    thresholds = [float(i) / float(100) for i in range(1,100,1)]
    temp = 0
    print "the accs are "
    for threshold in thresholds:
        temp2 = balancedacc(obsvs, preds, threshold)
        print temp2
        temp = max(temp, temp2)
    return temp

# obsvs is binary
# preds is probabilities
def subjauc(subjs, obsvs, preds):
    ss = set(subjs)
    aucs = []
    for s in ss:
        sind = get_indices(s, subjs)
        obsvs_s = obsvs[sind]
        preds_s = preds[sind]
        fpr, tpr, thresholds = sklearn.metrics.roc_curve(obsvs_s, preds_s)
        auc_s = sklearn.metrics.auc(fpr, tpr)
        aucs.append(auc_s)
    return np.nanmean(aucs)

# sids = ORIGsids
# x and y both have a subject_id column
# return features and targets given a subject ID, x, y
# [features, targets] = subjXY (sids[0], x, y)
def subjXY(sid, x, y):
    sidscol = np.array(x[:,0])
    inds = np.where(sidscol == sid)
    features = np.squeeze(x[inds,1:])
    targets = np.squeeze(y[inds,1:])
    return [features, targets]

def subjY(sid, y):
    sidscol = np.array(y[:,0])
    inds = np.where(sidscol == sid)
    targets = np.squeeze(y[inds,1:])
    return targets

# x and y both have a subject_id column
# return all other features and targets given a subject ID, x, y
# [features, targets] = notsubjXY (sids[0], x, y)
def notsubjXY(sid, x, y):
    sidscol = np.array(x[:,0])
    inds = np.where(sidscol != sid)
    features = np.squeeze(x[inds,1:])[1:,]
    targets = np.squeeze(y[inds,1:])[1:]
    return [features, targets]

def notsubjY(sid, y):
    sidscol = np.array(y[:,0])
    inds = np.where(sidscol != sid)
    targets = np.squeeze(y[inds,1:])[1:]
    return targets

# x and y both have a subject_id column
# return accuracy of a logistic regression model
# through LOOCV
def loocv_acc(sids, x, y):
    # obsvs = np.zeros(shape=(n,nTRIALS))
    # preds = np.zeros(shape=(n,nTRIALS))
    obsvs = []
    preds = []
    probs = []
    print preds
    if len(x) == 0:
        for f in range(n):
            test_y = subjY(sids[f], y)
            train_y = notsubjY(sids[f], y)
            ym = np.mean(train_y )
            yb = ym > 0.5
            test_p = [yb for _i in range(len(test_y))]
            test_prob = [ym for _i in range(len(test_y))]
            # print test_p
            # print len(test_p)
            # print preds[f, :]
            # preds[f, :] = test_p
            # obsvs[f, :] = test_y
            preds.extend(test_p)
            obsvs.extend(test_y)
            probs.extend(test_prob)
    else:
        for f in range(n):
            if sids[f] > 10:
                [test_x, test_y] = subjXY(sids[f], x, y)
                [train_x, train_y] = notsubjXY(sids[f], x, y)
                if x.shape[1] == 2:
                    test_x = test_x.reshape(len(test_x),1)
                    train_x = train_x.reshape(len(train_x),1)
                mf = LogisticRegression(solver='liblinear') # penalty = 'l1', C = 1/LAMBDA,
                mf.fit(train_x, train_y)
                test_p = mf.predict(test_x)
                temp = mf.predict_proba(test_x)
                test_prob = np.reshape(np.array(temp[:,1]), (-1))
                # print test_p
                # preds[f,:] = test_p
                # obsvs[f,:] = test_y
                preds.extend(test_p)
                obsvs.extend(test_y)
                probs.extend(test_prob)
    # predsbool = np.reshape(np.array(preds).astype(bool), (nTRIALS*n))
    # obsvsbool = np.reshape(np.array(obsvs).astype(bool), (nTRIALS*n))
    predsbool = np.reshape(np.array(preds).astype(bool), (-1))
    obsvsbool = np.reshape(np.array(obsvs).astype(bool), (-1))
    probs = np.reshape(np.array(probs), (-1))
    bacc = balancedacc(obsvsbool, probs, 0.5)
    print "balanced accuracy is : ", bacc
    # acc = sklearn.metrics.accuracy_score(obsvsbool, predsbool)
    # # ll = sklearn.metrics.log_loss(np.squeeze(obsvs), np.squeeze(preds)) / float(len(preds))
    # print "accuracy is : ", acc
    fpr, tpr, thresholds = sklearn.metrics.roc_curve(obsvsbool, probs)
    auc = sklearn.metrics.auc(fpr, tpr)
    # print "log loss is : ", ll
    return [bacc, auc]

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
    # indices in x of the most
    # stable features
    ind_a = np.load("areas_areas_indices_" + ET + ".npy") # 0 indexed
    featrank = [ind + 1 for ind in ind_a] # 1 indexed
    #featrank = [7, 4, 2, 8, 1, 5, 3, 6]
    featranknames = [names[f-1] for f in featrank]
elif it == "with interactions":
    # x = np.genfromtxt('x_stdz_interactions.csv', delimiter=",")
    # names = np.load('x_stdz_interactions.npy')
    x = np.genfromtxt('x' + suffix + filename + str(gamma) + '_interactions.csv', delimiter=",")
    names = np.load('x' + suffix + filename + str(gamma) + '_interactions_names.npy')
    # ind_m = np.load("areas_maxes_indices.npy")
    # featrank = [ind_m[i]+1 for i in range(len(ind_m))]

    ind_a = np.load("areas_areas_indices_"+ET+ suffix + filename + str(gamma)+".npy")
    featrank = [ind_a[i]+1 for i in range(len(ind_a))]
    featranknames = [names[f-1] for f in featrank]
else:
    print "it parameter was not recognized"

print featranknames

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

# xnew = subsetx(x,featrank[0:1])
# print xnew
# loocv_acc(xnew,y)

baccs = np.zeros(shape=(T, len(featrank) + 1))
aucs = np.zeros(shape=(T, len(featrank) + 1))
for t in range(T):
    if t == 0:
        sids = ORIGsids
    else:
        np.random.seed(t) ; sids = np.random.choice(ORIGsids, n, replace=True)
    [mbacc,auc] = loocv_acc(sids, [], y)
    baccs[t,0] = mbacc
    aucs[t,0] = auc
    for m in range(len(featrank)):
        ml = featrank[0: m+1]
        xnew = subsetx(x,ml)
        [mbacc, mauc] = loocv_acc(sids, xnew, y)
        baccs[t, m+1] = mbacc
        aucs[t, m+1] = mauc
np.save("bmaccs_" + suffix + filename + str(gamma) + ".npy", bmaccs)
np.save("aucs_" + suffix + filename + str(gamma) + ".npy", aucs)
accs = np.load("accs_" + suffix + filename + str(gamma) + ".npy")
print accs

baccs = np.load("baccs_" + suffix + filename + str(gamma) + ".npy")
aucs = np.load("aucs_" + suffix + filename + str(gamma) + ".npy")

print "experiment type is : ", ET

plt.figure()
plt.xlabel("Number of Features")
plt.ylabel("Max Balanced Accuracy in LOOCV")
x = ["Baseline"]
for name in featranknames:
    x.append("+ " + name)

if graphtype == "all":
    ### # all bootstraps
    for t in range(T):
        plt.plot(x, baccs[t,:])
    plt.title("Each line is for a bootstrap resampling T = " + str(T) + " bootstrap resamplings")
elif graphtype == "m":
    ### # just the means and stds
    plt.plot(x, baccs.mean(axis=0))
    #plt.errorbar(x, baccs.mean(axis=0), yerr = baccs.std(axis=0))
    for i, txt in enumerate(baccs.mean(axis=0)):
        plt.annotate(round(txt,2), (x[i], baccs.mean(axis=0)[i]))
    plt.title("Mean Balanced Accuracy for T = " + str(T) + " bootstrap resamplings for " + ET)
elif graphtype == "ms":
    ### # just the means and stds
    plt.errorbar(x, baccs.mean(axis=0), yerr = baccs.std(axis=0))
    for i, txt in enumerate(baccs.mean(axis=0)):
        plt.annotate(round(txt,2), (x[i], baccs.mean(axis=0)[i]))
    plt.title("Means and STDs for T = " + str(T) + " bootstrap resamplings for " + ET)
elif graphtype == "one":
    plt.plot(x, baccs[0,:])
    for i, txt in enumerate(baccs[0,:]):
        plt.annotate(round(txt,2), (x[i], baccs[0,:][i]))
    plt.title("Only for Original Sample")
else:
    print "graphtype parameter not recognized"
plt.show()

plt.figure()
plt.xlabel("Number of Features")
plt.ylabel("AUC in LOOCV")
plt.plot(x,aucs.mean(axis=0))
#plt.errorbar(x,aucs.mean(axis=0), yerr = aucs.std(axis=0))
for i, txt in enumerate(aucs.mean(axis=0)):
    plt.annotate(round(txt,2), (x[i], aucs.mean(axis=0)[i]))
plt.title("Means and STDs for T = " + str(T) + " bootstrap resamplings for " + ET)

plt.show()

# plt.figure()

# plt.bar(x, accs)
# for i, txt in enumerate(accs):
#     plt.annotate(round(txt,4), (x[i], accs[i]))
# plt.show()
