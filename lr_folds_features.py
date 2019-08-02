## LOOCV

from sklearn.linear_model import LogisticRegression
import numpy as np
import matplotlib.pyplot as plt
import sklearn
import csv
from decimal import Decimal
colors = ['aqua', 'darkorange', 'cornflowerblue']

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
        temp2 = accthreshold(obsvs, preds, threshold)
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
        print "obsvs_s : ", obsvs_s
        print "preds_s : ", preds_s
        fpr, tpr, thresholds = sklearn.metrics.roc_curve(obsvs_s, preds_s)
        auc_s = sklearn.metrics.auc(fpr, tpr)
        aucs.append(auc_s)
    return np.nanmean(aucs)


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

# return all other features and targets given a subject ID, x, y
# [features, targets] = notsubjXY (sids[0], x, y)
def notsubjXY(sid, x, y):
    sidscol = np.array(x[:,0])
    # print len(sidscol)
    # print sid
    inds = np.where(sidscol != sid)
    features = np.squeeze(x[inds,1:])[1:,]
    targets = np.squeeze(y[inds,1:])[1:]
    return [features, targets]

def notsubjY(sid, y):
    sidscol = np.array(y[:,0])
    inds = np.where(sidscol != sid)
    targets = np.squeeze(y[inds,1:])[1:]
    return targets

# x has a subject_id column
# feats is a 0-indexed list of desired column numbers
# return x with only the subject_id col and the columns in feats
# don't need to include 0, the subject_id column
# in the list feats
def subsetxfile(filename, feats):
    x = np.genfromtxt(filename, delimiter=",")
    xnew = np.zeros(shape=(x.shape[0], len(feats) + 1))
    xnew[:,0] = x[:,0]
    for col in range(len(feats)):
        colind = feats[col]
        xnew[:,col+1] = x[:,colind]
    return xnew

# input matrix, output matrix without column
# names or rownames
def strip(m):
    m = m[1:, 1:]
    return m

# input filename, output matrix without column
# names or rownames
def stripfile(filename):
    m = np.genfromtxt(filename, delimiter=",")
    m = m[1:, 1:]
    return m

# given filename, write a csv file of the same name
# with the additional suffix _stdz
# standardize with the file's own m and s
def standardizeIN(filename):
    colnames = np.empty(shape=[9])
    with open(filename+'.csv', 'rb') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        counter = -1
        for row in reader:
            if counter <0:
                colnames = row
            counter = counter + 1
    ft = np.empty(shape=[counter,9])
    with open(filename+'.csv', 'rb') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        counter = -1
        for row in reader:
            if counter >= 0:
                ft[counter, :] = row
            counter = counter + 1
    csvfile.close()
    for col in range(1,9):
        m = ft[:,col].mean(axis=0)
        s = ft[:,col].std(axis=0)
        print "for col ", col, " mean is : ", m
        print "for col ", col, " std is : ", s
        ft[:,col] = (ft[:,col] - m) / s
    with open(filename + '_stdz.csv', 'w') as writeFile:
        writer = csv.writer(writeFile)
        writer.writerow(colnames)
        writer.writerows(ft)
    writeFile.close()
    return

# given filename, write a csv file of the same name
# with the additional suffix _stdz
# also pass a mean and std
def standardizeEX(filename, m, s):
    colnames = np.empty(shape=[9])
    with open(filename+'.csv', 'rb') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        counter = -1
        for row in reader:
            if counter <0:
                colnames = row
            counter = counter + 1
    ft = np.empty(shape=[counter,9])
    with open(filename+'.csv', 'rb') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        counter = -1
        for row in reader:
            if counter >= 0:
                ft[counter, :] = row
            counter = counter + 1
    csvfile.close()
    for col in range(1,9):
        ft[:,col] = (ft[:,col] - m[col-1]) / s[col-1]
    with open(filename + '_stdz.csv', 'w') as writeFile:
        writer = csv.writer(writeFile)
        writer.writerow(colnames)
        writer.writerows(ft)
    writeFile.close()
    return

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

# gps is a filename, with 2nd column as subject ids
# model is a string that matches one of the if statemetns
# x is an array of features, with the 1st column as subject ids
# y is an array of targets, with the 1st column as subject ids
def runmodel(gps, model, NFEATS, x, y, SEED, LAMBDA):
    # stable features
    ind_a = np.load("areas_areas_indices_" + data + ".npy") # 0 indexed
    featrank = [ind + 1 for ind in ind_a] # 1 indexed
    #featrank = [7, 4, 2, 8, 1, 5, 3, 6]
    featranknames = [names[f-1] for f in featrank]
    ml = featrank[0: NFEATS]
    xnew = subsetx(x,ml)

    sf = np.genfromtxt(gps, delimiter=",",usecols=np.arange(0,2)) # subject file
    ORIGsids = np.array(sf[1:,1]) # subject ids
    ORIGsids = list(filter(lambda x: x > 10, ORIGsids))
    print ORIGsids
    n = len(ORIGsids)
    # print n
    if SEED == 0:
        sids = ORIGsids
    else:
        np.random.seed(SEED) ; sids = np.random.choice(ORIGsids, n)
    obsvs = []
    preds = []
    probs = []
    subjs = []
    for fold in range(n):
        if model == model == "topfeatures":
            [test_x, test_y] = subjXY(sids[fold], xnew, y)
            [train_x, train_y] = notsubjXY(sids[fold], xnew, y)
            mf = LogisticRegression(solver='liblinear', penalty = 'l1', C = 1/LAMBDA)
            mf.fit(train_x, train_y)
            test_pred = mf.predict(test_x)
            temp = mf.predict_proba(test_x)
            test_prob = np.reshape(np.array(temp[:,1]), (-1))
        elif model == "allfeatures"  or model == "top2features" or model == "top3features" or model == "top5features" or model == "top10features":
            [test_x, test_y] = subjXY(sids[fold], x, y)
            [train_x, train_y] = notsubjXY(sids[fold], x, y)
            mf = LogisticRegression(solver='liblinear') # penalty = 'l1', C = 1/LAMBDA,
            mf.fit(train_x, train_y)
            test_pred = mf.predict(test_x)
            temp = mf.predict_proba(test_x)
            test_prob = np.reshape(np.array(temp[:,1]), (-1))
        elif model == "baseline":
            test_y = subjY(sids[fold], y)
            train_y = notsubjY(sids[fold], y)
            ym = np.mean(train_y)
            yb = ym > 0.5
            test_pred = [yb for _i in range(len(test_y))]
            if yb == 1:
                print "predict 1 always with probability ", ym
                test_prob = [ym for _i in range(len(test_y))]
            else:
                print "predict 0 always with probability ", 1- ym
                test_prob = [1- ym for _i in range(len(test_y))]
        else:
            print "model parameter not recognized"
        preds.append(test_pred)
        obsvs.append(test_y)
        probs.append(test_prob)
        subjs.append([sids[fold] for i in range(len(test_pred))])
    subjs = np.array(subjs).reshape(-1)
    probs = np.array(probs).reshape(-1)
    predsbool = np.reshape(np.array(preds).astype(bool), (-1))
    obsvsbool = np.reshape(np.array(obsvs).astype(bool), (-1))
    bacc3 = balancedacc(obsvsbool, probs, 0.3)
    bacc5 = balancedacc(obsvsbool, probs, 0.5)
    bacc7 = balancedacc(obsvsbool, probs, 0.7)
    mbacc = maxbacc(obsvsbool, probs)
    fpr, tpr, thresholds = sklearn.metrics.roc_curve(obsvsbool, probs)
    print ""
    print "for model ", model
    auc = sklearn.metrics.auc(fpr, tpr)
    sauc = subjauc(subjs, obsvsbool, probs)
    with open('predictions_'+data+'.csv', 'w') as writeFile:
        writer = csv.writer(writeFile)
        writer.writerow(["subjid", "truth", "prob"])
        writer.writerows(zip(subjs,obsvsbool, probs))
    writeFile.close()
    ll = sklearn.metrics.log_loss(np.squeeze(obsvsbool), np.squeeze(probs)) / float(len(probs))
    print "balanced accuracy is : ", round(bacc5,4)
    print "auc is : ", round(auc,4)
    print "log loss is : ", '%.4E' % Decimal(ll)
    return [bacc3, bacc5, bacc7, mbacc, fpr, tpr, auc, sauc, ll]

## Decide where features are coming from
ET = "random"
specificNFEATS = 45
ETsuffix = "_" + ET
## Decide what data to test in
data = "random" # where you're testing models in
suffix = "_"+data

# select gps
if data == "1block":
    gps = 'gps.csv'
elif data == "3block":
    gps = 'gps_3block.csv'
elif data == "random":
    gps = 'gps_random.csv'
else:
    print "error"

filename = '_normed_standardized_Actual_EV' # '_normed_EV'
# filename = '_normed_EV'
gamma = 0.5
LAMBDA = 1.0
LAMBDAS = [0.1, 1.0, 10.0]
nSEEDS = 3

ind_a = np.load("areas_areas_indices_" + ET + ETsuffix + filename + str(gamma)+ ".npy")
names = np.load('x_' + ET + filename + str(gamma) + '_interactions_names.npy')


models = ["topfeatures"] # ["baseline", "topfeatures", "topfeatures"]
NFEATSs = [len(ind_a)] #[0, specificNFEATS, len(ind_a)]

x = np.genfromtxt("x" + suffix + filename + str(gamma) + "_interactions.csv", delimiter=",")
y = np.genfromtxt("y" + suffix + filename + str(gamma) + ".csv", delimiter=",")

with open('lr_results.csv', 'w') as writeFile:
    writer = csv.writer(writeFile)
    results = []
    for SEED in range(nSEEDS):
        for LAMBDA in LAMBDAS:
            for model, NFEATS in zip(models, NFEATSs):
                [bacc3, bacc5, bacc7, mbacc, fpr, tpr, auc, sauc, ll] = runmodel(gps, model, NFEATS, x, y, SEED, LAMBDA)
                writer.writerow([bacc5, str(LAMBDA), str(SEED)])
                results.append([bacc5, str(LAMBDA), str(SEED)])

# results = []
# for model, NFEATS in zip(models, NFEATSs):

#     [bacc3, bacc5, bacc7, mbacc, fpr, tpr, auc, sauc, ll] = runmodel(gps, model, NFEATS, x, y)
#     results.append([model, str(NFEATS), round(bacc3,4), round(bacc5,4),
#         round(bacc7,4), round(mbacc,4),
#         round(auc,4),round(sauc,4),'%.4E' % Decimal(ll)])

#     # [acc, fpr, tpr, auc,ll] = runmodel(gps, model, NFEATS, x, y)
#     # results.append([model, str(NFEATS), round(acc,4), round(auc,4), '%.4E' % Decimal(ll)])

# with open('results.csv', 'w') as writeFile:
#     writer = csv.writer(writeFile)
#     writer.writerow(["model", "number of features", "bacc3","bacc5","bacc7", "mbacc", "auc", "sauc", "ll"])
#     writer.writerows(results)
# writeFile.close()


