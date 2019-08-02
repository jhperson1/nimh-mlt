## Transfer


from sklearn.linear_model import LogisticRegression
import numpy as np
import matplotlib.pyplot as plt
import sklearn
import csv
from decimal import Decimal

# where to get your features from
ET = "random"
specificNFEATS = 7
# ET = "1block"
# specificNFEATS = 8
it = "with interactions"
filename = '_normed_standardized_Actual_EV'
gamma = 0.5

suffix = "_"+ET

# ind_a = np.load("areas_areas_indices_" + ET + ".npy") # 0 indexed
print "areas_areas_indices_" + ET + suffix + filename + str(gamma)+ ".npy"
ind_a = np.load("areas_areas_indices_" + ET + suffix + filename + str(gamma)+ ".npy") # 0 indexed
featrank = [ind + 1 for ind in ind_a] # 1 indexed
colors = ['aqua', 'darkorange', 'cornflowerblue']
#colnames = ['subject_id','age','gender', 'ptype', 'CertainAmount', 'GamblingRange', 'onebImpMood', 'Ag3CertainReward', 'Ag3GamblingReward']

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
        fpr, tpr, thresholds = sklearn.metrics.roc_curve(obsvs_s, preds_s)
        auc_s = sklearn.metrics.auc(fpr, tpr)
        aucs.append(auc_s)
    return np.nanmean(aucs)


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

# x has a subject_id column
# sids is a list of desired subject ids
# return x without subject ids or column names, except now only for subjects in sids
def subsetfilesubj(filename, sids):
    x = np.genfromtxt(filename+'.csv', delimiter=",")
    sidscol = np.array(x[:,0])
    inds = []
    for sid in sids:
        ind = np.where(sidscol == sid)
        inds.append(ind)
    inds = np.array(inds).reshape((-1))
    # print inds
    fts = np.squeeze(x[inds,])
    with open(filename + '_overlap_frm'+ traintype + '.csv', 'w') as writeFile:
        writer = csv.writer(writeFile)
        writer.writerow(np.zeros(x.shape[1]))
        writer.writerows(fts)
    writeFile.close()
    return fts[:,1:]

# x has a subject_id column
# sids is a list of desired subject ids
# return x without subject ids or column names, except now only for subjects in sids
def subsetfilesubjSUBJS(filename, sids):
    x = np.genfromtxt(filename+'.csv', delimiter=",")
    sidscol = np.array(x[:,0])
    inds = []
    for sid in sids:
        ind = np.where(sidscol == sid)
        inds.append(ind)
    inds = np.array(inds).reshape((-1))
    # print inds
    fts = np.squeeze(x[inds,])
    with open(filename + '_overlap_frm'+ traintype + '.csv', 'w') as writeFile:
        writer = csv.writer(writeFile)
        writer.writerow(np.zeros(x.shape[1]))
        writer.writerows(fts)
    writeFile.close()
    return fts[:,0]

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
# also pass a mean and std
def standardize(filename, m, s):
    temp = np.genfromtxt(filename+".csv", delimiter=",")
    ft = np.empty(shape=[temp.shape[0]-1,temp.shape[1]])
    colnames = np.empty(shape=[9])
    with open(filename+'.csv', 'rb') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        counter = -1
        for row in reader:
            if counter >= 0:
                ft[counter, :] = row
            else:
                colnames = row
            counter = counter + 1
    csvfile.close()
    for col in range(1,9):
        ft[:,col] = (ft[:,col] - m[col-1]) / s[col-1]
    with open(filename + '_stdz_frm_'+ traintype + '.csv', 'w') as writeFile:
        writer = csv.writer(writeFile)
        writer.writerow(colnames)
        writer.writerows(ft)
    writeFile.close()
    return

# train model on only subjects that overlap between train and test set
# test model on only subjects also in train set
def runmodeloverlap2(model, traintype, testtype, NFEATS):
    # define variables

    # find overlap
    trainsubj = np.genfromtxt('y'+str(trainsuffix)+filename1+'.csv', delimiter=',')[1:,0]
    testsubj = np.genfromtxt('y'+str(testsuffix)+filename1+'.csv', delimiter=',')[1:,0]

    overlapsubj = list(set(trainsubj) & set(testsubj))
    print str(len(overlapsubj)) + " overlap subjects between " + traintype + " and " + testtype
    print overlapsubj

    # create gps file
    rows = zip(range(len(overlapsubj)), overlapsubj)
    with open('gps_overlap_'+ traintype + '_' + testtype + '.csv', 'w') as writeFile:
        writer = csv.writer(writeFile)
        writer.writerow(['index', 'subject_id'])
        writer.writerows(rows)
    writeFile.close()

    train_y = subsetfilesubj('y'+str(trainsuffix)+filename1, overlapsubj)
    test_y = subsetfilesubj('y'+str(testsuffix)+filename1, overlapsubj)
    subjs = subsetfilesubjSUBJS('y'+str(testsuffix)+filename1, overlapsubj)

    # define variables
    np.save("test_obsv.npy", test_y)
    if model == "topfeatures":
        train_x = strip(subsetxfile('x'+str(trainsuffix)+filename2+'_overlap_frm'+ testtype +'.csv', featrank[0:NFEATS]))
        _temp = subsetfilesubj('x'+str(testsuffix)+filename2, overlapsubj)
        test_x = strip(subsetxfile('x'+str(testsuffix) +filename2+ '_overlap_frm'+ traintype +'.csv', featrank[0:NFEATS]))
    elif model == "baseline":
        pass
    else:
        print "model parameter not recognized"
    # # debug variables
    # print testtype
    # print test_x[0:9,:]
    # print test_y[0:9]
    # print traintype
    # print train_x[0:9,:]
    # print train_y[0:9]


    # create model and start predictions
    if model == "topfeatures":
        mf = LogisticRegression(solver='liblinear') # penalty = 'l1', C = 1/LAMBDA,
        mf.fit(train_x, train_y)
        test_pred = mf.predict(test_x)
        temp = mf.predict_proba(test_x)
        test_prob = np.reshape(np.array(temp[:,1]), (-1))
    elif model == "baseline":
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

    test_prob = np.reshape(np.array(test_prob), (-1))
    predsbool = np.reshape(np.array(test_pred).astype(bool), (-1))
    obsvsbool = np.reshape(np.array(test_y).astype(bool), (-1))

    # acc = sklearn.metrics.accuracy_score(obsvsbool, predsbool)
    bacc3 = balancedacc(obsvsbool, test_prob, 0.3)
    bacc5 = balancedacc(obsvsbool, test_prob, 0.5)
    bacc7 = balancedacc(obsvsbool, test_prob, 0.7)
    mbacc = maxbacc(obsvsbool, test_prob)
    fpr, tpr, thresholds = sklearn.metrics.roc_curve(obsvsbool, test_prob)
    print ""
    print "for model ", model
    sauc = subjauc(subjs, obsvsbool, test_prob)
    auc = sklearn.metrics.auc(fpr, tpr)
    ll = sklearn.metrics.log_loss(np.squeeze(obsvsbool), np.squeeze(test_prob)) / float(len(test_prob))
    print "balanced accuracy is : ", round(bacc5,4)
    print "auc is : ", round(auc,4)
    print "log loss is : ", '%.4E' % Decimal(ll)
    return [bacc3, bacc5, bacc7, mbacc, fpr, tpr, auc, sauc, ll]


# train model on full train set
# test model on only subjects also in train set
def runmodeloverlap(model, traintype, testtype, NFEATS):
    # define variables
    train_y = stripfile('y'+str(trainsuffix)+filename1+'.csv')

    # find overlap
    trainsubj = np.genfromtxt('y'+str(trainsuffix)+filename1+'.csv', delimiter=',')[1:,0]
    testsubj = np.genfromtxt('y'+str(testsuffix)+filename1+'.csv', delimiter=',')[1:,0]

    overlapsubj = list(set(trainsubj) & set(testsubj))
    print str(len(overlapsubj)) + " overlap subjects between " + traintype + " and " + testtype
    print overlapsubj

    # create gps file
    rows = zip(range(len(overlapsubj)), overlapsubj)
    with open('gps_overlap_'+ traintype + '_' + testtype + '.csv', 'w') as writeFile:
        writer = csv.writer(writeFile)
        writer.writerow(['index', 'subject_id'])
        writer.writerows(rows)
    writeFile.close()

    test_y = subsetfilesubj('y'+str(testsuffix)+filename1, overlapsubj)
    subjs = subsetfilesubjSUBJS('y'+str(testsuffix)+filename1, overlapsubj)

    # define variables
    np.save("test_obsv.npy", test_y)
    if model == "topfeatures":
        train_x = strip(subsetxfile('x'+str(trainsuffix)+filename2+'.csv', featrank[0:NFEATS]))
        _temp = subsetfilesubj('x'+str(testsuffix)+filename2, overlapsubj)
        test_x = strip(subsetxfile('x'+str(testsuffix) +filename2+ '_overlap_frm'+ traintype +'.csv', featrank[0:NFEATS]))
    elif model == "baseline":
        pass
    else:
        print "model parameter not recognized"
    # # debug variables
    # print testtype
    # print test_x[0:9,:]
    # print test_y[0:9]
    # print traintype
    # print train_x[0:9,:]
    # print train_y[0:9]


    # create model and start predictions
    if model == "topfeatures":
        mf = LogisticRegression(solver='liblinear') # penalty = 'l1', C = 1/LAMBDA,
        mf.fit(train_x, train_y)
        test_pred = mf.predict(test_x)
        temp = mf.predict_proba(test_x)
        test_prob = np.reshape(np.array(temp[:,1]), (-1))
    elif model == "baseline":
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

    test_prob = np.reshape(np.array(test_prob), (-1))
    predsbool = np.reshape(np.array(test_pred).astype(bool), (-1))
    obsvsbool = np.reshape(np.array(test_y).astype(bool), (-1))

    # acc = sklearn.metrics.accuracy_score(obsvsbool, predsbool)
    bacc3 = balancedacc(obsvsbool, test_prob, 0.3)
    bacc5 = balancedacc(obsvsbool, test_prob, 0.5)
    bacc7 = balancedacc(obsvsbool, test_prob, 0.7)
    mbacc = maxbacc(obsvsbool, test_prob)
    fpr, tpr, thresholds = sklearn.metrics.roc_curve(obsvsbool, test_prob)
    print ""
    print "for model ", model
    sauc = subjauc(subjs, obsvsbool, test_prob)
    auc = sklearn.metrics.auc(fpr, tpr)
    ll = sklearn.metrics.log_loss(np.squeeze(obsvsbool), np.squeeze(test_prob)) / float(len(test_prob))
    print "balanced accuracy is : ", round(bacc5,4)
    print "auc is : ", round(auc,4)
    print "log loss is : ", '%.4E' % Decimal(ll)
    return [bacc3, bacc5, bacc7, mbacc, fpr, tpr, auc, sauc, ll]

def runmodel(model, traintype, testtype, NFEATS):
    # define variables
    train_y = stripfile('y'+str(trainsuffix)+filename1+'.csv')
    test_y = stripfile('y'+str(testsuffix)+filename1+'.csv')
    subjs = np.genfromtxt('y'+str(testsuffix)+filename1+'.csv', delimiter=",")[1:,0]
    np.save("test_obsv.npy", test_y)
    if model == "topfeatures":
        train_x = strip(subsetxfile('x'+str(trainsuffix) + filename2 +'.csv', featrank[0:NFEATS]))
        test_x = strip(subsetxfile('x'+str(testsuffix) + filename2 +'.csv', featrank[0:NFEATS]))
    elif model == "baseline":
        pass
    else:
        print "model parameter not recognized"

    # create model and start predictions
    if model == "topfeatures":
        mf = LogisticRegression(solver='liblinear', penalty = 'l1', C = 1/LAMBDA)
        mf.fit(train_x, train_y)
        test_pred = mf.predict(test_x)
        temp = mf.predict_proba(test_x)
        test_prob = np.reshape(np.array(temp[:,1]), (-1))
    elif model == "baseline":
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

    test_prob = np.reshape(np.array(test_prob), (-1))
    predsbool = np.reshape(np.array(test_pred).astype(bool), (-1))
    obsvsbool = np.reshape(np.array(test_y).astype(bool), (-1))

    bacc3 = balancedacc(obsvsbool, test_prob, 0.3)
    bacc5 = balancedacc(obsvsbool, test_prob, 0.5)
    bacc7 = balancedacc(obsvsbool, test_prob, 0.7)
    mbacc = maxbacc(obsvsbool, test_prob)
    fpr, tpr, thresholds = sklearn.metrics.roc_curve(obsvsbool, test_prob)
    print ""
    print "for model ", model
    sauc = subjauc(subjs, obsvsbool, test_prob)
    auc = sklearn.metrics.auc(fpr, tpr)
    ll = sklearn.metrics.log_loss(np.squeeze(obsvsbool), np.squeeze(test_prob)) / float(len(test_prob))
    print "balanced accuracy is : ", round(bacc5,4)
    print "auc is : ", round(auc,4)
    print "log loss is : ", '%.4E' % Decimal(ll)
    return [bacc3, bacc5, bacc7, mbacc, fpr, tpr, auc, sauc, ll]

models =  ["topfeatures"] # ["baseline", "topfeatures", "topfeatures" ]
NFEATSs = [len(featrank)] # [0, specificNFEATS, len(featrank)]

# filename1 = '_normed_standardized_Actual_EV0.5' # to load y
# filename2 = '_normed_standardized_Actual_EV0.5_interactions' # to load x

filename1 = '_normed_standardized_Actual_EV0.5' # to load y
filename2 = '_normed_standardized_Actual_EV0.5_interactions' # to load x
# filename2 = '_normed_standardized_Actual_EV0.5_interactions_stdz' # to load x

traintype = 'random'
testtype = '1block'

if traintype == "1block":
    trainsuffix = "_1block"
elif traintype == "random":
    trainsuffix = "_random"

if testtype == "3block":
    testsuffix = "_3block"
elif testtype == "1block":
    testsuffix = "_1block"
elif testtype == "random":
    testsuffix = "_random"


with open('results.csv', 'w') as writeFile:
    writer = csv.writer(writeFile)

    # # overlaps, train and test on overlaps
    # writer.writerow(["overlaps 2"])
    # plt.title('Overlaps 2: Receiver Operating Characteristic'+ ' Fts From '+ ET)
    # for model, NFEATS, color in zip(models, NFEATSs, colors):
    #     [bacc3, bacc5, bacc7, mbacc, fpr, tpr, auc, sauc, ll] = runmodeloverlap2(model, traintype, testtype, NFEATS)
    #     writer.writerow([model, str(NFEATS), round(bacc3,4), round(bacc5,4),
    #         round(bacc7,4), round(mbacc,4),
    #         round(auc,4),round(sauc,4),'%.4E' % Decimal(ll)])
    #     plt.plot(fpr, tpr, 'b', color=color, label = model + ' AUC = %0.4f' % auc)
    # plt.legend(loc = 'lower right')
    # plt.plot([0, 1], [0, 1],'r--')
    # plt.xlim([0, 1])
    # plt.ylim([0, 1])
    # plt.ylabel('True Positive Rate')
    # plt.xlabel('False Positive Rate')
    # plt.savefig('roc_overlaps_'+ET)
    # plt.show()

    # # overlaps
    # writer.writerow(["overlaps"])
    # plt.title('Overlaps: Receiver Operating Characteristic'+ ' Fts From '+ ET)
    # for model, NFEATS, color in zip(models, NFEATSs, colors):
    #     [bacc3, bacc5, bacc7, mbacc, fpr, tpr, auc, sauc, ll] = runmodeloverlap(model, traintype, testtype, NFEATS)
    #     writer.writerow([model, str(NFEATS), round(bacc3,4), round(bacc5,4),
    #         round(bacc7,4), round(mbacc,4),
    #         round(auc,4),round(sauc,4),'%.4E' % Decimal(ll)])
    #     plt.plot(fpr, tpr, 'b', color=color, label = model + ' AUC = %0.4f' % auc)
    # plt.legend(loc = 'lower right')
    # plt.plot([0, 1], [0, 1],'r--')
    # plt.xlim([0, 1])
    # plt.ylim([0, 1])
    # plt.ylabel('True Positive Rate')
    # plt.xlabel('False Positive Rate')
    # plt.savefig('roc_overlaps_'+ET)
    # plt.show()

    LAMBDA = 0.1

    # all subjects
    writer.writerow(["all test subjects"])
    writer.writerow([LAMBDA])
    writer.writerow(["model", "number of features", "bacc3","bacc5","bacc7", "mbacc", "auc", "sauc", "ll"])
    #plt.title('All Subjects: Receiver Operating Characteristic'+ ' Fts From '+ ET)
    for model, NFEATS, color in zip(models, NFEATSs, colors):
        [bacc3, bacc5, bacc7, mbacc, fpr, tpr, auc, sauc, ll] = runmodel(model, traintype, testtype, NFEATS)
        writer.writerow([model, str(NFEATS), round(bacc3,4), round(bacc5,4),
            round(bacc7,4), round(mbacc,4),
            round(auc,4),round(sauc,4),'%.4E' % Decimal(ll)])
        #plt.plot(fpr, tpr, 'b', color=color, label = model + ' AUC = %0.4f' % auc)
    # plt.legend(loc = 'lower right')
    # plt.plot([0, 1], [0, 1],'r--')
    # plt.xlim([0, 1])
    # plt.ylim([0, 1])
    # plt.ylabel('True Positive Rate')
    # plt.xlabel('False Positive Rate')
    # plt.savefig('roc_allsubj_'+ET)
    # plt.show()

    LAMBDA = 1.0

    # all subjects
    writer.writerow(["all test subjects"])
    writer.writerow([LAMBDA])
    writer.writerow(["model", "number of features", "bacc3","bacc5","bacc7", "mbacc", "auc", "sauc", "ll"])
    #plt.title('All Subjects: Receiver Operating Characteristic'+ ' Fts From '+ ET)
    for model, NFEATS, color in zip(models, NFEATSs, colors):
        [bacc3, bacc5, bacc7, mbacc, fpr, tpr, auc, sauc, ll] = runmodel(model, traintype, testtype, NFEATS)
        writer.writerow([model, str(NFEATS), round(bacc3,4), round(bacc5,4),
            round(bacc7,4), round(mbacc,4),
            round(auc,4),round(sauc,4),'%.4E' % Decimal(ll)])

    LAMBDA = 10.0

    # all subjects
    writer.writerow(["all test subjects"])
    writer.writerow([LAMBDA])
    writer.writerow(["model", "number of features", "bacc3","bacc5","bacc7", "mbacc", "auc", "sauc", "ll"])
    #plt.title('All Subjects: Receiver Operating Characteristic'+ ' Fts From '+ ET)
    for model, NFEATS, color in zip(models, NFEATSs, colors):
        [bacc3, bacc5, bacc7, mbacc, fpr, tpr, auc, sauc, ll] = runmodel(model, traintype, testtype, NFEATS)
        writer.writerow([model, str(NFEATS), round(bacc3,4), round(bacc5,4),
            round(bacc7,4), round(mbacc,4),
            round(auc,4),round(sauc,4),'%.4E' % Decimal(ll)])
writeFile.close()
