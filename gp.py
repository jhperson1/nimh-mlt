import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from scipy.stats.stats import pearsonr

# data = pd.read_csv("10data_for_jess_random.csv")
# ntrials = 90

data = pd.read_csv("10data_for_jess_random.csv")
ntrials = 90
final_name = "gps_random.csv"

nrows = data.shape[0]
print nrows
print data.shape
data['PredictionError'] = data['Actual'] \
                          - data['Gamble'] * (data['Outcome1Amount'] + data['Outcome2Amount']) / 2 \
                          - (1 - data['Gamble']) * data['CertainAmount']
data['ExpectedGamble'] = (data['Outcome1Amount'] + data['Outcome2Amount']) / 2
data['DiffCertainExpectedGamble'] = data['CertainAmount'] - data['ExpectedGamble']
data['HigherOutcome1'] = data[["Outcome1Amount", "Outcome2Amount"]].max(axis=1)
data['LowerOutcome1'] = data[["Outcome1Amount", "Outcome2Amount"]].min(axis=1)
data['Mood'] = data['Happiness']

# print data.head()

# map subject to all row indices
ID = {}
for i in range(nrows):
    # if i == 3239:
    #     print "is the error here?"
    #     id = data.iloc[i,0]
    #     print "what about here here?"
    #     ID[id] = ID.get(id, []) + [i]
    #     print "here?"
    # else:
    id = int(data.iloc[i,0])
    ID[id] = ID.get(id, []) + [i]
nsubj = len(ID)

# map subject characteristics to strings
def ptype2str(x):
    str = ""
    if x == 1:
        str = "Depressed"
    else:
        str = "Healthy"
    return str

def gender2str(x):
    str = ""
    if x == 1:
        str = "Female"
    else:
        str = "Male"
    return str

# ## Create a dictionary to store all subjects
# subjDF = {}
# subjs = ID.keys()
# for subj in subjs:
#     # subject data frame, string ID, string patient type
#     subjD = data.iloc[ID[subj],:]
#     subjID = str(subjD.loc[:,'subject_id'].iloc[0])
#     subjptype = ptype2str(subjD.loc[:,'ptype'].iloc[0])
#     subjDF[subj] = subjD

# ## Logistic Regression with all features
# featurenames = ['Mood']
# X = data[featurenames].copy()
# #clf = LogisticRegression(random_state=0, solver='lbfgs',
# #                         multi_class='multinomial').fit(X, y)


# ## Data Characteristics get ranges of subject characteristics
# shapss = data.loc[:,'SHAPS']
# sl = min(shapss)
# sh = max(shapss)
# print sl, sh
#
# mfqs = data.loc[:,'MFQ']
# ml = min(mfqs)
# mh = max(mfqs)
# print ml, mh
#
# ages = data.loc[:,'age']
# al = min(ages)
# ah = max(ages)
# print al, ah
#
# genders = data.loc[:,'gender']
# counts = genders.value_counts()
# f = counts[1] / ntrials
# m = counts[2] / ntrials
# print f, m
#
# genders = data.loc[:,'ptype']
# counts = genders.value_counts()
# d = counts[1] / ntrials
# h = counts[2] / ntrials
# print d, h
#
# subjs = ID.keys()
# gps = []
# for subj in subjs:
#     # subject data frame, string ID, string patient type
#     subjD = data.iloc[ID[subj],:]
#     subjID = str(subjD.loc[:,'subject_id'].iloc[0])
#     subjptype = ptype2str(subjD.loc[:,'ptype'].iloc[0])
#     gp = sum(subjD.loc[:,'Gamble']) / subjD.loc[:,'Gamble'].size
#     gps.append(gp)
# hgp = max(gps)
# lgp = min(gps)
# print hgp, lgp

# # Visualizing all probabilities
# subjs = ID.keys()
# c = 0
# gps = []
# for subj in subjs:
#     # subject data frame, string ID, string patient type
#     subjD = data.iloc[ID[subj],:]
#     subjID = str(subjD.loc[:,'subject_id'].iloc[0])
#     subjptype = ptype2str(subjD.loc[:,'ptype'].iloc[0])
#     gp = sum(subjD.loc[:,'Gamble']) / subjD.loc[:,'Gamble'].size
#     gps.append(gp)
# bins = np.linspace(0, 1, 100)
# plt.hist(gps)
# plt.title("Gambling Probabilities")
# plt.xlabel("Gambling Probability")
# plt.show()

# ### Visualizing the Data
# subjs = ID.keys()#[0:5] + ID.keys()[25:30]
# c = 0
# for subj in subjs:
#     # subject data frame, string ID, string patient type
#     subjD = data.iloc[ID[subj],:]
#     subjID = str(subjD.loc[:,'subject_id'].iloc[0])
#     subjptype = ptype2str(subjD.loc[:,'ptype'].iloc[0])
#
#     # plt.plot('time', 'CertainAmount', data=subjD, label="non gamble amount")
#     # plt.plot('time', 'Outcome1Amount', data=subjD, label="gamble amount 1")
#     # plt.plot('time', 'Outcome2Amount', data=subjD, label="gamble amount 2")
#     #plt.plot('time', 'Actual', data=subjD, label="round outcome amount", marker='o')
#     #plt.plot('time', 'PredictionError', data=subjD, label="prediction error", marker='o')
#     #plt.plot('time', 'ExpectedGamble', data=subjD, label="expected gamble amount", marker='o')
#     #plt.plot('time', 'CertainAmount', data=subjD, label="certain amount", marker="o")
#     # plt.plot('time', 'DiffCertainExpectedGamble', data=subjD, label="diff", marker="o")
#     # plt.title(subjID + " " + subjptype)
#     # plt.legend(loc=3)
#     # plt.show()
#
#     bins = np.linspace(0, 1, 50)
#     plt.hist([x for x in subjD.loc[:,'Happiness'] if str(x) != 'nan'], bins)
#     plt.title("Moods of " + subjID + " " + subjptype)
#     plt.show()
#
#     # bins = np.linspace(-10, 10, 50)
#     # plt.hist(subjD.loc[:, "CertainAmount"], bins, alpha=0.5, label="CR")
#     # plt.hist(subjD.loc[:,'HigherOutcome1'], bins, alpha=0.5, label="H")
#     # plt.hist(subjD.loc[:, 'LowerOutcome1'], bins, alpha=0.5, label="L")
#     # plt.legend(loc='upper right')
#     # plt.show()
#     c += 1
#     print c

    # # Happiness
    # h = subjD.loc[:, 'Happiness']
    # xs = np.arange(len(h))
    # yh = np.array(h)
    # yhm = np.isfinite(yh)
    # plt.plot(xs[yhm], yh[yhm], label="happiness", marker='o')
    # plt.plot('time', 'Gamble', data=subjD, label="decision to gamble", marker='o')
    # plt.ylim((-1.5,1.5))
    # #plt.title(subjptype)
    # plt.legend(loc=3)
    # plt.show()



### Research Question 1:
# Can we predict overall gambling probability
# from subject characteristics


## Set Up Dataframe: gps
# every row has subject characteristics &
# global gambling probabilities
gps = data.drop_duplicates(subset = ['subject_id', 'age', 'gender',
                                     'ptype', 'MFQ', 'SHAPS'])
gps = gps.drop(['time', 'CertainAmount',
                'Outcome1Amount', 'Outcome2Amount',
                'Gamble', 'Actual', 'Happiness'], axis = 1)
gps['GamblingProbability'] = 0.0

# index of 'Gamble'
gi = list(data).index('Gamble')
# index of 'GamblingProbability'
gpi = list(gps).index('GamblingProbability')
for i in range(nsubj):
    id = int(gps.iloc[i,0])
    indices = ID[id]
    gambles = data.iloc[indices,gi]
    gps.iloc[i,gpi] = float(sum(gambles))/float(len(gambles))

gps.to_csv(final_name)
print ("saved file ")

# ## Plot data in gps
# xnames = ['age', 'gender', 'ptype', 'MFQ', 'SHAPS']
# # for xname in xnames:
# #     gps.plot(kind='scatter',x=xname,y='GamblingProbability',color='red')
# #     plt.show()

# ## Run linear regressions
# y = gps.iloc[:,gpi]
# X = gps.loc[:,xnames]

# # result = sm.ols(formula="GamblingProbability ~ age + gender + ptype + MFQ + SHAPS", data=gps).fit()
# # print result.params
# # print result.summary()

# ### Linear Regression

# # feature = 'ptype'
# # y = gps.iloc[:,gpi]
# # x = gps.loc[:,feature].reshape((-1, 1))
# # model = LinearRegression()
# # model.fit(x, y)
# # r_sq = model.score(x, y)
# # print('coefficient of determination:', r_sq)
# # print('intercept:', model.intercept_)
# # print('slope:', model.coef_)
# # y_pred = model.intercept_ + model.coef_ * x
# #
# # plt.plot(x, y, '.')
# # plt.plot(x, model.intercept_ + model.coef_ * x, '-')
# # plt.xlabel(feature)
# # plt.ylabel("Gambling Probability")
# # plt.title("GP vs. " + feature + ": r^2 = " + str(r_sq))
# # plt.show()

# ## Pearson Correlation

# feature = 'ptype'
# feature2 = 'SHAPS'
# #y = gps.iloc[:,gpi]
# x1 = gps.loc[:,feature]
# x2 = gps.loc[:,feature2]

# r = pearsonr(x1,x2)
# print (r)

### Research Question 2
# Can we predict subjects' decision to gamble
# from their characteristics
