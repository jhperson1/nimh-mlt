import numpy as np
import tensorflow as tf
import sklearn.metrics
import csv

# import random
# random.seed(2019)
UNITS = 5
SUFFIX = "nn"
training_epochs = 3000
LAYERS = 2
REG = "l1"
NOISE = "adversarial"
adv = "sign" # can include "nosign"

LAMBDAs = [1e-1]#,1e-1,1.0,10.0] #[1e-1, 1e-2, 1e-3, 1e-4]
epsilons = [0.0]#, 0.0] #[5.0, 1.0, 1e-1, 1e-2, 1e-3]
nSEEDS = 3

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

def run(NOISE, adv, LAMBDA, epsilon, REG, SEED):
    tf.set_random_seed(SEED)

    x = np.genfromtxt('x_random_normed_standardized_Actual_EV0.5_stdz.csv', delimiter=",")
    y = np.genfromtxt('y_random_normed_standardized_Actual_EV0.5.csv', delimiter=",")

    sf = np.genfromtxt('gps_random.csv', delimiter=",", usecols=((0,1)))  # subject file
    ORIGsids = np.array(sf[1:,1]) # subject ids
    n = len(ORIGsids)

    if SEED == 0:
        sids = ORIGsids
    else:
        np.random.seed(SEED) ; sids = np.random.choice(ORIGsids, len(ORIGsids))
    # train_m3 = train_x3.mean(axis=0)
    # train_s3 = train_x3.std(axis=0)
    # train_x3 = (train_x3 - train_m3 )/ train_s3

    # return features and targets given a subject ID, x, y
    # [features, targets] = subjXY (sids[0], x, y)
    def subjXY(sid, x, y):
        sidscol = np.array(x[:,0])
        inds = np.where(sidscol == sid)
        features = np.squeeze(x[inds,1:])
        targets = np.squeeze(y[inds,1:])
        return [features, targets]

    # return all other features and targets given a subject ID, x, y
    # [features, targets] = notsubjXY (sids[0], x, y)
    def notsubjXY(sid, x, y):
        sidscol = np.array(x[:,0])
        inds = np.where(sidscol != sid)
        features = np.squeeze(x[inds,1:])[1:,]
        targets = np.squeeze(y[inds,1:])[1:]
        return [features, targets]

    # newer version

    print "the number of features should be : ", x.shape[1]-1

    X = tf.placeholder(tf.float32, shape = [None,x.shape[1]-1])
    N = tf.placeholder(tf.float32, shape = [None,x.shape[1]-1])
    Ga = tf.placeholder(tf.float32, shape = [None,x.shape[1]-1])
    Y = tf.placeholder(tf.float32, shape = [None, 1])
    if REG == "l2":
        l = tf.contrib.layers.l2_regularizer(scale=LAMBDA)
    elif REG == "l1":
        l = tf.contrib.layers.l1_regularizer(scale=LAMBDA)
    else:
        "parameter 'REG' not recognized"
    if LAYERS == 2:
        H1 = tf.contrib.layers.fully_connected(X,UNITS,activation_fn=tf.nn.sigmoid,weights_regularizer=l)
        H2 = tf.contrib.layers.fully_connected(H1,UNITS,activation_fn=tf.nn.sigmoid,weights_regularizer=l)
        L = tf.contrib.layers.fully_connected(H2,1,activation_fn=None,weights_regularizer=l)
    elif LAYERS == 1:
        H1 = tf.contrib.layers.fully_connected(X,UNITS,activation_fn=tf.nn.sigmoid,weights_regularizer=l)
        L = tf.contrib.layers.fully_connected(H1,1,activation_fn=None,weights_regularizer=l)
    else:
        print "LAYERS parameter not recognized"
    l_loss = tf.losses.get_regularization_loss()
    P = tf.nn.sigmoid(L)
    cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=Y,logits=L)) # cross entropy
    if adv == "sign":
        A = tf.stop_gradient(tf.sign(tf.gradients(cost,X)))
    elif adv == "nosign":
        A = tf.stop_gradient(tf.gradients(cost,X))
    else:
        print "not recognized value of adv"
    XA = tf.stop_gradient(X + N * A)
    XGa = tf.stop_gradient(X + N * Ga)
    optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost+l_loss)
    G = tf.stop_gradient(tf.gradients(L,X))
    H = tf.stop_gradient(tf.reduce_sum(tf.squeeze(tf.hessians(L,X)),axis=2))

    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

    # # old version

    # X = tf.placeholder(tf.float32, shape = [None,x.shape[1]-1])
    # N = tf.placeholder(tf.float32, shape = [None,x.shape[1]-1])
    # Y = tf.placeholder(tf.float32, shape = [None, 1])

    # if REG == "l2":
    #     l = tf.contrib.layers.l2_regularizer(scale=LAMBDA)
    # elif REG == "l1":
    #     l = tf.contrib.layers.l1_regularizer(scale=LAMBDA)
    # else:
    #     "parameter 'REG' not recognized"

    # if LAYERS == 2:
    #     H1 = tf.contrib.layers.fully_connected(X,UNITS,activation_fn=tf.nn.sigmoid,weights_regularizer=l)
    #     H2 = tf.contrib.layers.fully_connected(H1,UNITS,activation_fn=tf.nn.sigmoid,weights_regularizer=l)
    #     L = tf.contrib.layers.fully_connected(H2,1,activation_fn=None,weights_regularizer=l)
    # elif LAYERS == 1:
    #     H1 = tf.contrib.layers.fully_connected(X,UNITS,activation_fn=tf.nn.sigmoid,weights_regularizer=l)
    #     L = tf.contrib.layers.fully_connected(H1,1,activation_fn=None,weights_regularizer=l)
    # else:
    #     print "LAYERS parameter not recognized"
    # G = tf.stop_gradient(tf.gradients(L,X))
    # P = tf.nn.sigmoid(L)
    # l_loss = tf.losses.get_regularization_loss()
    # cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=Y,logits=L)) # cross entropy
    # optimizer = tf.train.AdamOptimizer().minimize(cost+l_loss)
    # if adv == "sign":
    #     A = tf.stop_gradient(tf.sign(tf.gradients(cost,X)))
    # elif adv == "nosign":
    #     A = tf.stop_gradient(tf.gradients(cost,X))
    # else:
    #     print "not recognized value of adv"
    # XA = tf.stop_gradient(X + N * A)
    # init = tf.global_variables_initializer()
    # saver = tf.train.Saver()

    final_test_costs = []
    obsvs = []
    preds = []
    grads = []
    for fold in range(n):
        [test_x, test_y] = subjXY(sids[fold], x, y)
        [train_x, train_y] = notsubjXY(sids[fold], x, y)
        test_y = np.reshape(test_y, (-1,1))
        train_y = np.reshape(train_y, (-1,1))
        print train_x.shape
        print train_y.shape
        # m = train_x.mean(axis=0)
        # s = train_x.std(axis=0)
        # if s == 0.0:
        #     train_x = (train_x - m )
        #     test_x = (test_x - m )
        # else:
        #     train_x = (train_x - m )/ s
        #     test_x = (test_x - m )/ s

        o = test_y
        with tf.Session() as sess:
            sess.run(init)

            # try new version from nn_boot_hessians.py
            for epoch in range(training_epochs):
                n = np.random.uniform(low=0.0, high=epsilon, size=train_x.shape)
                xa = np.squeeze(sess.run(XA, feed_dict={X : train_x, Y: 1.0 - train_y, N : n}))
                if epoch != training_epochs - 1:
                    if NOISE == "adversarial":
                        _, c = sess.run([optimizer, cost], feed_dict={X: np.concatenate([train_x, xa], axis=0), Y: np.concatenate([train_y, train_y], axis=0)})
                    elif NOISE == "gaussian":
                        ga = np.random.normal(size = train_x.shape)
                        xga = np.squeeze(sess.run(XGa, feed_dict={X : train_x, Ga: ga, N : n}))
                        _, c= sess.run([optimizer, cost], feed_dict={X: np.concatenate([train_x, xga], axis=0), Y: np.concatenate([train_y, train_y], axis=0)})
                    else:
                        print "encountered error"
                else:
                    if NOISE == "adversarial":
                        _, c, g = sess.run([optimizer, cost, G], feed_dict={X: np.concatenate([train_x, xa], axis=0), Y: np.concatenate([train_y, train_y], axis=0)})
                    elif NOISE == "gaussian":
                        ga = np.random.normal(size = xs.shape)
                        xga = np.squeeze(sess.run(XGa, feed_dict={X : train_x, Ga: ga, N : n}))
                        _, c, g = sess.run([optimizer, cost, G], feed_dict={X: np.concatenate([train_x, xga], axis=0), Y: np.concatenate([train_y, train_y], axis=0)})
                    else:
                        print "encountered error"
                print("Epoch: {} - train cost: {}".format(epoch, c))
            p = sess.run(P, feed_dict={X: test_x})
            print p

            # # old version
            # for epoch in range(training_epochs):
            #     n = np.random.uniform(low=0.0, high=epsilon, size=train_x.shape)
            #     xa = np.squeeze(sess.run(XA, feed_dict={X : train_x, Y: 1.0 - train_y.reshape((len(train_y), 1)), N : n}))
            #     _, train_c, g = sess.run([optimizer, cost, G], feed_dict={X: np.concatenate([train_x, xa], axis=0), Y: np.concatenate([train_y.reshape((len(train_y), 1)), train_y.reshape((len(train_y), 1))], axis=0)})
            #     test_c, g = sess.run([cost, G], feed_dict={X: test_x, Y: np.transpose([test_y])})
            #     #print("Epoch: {} - train cost: {} - test cost: {}".format(epoch, train_c, test_c))
            #     if epoch == training_epochs - 1:
            #         final_test_costs.append(test_c)
            # p = sess.run(P, feed_dict={X: test_x})
            # print p

            saver.save(sess, "./model"+str(fold+1)+"_"+SUFFIX+"_lambda"+str(LAMBDA)+".ckpt") # checkpoint
        preds.append(p)
        obsvs.append(o)
        grads.append(g)
    np.save(SUFFIX+"_SEED"+str(SEED)+"_layers"+str(LAYERS)+"_lambda"+str(LAMBDA)+"_adv"+str(adv)+"_reg"+str(REG)+"_units"+str(UNITS)+"_pred.npy", preds)
    np.save(SUFFIX+"_SEED"+str(SEED)+"_layers"+str(LAYERS)+"_lambda"+str(LAMBDA)+"_adv"+str(adv)+"_reg"+str(REG)+"_units"+str(UNITS)+"_obsv.npy", obsvs)
    np.save(SUFFIX+"_SEED"+str(SEED)+"_layers"+str(LAYERS)+"_lambda"+str(LAMBDA)+"_adv"+str(adv)+"_reg"+str(REG)+"_units"+str(UNITS)+"_grads.npy", grads)
    np.save(SUFFIX+"_SEED"+str(SEED)+"_layers"+str(LAYERS)+"_lambda"+str(LAMBDA)+"_adv"+str(adv)+"_reg"+str(REG)+"_units"+str(UNITS)+"_testcosts.npy", final_test_costs)

    # preds = np.squeeze(np.load(SUFFIX+"_lambda"+str(LAMBDA)+"_units"+str(UNITS)+"_pred.npy"))
    # obsvs = np.load(SUFFIX+"_lambda"+str(LAMBDA)+"_units"+str(UNITS)+"_obsv.npy")

    f = lambda x : x > 0.5
    predsbool = np.array(map(f, preds))
    totaln = len(predsbool) * len(predsbool[0])
    preds = np.reshape(np.array(preds), [totaln])
    # obsvs = np.squeeze(np.array(obsvs), (-1))
    predsbool = np.reshape(predsbool, [totaln])
    obsvsbool = np.squeeze(np.array(obsvs).astype(bool))
    obsvsbool = np.reshape(obsvsbool, [totaln])

    bacc = balancedacc(obsvsbool, preds, 0.5)

    acc = sklearn.metrics.accuracy_score(obsvsbool, predsbool)
    fpr, tpr, thresholds = sklearn.metrics.roc_curve(obsvsbool, predsbool)
    auc = sklearn.metrics.auc(fpr, tpr)
    # ll = sklearn.metrics.log_loss(np.squeeze(obsvs), np.squeeze(preds)) / totaln
    print SEED, LAMBDA, epsilon
    print sids
    print "balanced accuracy is : ", round(bacc,4)
    print "auc is : ", auc
    # print "log loss is : ", ll

    return [round(bacc,4), auc]#, ll]

# just in case it breaks somewhere
with open('nn_folds_results.csv', 'w') as f:
    w = csv.writer(f)
    results = []
    for SEED in range(nSEEDS):
        for epsilon in epsilons:
            for LAMBDA in LAMBDAs:
                print epsilon, LAMBDA, SEED
                result = run(NOISE, adv, LAMBDA, epsilon, REG, SEED)
                result.extend(["epsilon : " + str(epsilon), "lambda : "+ str(LAMBDA), "SEED : " + str(SEED)])
                w.writerow(result)
                results.append(result)
    f.close()

# all the stuff
with open('nn_folds_results_all.csv', 'w') as f:
    fw = csv.writer(f)
    fw.writerows(results)
    f.close()
