import numpy as np
import tensorflow as tf
from tensorflow.python.ops import array_ops
# import random
# random.seed(2019)

x = np.genfromtxt('x_random_normed_standardized_Actual_EV0.5_stdz.csv', delimiter=",")
y = np.genfromtxt('y_random_normed_standardized_Actual_EV0.5.csv', delimiter=",")

# get column names
xtest = np.genfromtxt('x_random_normed_standardized_Actual_EV0.5_stdz.csv', delimiter=",", names=True)
xcolnames = xtest.dtype.names
fnames = xcolnames[1:]
nFEATS = len(fnames)
np.save("featnames", fnames)



sf = np.genfromtxt('gps_random.csv', delimiter=",", usecols=((0,1))) # subject file
ORIGsids = np.array(sf[1:,1]) # subject ids

SUFFIX = "nnboot"
UNITS = 5
training_epochs = 3000
LAYERS = 2
REGs = ["l1"]
noises = ["adversarial"]
advs = ["sign"] # can include "nosign"
LAMBDAs = [1e-4]
epsilons = [1e-1] # adversarial noise
nSEEDS = 1


epsilonSGs = [0.2] # smooth grad

nSMOOTHGRAD = 50
nSUBJ =  len(ORIGsids)
nOBSV = y.shape[0] - 1
nTRIALS = nOBSV / nSUBJ

def check_symmetric(a, rtol=1e-05, atol=1e-08):
    return np.allclose(a, a.T, rtol=rtol, atol=atol)

def runseeds(NOISE, adv, LAMBDA, epsilon, REG, epsilonSG):
    hesss = np.zeros(shape=[nSEEDS*nSUBJ*nTRIALS,nFEATS,nFEATS])
    grads = np.zeros(shape=[nSEEDS*nSUBJ*nTRIALS,nFEATS])
    for SEED in range(1,nSEEDS + 1):

        tf.set_random_seed(SEED)

        if SEED == 0:
            sids = ORIGsids
        else:
            np.random.seed(SEED) ; sids = np.random.choice(ORIGsids, len(ORIGsids))

        # return features and targets given a subject ID, x, y
        # [features, targets] = subjXY (sids[0], x, y)
        def subjXY(sid, x, y):
            sidscol = np.array(x[:,0])
            inds = np.where(sidscol == sid)
            features = np.squeeze(x[inds,1:])
            targets = y[inds,1:]
            size = features.shape[0]
            return [features, targets, size]

        xs = np.zeros(shape=[x.shape[0]-1, x.shape[1]-1])
        ys = np.zeros(shape=[y.shape[0]-1, 1])
        xbool = np.zeros(shape=y.shape[0]-1)

        counter = 0
        for i in range(len(sids)):
            [sx, sy, size] = subjXY(sids[i], x, y)
            xs[range(counter, counter+size), :] = sx
            ys[range(counter, counter+size)] = sy
            counter = counter + size

        X = tf.placeholder(tf.float32, shape = [None,xs.shape[1]])
        N = tf.placeholder(tf.float32, shape = [None,xs.shape[1]])
        Ga = tf.placeholder(tf.float32, shape = [None,xs.shape[1]])
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
        #tf.get_default_graph().finalize()

        # m = xs.mean(axis=0)
        # s = xs.std(axis=0)
        # xs = (xs - m)/ s

        with tf.Session() as sess:
            sess.run(init)
            for epoch in range(training_epochs):
                n = np.random.uniform(low=0.0, high=epsilon, size=xs.shape)
                xa = np.squeeze(sess.run(XA, feed_dict={X : xs, Y: 1.0 - ys, N : n}))
                if epoch != training_epochs - 1:
                    if NOISE == "adversarial":
                        _, c = sess.run([optimizer, cost], feed_dict={X: np.concatenate([xs, xa], axis=0), Y: np.concatenate([ys, ys], axis=0)})
                    elif NOISE == "gaussian":
                        ga = np.random.normal(size = xs.shape)
                        xga = np.squeeze(sess.run(XGa, feed_dict={X : xs, Ga: ga, N : n}))
                        _, c= sess.run([optimizer, cost], feed_dict={X: np.concatenate([xs, xga], axis=0), Y: np.concatenate([ys, ys], axis=0)})
                    else:
                        print "encountered error"
                else:
                    if NOISE == "adversarial":
                        _, c, g, h = sess.run([optimizer, cost, G, H], feed_dict={X: np.concatenate([xs, xa], axis=0), Y: np.concatenate([ys, ys], axis=0)})
                        # print h.shape
                        # print h[0,:,:]
                        print "is symmetric? : ", check_symmetric(h[0,:,:])
                        print "is symmetric? : ", check_symmetric(h[211,:,:])
                        print "is symmetric? : ", check_symmetric(h[562,:,:])
                        print "is symmetric? : ", check_symmetric(h[736,:,:])
                    elif NOISE == "gaussian":
                        ga = np.random.normal(size = xs.shape)
                        xga = np.squeeze(sess.run(XGa, feed_dict={X : xs, Ga: ga, N : n}))
                        _, c, g, h = sess.run([optimizer, cost, G, H], feed_dict={X: np.concatenate([xs, xga], axis=0), Y: np.concatenate([ys, ys], axis=0)})
                    else:
                        print "encountered error"
                #print("Epoch: {} - train cost: {}".format(epoch, c))
            _save_path = saver.save(sess, "./model_temp" +"_SEED"+str(SEED)+"_layers"+str(LAYERS)+"_"+REG+"_lambda"+str(LAMBDA)+"_epsilon"+str(epsilon)+"_epsilonSG"+str(epsilonSG)+"_noise"+NOISE+"_sign"+adv+"_units"+str(UNITS)+ ".ckpt")



            saver.restore(sess, "./model_temp" +"_SEED"+str(SEED)+"_layers"+str(LAYERS)+"_"+REG+"_lambda"+str(LAMBDA)+"_epsilon"+str(epsilon)+"_epsilonSG"+str(epsilonSG)+"_noise"+NOISE+"_sign"+adv+"_units"+str(UNITS)+ ".ckpt")

            h = sess.run(H, feed_dict={X: xs, Y: ys})
            g = sess.run(G, feed_dict={X: xs, Y: ys})

            # regular hessian
            h = np.squeeze(h)
            print "hess non-zero entries : ", np.not_equal(h,0.0).sum()
            print "hess shape is : ", h.shape
            # # temph = np.zeros(shape=(31*57, 8, 8))
            # # for s in range(31*57):
            # #     temph[s,:,:] = h[s,:,s,:]
            hesss[(SEED-1)*nSUBJ*nTRIALS: (SEED)*nSUBJ*nTRIALS,:,:] = h[range(nSUBJ*nTRIALS), :, :]
            grads[(SEED-1)*nSUBJ*nTRIALS: (SEED)*nSUBJ*nTRIALS,:] = g
            # print "cumulative hess sum is : ", hesss.mean()
            if NOISE == "adversarial":
                np.save(SUFFIX+"_SEED"+str(SEED)+"_layers"+str(LAYERS)+"_"+REG+"_lambda"+str(LAMBDA)+"_epsilon"+str(epsilon)+"_epsilonSG"+str(epsilonSG)+"_noise"+NOISE+"_sign"+adv+"_units"+str(UNITS)+"_hesss.npy", np.squeeze(h))
                np.save(SUFFIX+"_SEED"+str(SEED)+"_layers"+str(LAYERS)+"_"+REG+"_lambda"+str(LAMBDA)+"_epsilon"+str(epsilon)+"_epsilonSG"+str(epsilonSG)+"_noise"+NOISE+"_sign"+adv+"_units"+str(UNITS)+"_grads.npy", np.squeeze(g))

            elif NOISE == "gaussian":
                np.save(SUFFIX+"_SEED"+str(SEED)+"_layers"+str(LAYERS)+"_"+REG+"_lambda"+str(LAMBDA)+"_epsilon"+str(epsilon)+"_epsilonSG"+str(epsilonSG)+"_noise"+NOISE+"_units"+str(UNITS)+"_hesss.npy", np.squeeze(h))
                np.save(SUFFIX+"_SEED"+str(SEED)+"_layers"+str(LAYERS)+"_"+REG+"_lambda"+str(LAMBDA)+"_epsilon"+str(epsilon)+"_epsilonSG"+str(epsilonSG)+"_noise"+NOISE+"_units"+str(UNITS)+"_grads.npy", np.squeeze(g))

            else:
                print "encountered error"

            # smooth grad hessians

            # smooth grad
            hess_sums = np.zeros(shape = np.squeeze(h).shape)
            for i in range(nSMOOTHGRAD):
                np.random.seed(i); n = np.random.uniform(low=0.0, high=epsilonSG, size=xs.shape)
                np.random.seed(i); ga = np.random.normal(size = xs.shape)
                xga = np.squeeze(sess.run(XGa, feed_dict={X : xs, Ga: ga, N : n}))
                h = sess.run(H, feed_dict={X: xga, Y: ys})
                hess_sums = hess_sums + np.squeeze(h)

            if NOISE == "adversarial":
                np.save(SUFFIX+"_SEED"+str(SEED)+"_"+REG+"_lambda"+str(LAMBDA)+"_epsilon"+str(epsilon)+"_epsilonSG"+str(epsilonSG)+"_noise"+NOISE+"_sign"+adv+"_units"+str(UNITS)+"_points"+ str(nSMOOTHGRAD) + "_hesss_sums.npy", hess_sums)
            elif NOISE == "gaussian":
                np.save(SUFFIX+"_SEED"+str(SEED)+"_"+REG+"_lambda"+str(LAMBDA)+"_epsilon"+str(epsilon)+"_epsilonSG"+str(epsilonSG)+"_noise"+NOISE+"_units"+str(UNITS)+"_points"+ str(nSMOOTHGRAD) +"_hesss_sums.npy", hess_sums)
            else:
                print "encountered error"

    if NOISE == "adversarial":
        np.save(SUFFIX+"_nSEED"+str(nSEEDS)+"_layers"+str(LAYERS)+"_"+REG+"_lambda"+str(LAMBDA)+"_epsilon"+str(epsilon)+"_epsilonSG"+str(epsilonSG)+"_noise"+NOISE+"_sign"+adv+"_units"+str(UNITS)+"_hesss.npy", hesss)
        np.save(SUFFIX+"_nSEED"+str(nSEEDS)+"_layers"+str(LAYERS)+"_"+REG+"_lambda"+str(LAMBDA)+"_epsilon"+str(epsilon)+"_epsilonSG"+str(epsilonSG)+"_noise"+NOISE+"_sign"+adv+"_units"+str(UNITS)+"_grads.npy", grads)

    elif NOISE == "gaussian":
        np.save(SUFFIX+"_nSEED"+str(nSEEDS)+"_layers"+str(LAYERS)+"_"+REG+"_lambda"+str(LAMBDA)+"_epsilon"+str(epsilon)+"_epsilonSG"+str(epsilonSG)+"_noise"+NOISE+"_units"+str(UNITS)+"_hesss.npy", hesss)
        np.save(SUFFIX+"_nSEED"+str(nSEEDS)+"_layers"+str(LAYERS)+"_"+REG+"_lambda"+str(LAMBDA)+"_epsilon"+str(epsilon)+"_epsilonSG"+str(epsilonSG)+"_noise"+NOISE+"_units"+str(UNITS)+"_grads.npy", grads)

    else:
        print "encountered error"

for NOISE in noises:
    for REG in REGs:
        for adv in advs:
            for LAMBDA in LAMBDAs:
                for epsilon in epsilons:
                    for epsilonSG in epsilonSGs:
                        print NOISE, adv, LAMBDA, epsilon, epsilonSG
                        runseeds(NOISE, adv, LAMBDA, epsilon, REG, epsilonSG)
