import numpy as np
import csv


filenames = ['x_3block_normed_standardized_Actual_EV0.5']
for filename in filenames:
# filename = 'x_1block_normed_EV0.5'
# filename = 'x_1block_normed_standardized_Actual_EV0.5'
    names = ['age', 'gender', 'diagnosis', 'mood',
          'current expected reward', 'current gambling range', 'current indicator',
          'past rewards', 'past reward prediction error']
    nf = len(names)
    # has subject_id column and all columns in names
    x = np.genfromtxt(filename+ '.csv', delimiter=",")
    x2 = np.zeros(shape=(x.shape[0], 1 + nf + nf * (nf-1) / 2))
    # align subject ID column
    x2[:,range(nf+1)] = x[:,range(nf+1)]
    # print x[:,0]
    # print x2[:,0]
    counter = nf+1
    for i in range(nf):
        for j in range(i+1, nf):
            v = x[:, i+1] * x[:, j+1]
            x2[:,counter] = v
            names.append(names[i] + ' * ' + names[j])
            counter = counter + 1
            # print x2[:,0]
    print "done with file : ", filename
    np.savetxt(filename+'_interactions.csv', x2, delimiter=",")

    np.save(filename+'_interactions_names.npy', names)
# with open('x_stdz_interactions.csv', 'w') as f:
#     w = f.writer()
#     w.writerows(x2)
#     w.close()

