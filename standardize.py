import numpy as np
import csv

# returns a file filename_stdz.csv that's been standardized
#filename = 'x_1block_normed_standardized_Actual_EV0.7'
def standardize(filename):
    temp = np.genfromtxt(filename + '.csv', delimiter=",")
    print temp.shape
    ft = np.empty(shape=[temp.shape[0]-1,temp.shape[1]])
    colnames = np.empty(shape=[temp.shape[1]])
    with open(filename+'.csv', 'rb') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        counter = -1
        for row in reader:
            if counter >= 0:
                ft[counter, :] = row
            else:
                colnames = row
            counter = counter + 1
    # print counter
    # print ft.shape
    # print ft
    # print colnames

    csvfile.close()
    means = []
    stds = []
    for col in range(1,temp.shape[1]):

        m = ft[:,col].mean(axis=0)
        means.append(m)
        s = ft[:,col].std(axis=0)
        stds.append(s)
        if col == 1:
            print ft[:,col]
            print m
            print s
        if s == 0.0 :
            ft[:,col] = (ft[:,col] - m)
        else:
            ft[:,col] = (ft[:,col] - m) / s
    # m = ft.mean(axis=0)[1:]
    # s = ft.std(axis=0)[1:]
    print means
    print stds
    # np.save("mean"+extension+".npy", means)
    # np.save("std"+extension+".npy", stds)

    with open(filename + '_stdz.csv', 'w') as writeFile:
        writer = csv.writer(writeFile)
        writer.writerow(colnames)
        writer.writerows(ft)

    writeFile.close()
    return

# filename = 'x_random_normed_standardized_Actual_EV0.5_interactions'
filename = 'x_random_normed_standardized_Actual_EV0.5'
standardize(filename)

# # list of file prefixes to convert
# extensions = ["", "_3block", "_random"]


# for extension in extensions:
#     fp = "x" + extension
#     temp = np.genfromtxt(fp + '.csv', delimiter=",")
#     print temp.shape
#     ft = np.empty(shape=[temp.shape[0]-1,temp.shape[1]])
#     colnames = np.empty(shape=[temp.shape[1]])
#     with open(fp+'.csv', 'rb') as csvfile:
#         reader = csv.reader(csvfile, delimiter=',')
#         counter = -1
#         for row in reader:
#             if counter >= 0:
#                 ft[counter, :] = row
#             else:
#                 colnames = row
#             counter = counter + 1
#     # print counter
#     # print ft.shape
#     # print ft
#     # print colnames

#     csvfile.close()
#     means = []
#     stds = []
#     for col in range(1,temp.shape[1]):

#         m = ft[:,col].mean(axis=0)
#         means.append(m)
#         s = ft[:,col].std(axis=0)
#         stds.append(s)
#         if col == 1:
#             print ft[:,col]
#             print m
#             print s
#         ft[:,col] = (ft[:,col] - m) / s
#     # m = ft.mean(axis=0)[1:]
#     # s = ft.std(axis=0)[1:]
#     print means
#     print stds
#     np.save("mean"+extension+".npy", means)
#     np.save("std"+extension+".npy", stds)

#     with open(fp + '_stdz.csv', 'w') as writeFile:
#         writer = csv.writer(writeFile)
#         writer.writerow(colnames)
#         writer.writerows(ft)

#     writeFile.close()
#     # np.save(fp+".npy", ft)
#     # print "done with ", fp
#     #np.savetxt(fp + "_stdz.csv", ff, delimiter=",")