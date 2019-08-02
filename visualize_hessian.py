import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 8})
import csv
import subprocess

# SUFFIX = "nnboot"
# UNITS = 5
# training_epochs = 3000
# LAYERS = 2
# REGs = ["l1"]
# noises = ["adversarial"]
# advs = ["sign"] # can include "nosign"
# LAMBDAs = [1e-1, 1e-2, 1e-3,1e-4]
# epsilons = [5.0, 1.0, 1e-1, 1e-2, 1e-3]
# nSEEDS = 5

SUFFIX = "nnboot"
UNITS = 5
training_epochs = 3000
LAYERS = 2
REGs = ["l1"]
noises = ["adversarial"]
advs = ["sign"] # can include "nosign"
LAMBDAs = [1e-4]
epsilons = [1e-1]
nSEEDS = 100
nSMOOTHGRAD = 50
epsilonSGs = [0.2]

def check_symmetric(a, rtol=1e-05, atol=1e-08):
    return np.allclose(a, a.T, rtol=rtol, atol=atol)

fnames = np.load("featnames.npy")
nFEATS = len(fnames)

# names = ["Age", "Gender", "Diagnosis",
#          "Current Expected Reward",
#         "Current Gambling Range", "Mood",
#         "Past Not Gamble Reward", "Past Gamble Reward"]

import seaborn as sns

# https://matplotlib.org/3.1.1/gallery/images_contours_and_fields/image_annotated_heatmap.html

def heatmap(data, row_labels, col_labels, ax=None,
            cbar_kw={}, cbarlabel="", **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Parameters
    ----------
    data
        A 2D numpy array of shape (N, M).
    row_labels
        A list or array of length N with the labels for the rows.
    col_labels
        A list or array of length M with the labels for the columns.
    ax
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
        not provided, use current axes or create a new one.  Optional.
    cbar_kw
        A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
    cbarlabel
        The label for the colorbar.  Optional.
    **kwargs
        All other arguments are forwarded to `imshow`.
    """

    if not ax:
        ax = plt.gca()

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    cbar = ax.figure.colorbar(im, orientation= "horizontal", ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=0, va="bottom")

    # We want to show all ticks...
    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_yticks(np.arange(data.shape[0]))
    # ... and label them with the respective list entries.
    ax.set_xticklabels(col_labels)
    ax.set_yticklabels(row_labels)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False,
                   labeltop=True, labelbottom=False)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=-30, ha="right",
             rotation_mode="anchor")

    # Turn spines off and create white grid.
    for edge, spine in ax.spines.items():
        spine.set_visible(False)

    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar


def annotate_heatmap(im, data=None, valfmt="{x:.2f}",
                     textcolors=["black", "white"],
                     threshold=None, **textkw):
    """
    A function to annotate a heatmap.

    Parameters
    ----------
    im
        The AxesImage to be labeled.
    data
        Data used to annotate.  If None, the image's data is used.  Optional.
    valfmt
        The format of the annotations inside the heatmap.  This should either
        use the string format method, e.g. "$ {x:.2f}", or be a
        `matplotlib.ticker.Formatter`.  Optional.
    textcolors
        A list or array of two color specifications.  The first is used for
        values below a threshold, the second for those above.  Optional.
    threshold
        Value in data units according to which the colors from textcolors are
        applied.  If None (the default) uses the middle of the colormap as
        separation.  Optional.
    **kwargs
        All other arguments are forwarded to each call to `text` used to create
        the text labels.
    """

    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max())/2.

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center",
              verticalalignment="center")
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
            text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
            texts.append(text)

    return texts



def subprocess_cmd(command):
    process = subprocess.Popen(command.split(), stdout=subprocess.PIPE)
    output, error = process.communicate()

# ratio of positives in list l
def posrate(l):
    b = len(l)
    a = 0
    for e in l:
        if e > 0.0:
            a = a + 1
    return float(a) / float(b)

# ratio of negatives in list l
def negrate(l):
    b = len(l)
    a = 0
    for e in l:
        if e < 0.0:
            a = a + 1
    return float(a) / float(b)

def visSG(NOISE, adv, LAMBDA, epsilon, REG, SEED):
    # smooth grads

    fn = SUFFIX+"_SEED"+str(SEED)+"_"+REG+"_lambda"+str(LAMBDA)+"_epsilon"+str(epsilon)+"_epsilonSG"+str(epsilonSG)+"_noise"+NOISE+"_sign"+adv+"_units"+str(UNITS)+"_points"+ str(nSMOOTHGRAD) + "_hesss_sums.npy"
    # fn = SUFFIX+"_nSEED"+str(nSEEDS)+"_"+REG+"_lambda"+str(LAMBDA)+"_epsilon"+str(epsilon)+"_noise"+NOISE+"_sign"+adv+"_units"+str(UNITS)+"_hesss.npy"
    print fn
    hessssums = np.load(fn)
    hesss = hessssums / nSMOOTHGRAD
    print hesss.shape
    for ind in range(hesss.shape[0]):
        hi = hesss[ind,:,:]
        # print "is the hessian symmetric? : ", check_symmetric(hi)

    means = hesss.mean(axis=0)
    # print "are the means symmetric? : ", check_symmetric(means)
    stds = hesss.std(axis=0)

    poss = np.zeros(shape = means.shape)
    negs = np.zeros(shape = means.shape)

    for i in range(nFEATS):
        for j in range(nFEATS):
            v = hesss[:,i,j]
            poss[i,j] = posrate(v)
            negs[i,j] = negrate(v)

    print "are the poss symmetric? : ", check_symmetric(poss)
    print "are the negs symmetric? : ", check_symmetric(negs)

    with open('hessian_smoothgrad_results.csv', 'w') as f:
        fw = csv.writer(f)
        fw.writerow(['means'])
        fw.writerow(fnames)
        fw.writerows(means)
        fw.writerow(['stds'])
        fw.writerow(fnames)
        fw.writerows(stds)
        fw.writerow(['poss'])
        fw.writerow(fnames)
        fw.writerows(poss)
        fw.writerow(['negs'])
        fw.writerow(fnames)
        fw.writerows(negs)
        f.close()

    # b0 = "open hessian_smoothgrad_results.csv"
    # subprocess_cmd(b0)

    # fig = plt.figure()
    # ax1 = fig.add_subplot(211)
    # ax2 = fig.add_subplot(212)

    # cmap = sns.diverging_palette(250, 10, as_cmap=True)
    # _ax = sns.heatmap(means, center=0, annot=True, fmt=".2f",
    #                 yticklabels=fnames , cmap=cmap, ax=ax1) # xticklabels=fnames,
    # ax1.set_title("hessian smooth grad means")
    # _ax = sns.heatmap(stds, center=0, annot=True, fmt=".2f",
    #                  yticklabels=fnames , cmap=cmap, ax=ax2) # xticklabels=fnames,
    # ax2.set_title("hessian smooth grad stds")
    # plt.show()
    return [means, stds, poss, negs]

def visREG(NOISE, adv, LAMBDA, epsilon, REG, SEED):
    # fn = SUFFIX+"_nSEED"+str(nSEEDS)+"_layers"+str(LAYERS)+"_"+REG+"_lambda"+str(LAMBDA)+"_epsilon"+str(epsilon)+"_noise"+NOISE+"_sign"+adv+"_units"+str(UNITS)+"_hesss"
    fn = SUFFIX+"_SEED"+str(SEED)+"_layers"+str(LAYERS)+"_"+REG+"_lambda"+str(LAMBDA)+"_epsilon"+str(epsilon)+"_epsilonSG"+str(epsilonSG)+"_noise"+NOISE+"_sign"+adv+"_units"+str(UNITS)+"_hesss"
    #path = '/Users/jessicahuang/Dropbox/nimh'
    #fn = 'nnboot_nSEED1_l1_lambda0.1_epsilon5.0_noiseadversarial_signsign_units5_hesss.npy'
    #os.chdir(r'/Users/jessicahuang/Dropbox/nimh')
    hesss = np.load(fn+".npy")
    # hesss = np.empty(shape=(10,5,5))

    means = hesss.mean(axis=0)
    stds = hesss.std(axis=0)

    poss = np.zeros(shape = means.shape)
    negs = np.zeros(shape = means.shape)

    for i in range(nFEATS):
        for j in range(nFEATS):
            poss[i,j] = posrate(hesss[i,j])
            negs[i,j] = negrate(hesss[i,j])


    with open('hessian_results.csv', 'w') as f:
        fw = csv.writer(f)
        fw.writerow(['means'])
        fw.writerow(fnames)
        fw.writerows(means)
        fw.writerow(['stds'])
        fw.writerow(fnames)
        fw.writerows(stds)
        fw.writerow(['poss'])
        fw.writerow(fnames)
        fw.writerows(poss)
        fw.writerow(['negs'])
        fw.writerow(fnames)
        fw.writerows(negs)
        f.close()
        f.close()

    # with open(fn+'_stds.csv', 'w') as f:
    #     writer = csv.writer(f)
    #     writer.writerow([NOISE, adv, LAMBDA, epsilon, REG])
    #     writer.writerow(names)
    #     writer.writerows(stds)
    # f.close()

    # with open(fn+'_means.csv', 'w') as f:
    #     writer = csv.writer(f)
    #     writer.writerow([NOISE, adv, LAMBDA, epsilon, REG])
    #     writer.writerow(names)
    #     writer.writerows(means)
    # f.close()

    # b0 = "open hessian_results.csv"
    # subprocess_cmd(b0)
    # b1 = "open " + fn +'_means.csv'
    # subprocess_cmd(b1)
    # b2 = "open " + fn +'_stds.csv'
    # subprocess_cmd(b2)

    # plt.hist(hesss[:,6,6])
    # plt.show()

    return [means, stds, poss, negs]

for NOISE in noises:
    for REG in REGs:
        for adv in advs:
            for LAMBDA in LAMBDAs:
                for epsilon in epsilons:
                    for epsilonSG in epsilonSGs:
                        # positive ratio
                        psgs = np.zeros(shape=(nSEEDS, nFEATS, nFEATS))
                        # negative ratio
                        nsgs = np.zeros(shape=(nSEEDS, nFEATS, nFEATS))
                        # positive ratio minus negative ratio in smooth grads
                        pmnsgs = np.zeros(shape=(nSEEDS, nFEATS, nFEATS))
                        # means
                        msgs = np.zeros(shape=(nSEEDS, nFEATS, nFEATS))
                        for SEED in range(1,nSEEDS+1):
                            print NOISE, adv, LAMBDA, epsilon, SEED
                            # mreg, sreg, preg, nreg = visREG(NOISE, adv, LAMBDA, epsilon, REG, SEED)
                            msg, ssg, psg, nsg = visSG(NOISE, adv, LAMBDA, epsilon, REG, SEED)
                            print "is psg symmetric? : ", check_symmetric(psg)
                            print "is nsg symmetric? : ", check_symmetric(nsg)
                            psgs[SEED - 1, :, :] = psg
                            nsgs[SEED - 1, :, :] = nsg
                            pmnsgs[SEED - 1, :, :] = psg - nsg
                            msgs[SEED - 1, :, :] = msg


                            # fig, axes = plt.subplots(2, 3)
                            # cmap = sns.diverging_palette(250, 10, as_cmap=True)
                            # axes[0,0] = sns.heatmap(mreg, center=0, annot=True, fmt=".2f",
                            #                  yticklabels=fnames , cmap=cmap, ax=axes[0,0]) # xticklabels=fnames,
                            # axes[0,0].set_title("hessian means")
                            # axes[0,1] = sns.heatmap(sreg, center=0, annot=True, fmt=".2f",
                            #                  cmap=cmap, ax=axes[0,1]) # xticklabels=fnames,yticklabels=fnames ,
                            # axes[0,1].set_title("hessian stds")
                            # axes[0,2] = sns.heatmap(preg-nreg, center=0, annot=True, fmt=".2f",
                            #                  cmap=cmap, ax=axes[0,2]) # xticklabels=fnames,yticklabels=fnames ,
                            # axes[0,2].set_title("hessian prevalence")

                            # axes[1,0] = sns.heatmap(msg, center=0, annot=True, fmt=".2f",
                            #                  yticklabels=fnames , cmap=cmap, ax=axes[1,0]) # xticklabels=fnames,
                            # axes[1,0].set_title("hessian smoothgrad means")
                            # axes[1,1] = sns.heatmap(ssg, center=0, annot=True, fmt=".2f",
                            #                  cmap=cmap, ax=axes[1,1]) # xticklabels=fnames,yticklabels=fnames ,
                            # axes[1,1].set_title("hessian smoothgrad stds")
                            # axes[1,2] = sns.heatmap(psg-nsg, center=0, annot=True, fmt=".2f",
                            #                  cmap=cmap, ax=axes[1,2]) # xticklabels=fnames,yticklabels=fnames ,
                            # axes[1,2].set_title("hessian smoothgrad prevalence")
                            # plt.show()

                        cmap = sns.diverging_palette(250, 10, as_cmap=True)
                        ax = sns.heatmap(msgs.mean(axis=0), center=0, annot=True, fmt=".4f",
                                         yticklabels=fnames , cmap=cmap) # xticklabels=fnames,
                        ax.set_title("hessian smoothgrad average Hessian value across " +str(nSEEDS)+ " resamplings")
                        plt.show()

                        cmap = sns.diverging_palette(250, 10, as_cmap=True)
                        ax = sns.heatmap(pmnsgs.mean(axis=0), center=0, annot=True, fmt=".4f",
                                         yticklabels=fnames , cmap=cmap) # xticklabels=fnames,
                        ax.set_title("hessian smoothgrad expected prevalence across " +str(nSEEDS)+ " resamplings")
                        plt.show()

                        cmap = sns.diverging_palette(250, 10, as_cmap=True)
                        ax = sns.heatmap(psgs.mean(axis=0), center = 0.5, annot=True, fmt=".3f",
                                         yticklabels=fnames , cmap=cmap) # xticklabels=fnames,
                        ax.set_title("hessian smoothgrad expected positive ratio across " +str(nSEEDS)+ " resamplings")
                        plt.show()

                        cmap = sns.diverging_palette(250, 10, as_cmap=True)
                        ax = sns.heatmap(nsgs.mean(axis=0), center = 0.5, annot=True, fmt=".3f",
                                         yticklabels=fnames , cmap=cmap) # xticklabels=fnames,
                        ax.set_title("hessian smoothgrad expected negative ratio across " +str(nSEEDS)+ " resamplings")
                        plt.show()

                            # fig, axes = plt.subplots(2, 4)
                            # cmap = sns.diverging_palette(250, 10, as_cmap=True)
                            # axes[0,0] = sns.heatmap(mreg, center=0, annot=True, fmt=".2f",
                            #                  yticklabels=fnames , cmap=cmap, ax=axes[0,0]) # xticklabels=fnames,
                            # axes[0,0].set_title("hessian means")
                            # axes[0,1] = sns.heatmap(sreg, center=0, annot=True, fmt=".2f",
                            #                  cmap=cmap, ax=axes[0,1]) # xticklabels=fnames,yticklabels=fnames ,
                            # axes[0,1].set_title("hessian stds")
                            # axes[0,2] = sns.heatmap(preg, center=0, annot=True, fmt=".2f",
                            #                  cmap=cmap, ax=axes[0,2]) # xticklabels=fnames,yticklabels=fnames ,
                            # axes[0,2].set_title("hessian ratio of positives")
                            # axes[0,3] = sns.heatmap(nreg, center=0, annot=True, fmt=".2f",
                            #                  cmap=cmap, ax=axes[0,3]) # xticklabels=fnames,yticklabels=fnames ,
                            # axes[0,3].set_title("hessian ratio of negatives")

                            # axes[1,0] = sns.heatmap(msg, center=0, annot=True, fmt=".2f",
                            #                  yticklabels=fnames , cmap=cmap, ax=axes[1,0]) # xticklabels=fnames,
                            # axes[1,0].set_title("hessian smoothgrad means")
                            # axes[1,1] = sns.heatmap(ssg, center=0, annot=True, fmt=".2f",
                            #                  cmap=cmap, ax=axes[1,1]) # xticklabels=fnames,yticklabels=fnames ,
                            # axes[1,1].set_title("hessian smoothgrad stds")
                            # axes[1,2] = sns.heatmap(psg, center=0, annot=True, fmt=".2f",
                            #                  cmap=cmap, ax=axes[1,2]) # xticklabels=fnames,yticklabels=fnames ,
                            # axes[1,2].set_title("hessian smoothgrad ratio of positives")
                            # axes[1,3] = sns.heatmap(nsg, center=0, annot=True, fmt=".2f",
                            #                  cmap=cmap, ax=axes[1,3]) # xticklabels=fnames,yticklabels=fnames ,
                            # axes[1,3].set_title("hessian smoothgrad ratio of negatives")




                            # _ax = sns.heatmap(mreg, center=0, annot=True, fmt=".2f",
                            #                 yticklabels=fnames , cmap=cmap, ax=ax1) # xticklabels=fnames,
                            # ax1.set_title("hessian means")
                            # _ax = sns.heatmap(sreg, center=0, annot=True, fmt=".2f",
                            #                   cmap=cmap, ax=ax2) # xticklabels=fnames, yticklabels=fnames ,
                            # ax2.set_title("hessian stds")
                            # _ax = sns.heatmap(msg, center=0, annot=True, fmt=".2f",
                            #                 yticklabels=fnames , cmap=cmap, ax=ax3) # xticklabels=fnames,
                            # ax3.set_title("hessian smooth grad means")
                            # _ax = sns.heatmap(ssg, center=0, annot=True, fmt=".2f",
                            #                   cmap=cmap, ax=ax4) # xticklabels=fnames, yticklabels=fnames ,
                            # ax4.set_title("hessian smooth grad stds")
                            # # mng = plt.get_current_fig_manager()
                            # # mng.resize(*mng.window.maxsize())
                            # plt.show()
