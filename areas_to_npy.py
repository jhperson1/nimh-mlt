import numpy as np

ET = "random"
filename = '_normed_standardized_Actual_EV'
gamma = 0.5
suffix = "_"+ET

areas = np.load("areas_areas_" + ET + suffix + filename + str(gamma)+ ".npy")
maxes = np.load("areas_maxes_" + ET + suffix + filename + str(gamma)+ ".npy")
# sort indices based on largest to smallest area
ind_a = np.flip(np.argsort(areas))
# sort indices based on largest to smallest max stability
ind_m = np.flip(np.argsort(maxes))

np.save("areas_areas_indices_" + ET + suffix + filename + str(gamma)+ ".npy", ind_a)
np.save("areas_maxes_indices_" + ET + suffix + filename + str(gamma)+ ".npy", ind_m)