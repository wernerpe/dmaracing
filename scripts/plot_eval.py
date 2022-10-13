import matplotlib.pyplot as plt
import numpy as np


path = 'logs/tri_1v1_2/22_10_12_11_20_19_col_1_ar_0.1_rr_0.0/continuouseval_22_10_12_15_31_45.csv' #'../../supercloud/dmaracing/logs/tri_1v1/22_08_23_11_49_39/continuouseval_22_08_24_16_49_32.csv'

my_data = np.genfromtxt(path, delimiter=',')
keys = my_data[:,0]
mu = my_data[:,1]
sig = my_data[:, 2]
n_races = my_data[:, 3]

els = np.unique(keys)
latest_ratings = []
for el in els:
    idxs = np.where(keys==el)
    mus = mu[idxs]
    latest_ratings.append(mus[-1])

figure = plt.figure()
plt.scatter(keys, mu)
plt.plot(els, np.array(latest_ratings), c = 'r')
plt.xlabel('training step')
plt.ylabel('trueskill')
plt.legend(['eval progression', 'final skill'])
plt.show()
print('')