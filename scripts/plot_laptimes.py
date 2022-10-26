from tkinter.font import names
import matplotlib.pyplot as plt
import numpy as np
from tkinter.filedialog import askopenfilename
#path = askopenfilename(filetypes = [("Laptime Log", "*csv")], initialdir= 'logs/tri_single_blr/') 
path = '/home/peter/git/dmaracing/logs/tri_single_blr/laptimes_22_10_25_14_02_34.csv'#'/home/peter/git/dmaracing/logs/tri_single_blr/laptimes_22_10_23_17_56_49.csv'
my_data = np.genfromtxt(path, delimiter=',')
track_names = ['sharp_turns_track_ccw', 'sharp_turns_track_cw', 'orca_ccw', 'orca_cw', 'oversize1_ccw', 'oversize1_cw']

tracks = [[] for n in track_names]
for idx in range(len(my_data)):
    id = int(my_data[idx, 0])
    max_vel = my_data[idx, 1]
    avg_laptime = my_data[idx, 2]
    tracks[id].append([max_vel, avg_laptime])

fig, axs = plt.subplots(nrows=3, ncols=2, figsize = (30,20))
axsflat = [a for ax in axs for a in ax]
laptime_models = []
for name, ax, track_data in zip(track_names, axsflat, tracks):
    ax.set_title(name)
    data = np.array(track_data)
    #sort by maxvel
    idxs = np.argsort(data[:, 0])
    maxvel_sort = data[idxs, 0]
    avg_laptime = data[idxs, 1]
    deg = 2+1
    ax.scatter(maxvel_sort, avg_laptime)
    X = np.concatenate(tuple([maxvel_sort.reshape(-1,1)**idx for idx in range(deg)]), axis = 1)
    Y = avg_laptime.reshape(-1,1)
    w = np.linalg.inv(X.T@X)@X.T@Y

    x_test = np.linspace(1.5, 3.7, 50).reshape(-1,1)
    X_test = np.concatenate(tuple([x_test**idx for idx in range(deg)]), axis = 1)
    y_test = X_test@w
    ax.plot(x_test, y_test, c = 'r')
    #print(name, w)
    laptime_models.append(w)
for m in laptime_models:
    print(m)

total_model = np.array(laptime_models)
#print(laptime_models)
plt.show()