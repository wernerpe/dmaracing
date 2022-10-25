import matplotlib.pyplot as plt
import numpy as np
from tkinter.filedialog import askopenfilename

def make_radar_chart(ax, name, stats, label = '', color = 'b', attribute_labels = None,
                     plot_markers = None, plot_str_markers= True):

    if plot_markers is None:
        max = np.max(stats)
        plot_markers = np.linspace(0, 1, 9)
    labels = np.array(attribute_labels)

    angles = np.linspace(0, 2*np.pi, len(labels), endpoint=False)
    stats = np.concatenate((stats,[stats[0]]))
    angles = np.concatenate((angles,[angles[0]]))

    if not str(ax._projection_init[0])[-11:-6] == 'Polar':
        return NotImplementedError
    ax.plot(angles, stats, 'o-', linewidth=2, c = color, label=label)
    ax.fill(angles, stats, alpha=0.25, c = color)
    ax.set_thetagrids(angles[:-1] * 180/np.pi, labels)
    if plot_str_markers:
        plt.yticks(plot_markers)
    ax.set_title(name)
    ax.grid(True)

    #fig.savefig("static/images/%s.png" % name)

    return ax

#path = 'logs/tri_1v1_2/22_10_12_11_20_19_col_1_ar_0.1_rr_0.0/continuouseval_22_10_18_10_46_55.csv' #'../../supercloud/dmaracing/logs/tri_1v1/22_08_23_11_49_39/continuouseval_22_08_24_16_49_32.csv'
path = askopenfilename(filetypes = [("Continuous Eval Log", "*csv")], initialdir= 'logs/tri_1v1_2/') 

my_data = np.genfromtxt(path, delimiter=',')
keys = my_data[:,0]
mu = my_data[:,1]
sig = my_data[:, 2]
n_races = my_data[:, 3]
n_overtakes = my_data[:, 4]
n_collisions = my_data[:, 5]
frac_ego_offtrack = my_data[:, 6]
frac_ado_offtrack = my_data[:, 7]
frac_win_from_behind = my_data[:, 8]
fraction_of_race_led = my_data[:, 9]

els = np.unique(keys)
latest_ratings = []
policy_n_overtakes = []
policy_n_collisions = []
policy_frac_ego_offtrack = []
policy_frac_ado_offtrack = []
policy_frac_win_from_behind = []
policy_fraction_of_race_led = []
for el in els:
    idxs = np.where(keys==el)
    idx2 = np.where(n_overtakes != -1)
    idxs2 = np.intersect1d(idxs, idx2)

    mus = mu[idxs]
    latest_ratings.append(mus[-1])
    policy_n_overtakes.append(n_overtakes[idxs2])
    policy_n_collisions.append(n_collisions[idxs2])
    policy_frac_ego_offtrack.append(frac_ego_offtrack[idxs2])
    policy_frac_ado_offtrack.append(frac_ado_offtrack[idxs2])
    policy_frac_win_from_behind.append(frac_win_from_behind[idxs2])
    policy_fraction_of_race_led.append(fraction_of_race_led[idxs2])

stats = [policy_n_overtakes,
        policy_n_collisions,
        policy_frac_ego_offtrack,
        policy_frac_ado_offtrack,
        policy_frac_win_from_behind,
        policy_fraction_of_race_led]

stat_keys = ['n_overtakes',
             'n_collisions',
             'frac_ego_offtrack',
             'frac_ado_offtrack',
             'frac_win_from_behind',
             'fraction_of_race_led']

figure = plt.figure()
plt.scatter(keys, mu)
plt.plot(els, np.array(latest_ratings), c = 'r')
plt.xlabel('training step')
plt.ylabel('trueskill')
plt.legend(['eval progression', 'final skill'])

fig, ax_grid = plt.subplots(nrows=3, ncols = 2, figsize = (15, 10))
axs = ax_grid[:,0].tolist()+ax_grid[:,1].tolist()

for idx, ax in enumerate(axs):
    data_over_policies = stats[idx]
    mean_policy_stat = [np.mean(s) for s in data_over_policies]
    name = stat_keys[idx]
    ax.set_title(name)
    ax.plot(els, mean_policy_stat, 'r')
    ax.set_xlabel('training steps')

spider_keys = ['overtaking', 'collisions x 0.02', 'ego offtrack', 'ado offtrack', 'win from behind', 'fraction of race led']
policy_profiles = []
for idx, evaluations in enumerate(els):
    pol_profile = []
    for s_idx,stat in enumerate(stats):
        if 'collisions' in spider_keys[s_idx]:
            pol_profile.append(np.mean(stat[idx])*0.02)
        else:
            pol_profile.append(np.mean(stat[idx]))

    policy_profiles.append(pol_profile)
fig = plt.figure()
ax = fig.add_subplot(111, polar=True)
colors = ['r', 'g', 'b', 'k']
for col, driver_idx in zip(colors,[0, 4, 8, -1]) :
    art = make_radar_chart(ax,'drivers ', policy_profiles[driver_idx],'iter ' +str(els[driver_idx]), col, spider_keys)
ax.legend()
#mean_policy_stats = [np.mean(s) for s in data_over_policies]
plt.show()