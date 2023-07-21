import numpy as np

from matplotlib import pyplot as plt
from matplotlib.ticker import FormatStrFormatter

import seaborn as sns
sns.set()
sns.set_style("ticks")

import matplotlib
# matplotlib.use('Agg')
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.rcParams["legend.handlelength"] = 1.0
matplotlib.rcParams["legend.columnspacing"] = 1.5


### Team vs MPPI
team_vs_mppi_behind_w_regular = 2
team_vs_mppi_behind_l_regular = 3
team_vs_mppi_behind_w_crash = 0
team_vs_mppi_behind_l_crash = 0

team_vs_mppi_ahead_w_regular = 5
team_vs_mppi_ahead_l_regular = 0
team_vs_mppi_ahead_w_crash = 0
team_vs_mppi_ahead_l_crash = 0

team_vs_mppi_random_w_regular = 7
team_vs_mppi_random_l_regular = 1
team_vs_mppi_random_w_crash = 1
team_vs_mppi_random_l_crash = 1

team_vs_mppi_all_w_regular = team_vs_mppi_behind_w_regular + team_vs_mppi_ahead_w_regular + team_vs_mppi_random_w_regular
team_vs_mppi_all_l_regular = team_vs_mppi_behind_l_regular + team_vs_mppi_ahead_l_regular + team_vs_mppi_random_l_regular
team_vs_mppi_all_w_crash = team_vs_mppi_behind_w_crash + team_vs_mppi_ahead_w_crash + team_vs_mppi_random_w_crash
team_vs_mppi_all_l_crash = team_vs_mppi_behind_l_crash + team_vs_mppi_ahead_l_crash + team_vs_mppi_random_l_crash

team_vs_mppi_ahead_w_all = team_vs_mppi_ahead_w_regular + team_vs_mppi_ahead_w_crash + team_vs_mppi_ahead_l_crash + team_vs_mppi_ahead_l_regular
team_vs_mppi_ahead_w_regular /= team_vs_mppi_ahead_w_all
team_vs_mppi_ahead_w_crash /= team_vs_mppi_ahead_w_all
team_vs_mppi_ahead_l_crash /= team_vs_mppi_ahead_w_all
team_vs_mppi_ahead_l_regular /= team_vs_mppi_ahead_w_all

team_vs_mppi_behind_w_all = team_vs_mppi_behind_w_regular + team_vs_mppi_behind_w_crash + team_vs_mppi_behind_l_crash + team_vs_mppi_behind_l_regular
team_vs_mppi_behind_w_regular /= team_vs_mppi_behind_w_all
team_vs_mppi_behind_w_crash /= team_vs_mppi_behind_w_all
team_vs_mppi_behind_l_crash /= team_vs_mppi_behind_w_all
team_vs_mppi_behind_l_regular /= team_vs_mppi_behind_w_all

team_vs_mppi_random_w_all = team_vs_mppi_random_w_regular + team_vs_mppi_random_w_crash + team_vs_mppi_random_l_crash + team_vs_mppi_random_l_regular
team_vs_mppi_random_w_regular /= team_vs_mppi_random_w_all
team_vs_mppi_random_w_crash /= team_vs_mppi_random_w_all
team_vs_mppi_random_l_crash /= team_vs_mppi_random_w_all
team_vs_mppi_random_l_regular /= team_vs_mppi_random_w_all

team_vs_mppi_all_w_all = team_vs_mppi_ahead_w_all + team_vs_mppi_behind_w_all + team_vs_mppi_random_w_all
team_vs_mppi_all_w_regular /= team_vs_mppi_all_w_all
team_vs_mppi_all_w_crash /= team_vs_mppi_all_w_all
team_vs_mppi_all_l_crash /= team_vs_mppi_all_w_all
team_vs_mppi_all_l_regular /= team_vs_mppi_all_w_all


### Team vs Ego
team_vs_ego_behind_w_regular = 3
team_vs_ego_behind_l_regular = 0
team_vs_ego_behind_w_crash = 1
team_vs_ego_behind_l_crash = 1

team_vs_ego_ahead_w_regular = 5
team_vs_ego_ahead_l_regular = 0
team_vs_ego_ahead_w_crash = 0
team_vs_ego_ahead_l_crash = 0

team_vs_ego_random_w_regular = 8
team_vs_ego_random_l_regular = 1
team_vs_ego_random_w_crash = 1
team_vs_ego_random_l_crash = 0

team_vs_ego_all_w_regular = team_vs_ego_behind_w_regular + team_vs_ego_ahead_w_regular + team_vs_ego_random_w_regular
team_vs_ego_all_l_regular = team_vs_ego_behind_l_regular + team_vs_ego_ahead_l_regular + team_vs_ego_random_l_regular
team_vs_ego_all_w_crash = team_vs_ego_behind_w_crash + team_vs_ego_ahead_w_crash + team_vs_ego_random_w_crash
team_vs_ego_all_l_crash = team_vs_ego_behind_l_crash + team_vs_ego_ahead_l_crash + team_vs_ego_random_l_crash

team_vs_ego_ahead_w_all = team_vs_ego_ahead_w_regular + team_vs_ego_ahead_w_crash + team_vs_ego_ahead_l_crash + team_vs_ego_ahead_l_regular
team_vs_ego_ahead_w_regular /= team_vs_ego_ahead_w_all
team_vs_ego_ahead_w_crash /= team_vs_ego_ahead_w_all
team_vs_ego_ahead_l_crash /= team_vs_ego_ahead_w_all
team_vs_ego_ahead_l_regular /= team_vs_ego_ahead_w_all

team_vs_ego_behind_w_all = team_vs_ego_behind_w_regular + team_vs_ego_behind_w_crash + team_vs_ego_behind_l_crash + team_vs_ego_behind_l_regular
team_vs_ego_behind_w_regular /= team_vs_ego_behind_w_all
team_vs_ego_behind_w_crash /= team_vs_ego_behind_w_all
team_vs_ego_behind_l_crash /= team_vs_ego_behind_w_all
team_vs_ego_behind_l_regular /= team_vs_ego_behind_w_all

team_vs_ego_random_w_all = team_vs_ego_random_w_regular + team_vs_ego_random_w_crash + team_vs_ego_random_l_crash + team_vs_ego_random_l_regular
team_vs_ego_random_w_regular /= team_vs_ego_random_w_all
team_vs_ego_random_w_crash /= team_vs_ego_random_w_all
team_vs_ego_random_l_crash /= team_vs_ego_random_w_all
team_vs_ego_random_l_regular /= team_vs_ego_random_w_all

team_vs_ego_all_w_all = team_vs_ego_ahead_w_all + team_vs_ego_behind_w_all + team_vs_ego_random_w_all
team_vs_ego_all_w_regular /= team_vs_ego_all_w_all
team_vs_ego_all_w_crash /= team_vs_ego_all_w_all
team_vs_ego_all_l_crash /= team_vs_ego_all_w_all
team_vs_ego_all_l_regular /= team_vs_ego_all_w_all



ind = np.arange(4)

species = (
    "Front",
    "Back",
    "Random",
    "Total",
)
# weight_counts = {
#     "1": np.array([team_vs_mppi_ahead_w_regular, team_vs_mppi_behind_w_regular, team_vs_mppi_random_w_regular, team_vs_mppi_all_w_regular]),
#     "2": np.array([team_vs_mppi_ahead_w_crash, team_vs_mppi_behind_w_crash, team_vs_mppi_random_w_crash, team_vs_mppi_all_w_crash]),
#     "3": np.array([team_vs_mppi_ahead_l_crash, team_vs_mppi_behind_l_crash, team_vs_mppi_random_l_crash, team_vs_mppi_all_l_crash]),
#     "4": np.array([team_vs_mppi_ahead_l_regular, team_vs_mppi_behind_l_regular, team_vs_mppi_random_l_regular, team_vs_mppi_all_l_regular]),
# }
weight_counts_mppi = {
    "RL Team    vs.": np.array([team_vs_mppi_ahead_w_regular, team_vs_mppi_behind_w_regular, team_vs_mppi_random_w_regular, team_vs_mppi_all_w_regular]),
    "A": np.array([team_vs_mppi_ahead_w_crash, team_vs_mppi_behind_w_crash, team_vs_mppi_random_w_crash, team_vs_mppi_all_w_crash]),
    "B": np.array([team_vs_mppi_ahead_l_crash, team_vs_mppi_behind_l_crash, team_vs_mppi_random_l_crash, team_vs_mppi_all_l_crash]),
    "MPPI": np.array([team_vs_mppi_ahead_l_regular, team_vs_mppi_behind_l_regular, team_vs_mppi_random_l_regular, team_vs_mppi_all_l_regular]),
}
weight_counts_ego = {
    "C": np.array([team_vs_ego_ahead_w_regular, team_vs_ego_behind_w_regular, team_vs_ego_random_w_regular, team_vs_ego_all_w_regular]),
    "D": np.array([team_vs_ego_ahead_w_crash, team_vs_ego_behind_w_crash, team_vs_ego_random_w_crash, team_vs_ego_all_w_crash]),
    "E": np.array([team_vs_ego_ahead_l_crash, team_vs_ego_behind_l_crash, team_vs_ego_random_l_crash, team_vs_ego_all_l_crash]),
    "RL Ego": np.array([team_vs_ego_ahead_l_regular, team_vs_ego_behind_l_regular, team_vs_ego_random_l_regular, team_vs_ego_all_l_regular]),
}
width = 0.4  #0.5
patterns = ('', '//', '//', '')
colors_1 = ('r', 'r', 'b', 'b')
colors_2 = ('r', 'r', 'g', 'g')
text_size = 16
legend_size = 16
offset = (0.5, -0.37)
label_size = 14

fig, ax = plt.subplots(figsize=(6, 4))
bottom1 = np.zeros(4)
bottom2 = np.zeros(4)

for idx, ((label_mppi, weight_count_mppi), (label_ego, weight_count_ego)) in enumerate(zip(weight_counts_mppi.items(), weight_counts_ego.items())):
    if idx==1 or idx==2:
        label_mppi = None
        label_ego = None
    if idx==0:
        label_ego = None
    p1 = ax.bar(ind - 1.05*width/2, 100*weight_count_mppi, width, label=label_mppi, bottom=bottom1, hatch=patterns[idx], color=colors_1[idx])
    bottom1 += 100*weight_count_mppi
    p2 = ax.bar(ind + 1.05*width/2, 100*weight_count_ego, width, label=label_ego, bottom=bottom2, hatch=patterns[idx], color=colors_2[idx])
    bottom2 += 100*weight_count_ego
# ax.bar(np.NaN, np.NaN, width, label=r'Crashed', hatch=patterns[idx], color='w')  # [0.5, 0.5, 0.5, 0.5])

ax.set_title("Win rates on hardware", fontsize=text_size)
# ax.legend(loc="upper right")
ax.legend(loc='lower center', prop={'size': legend_size}, ncol=7, bbox_to_anchor=offset, handletextpad=0.5)

ax.set_xticks(ind)
ax.set_xticklabels(species)

ax.tick_params(axis='both', which='major', labelsize=label_size)

ax.set_xlabel(r"Initial position of RL Team", fontsize=text_size)
ax.set_ylabel(r"Win rate [%]", fontsize=text_size)

fig_name = './figures/hw_rank_data.png'
fig.savefig(fig_name, bbox_inches='tight', format='png')