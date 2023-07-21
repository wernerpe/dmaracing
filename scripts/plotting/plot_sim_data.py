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
# plt.rcParams["axes.axisbelow"] = False

from matplotlib.patches import Circle, RegularPolygon
from matplotlib.path import Path
from matplotlib.projections.polar import PolarAxes
from matplotlib.projections import register_projection
from matplotlib.spines import Spine
from matplotlib.transforms import Affine2D


def radar_factory(num_vars, frame='circle'):
    """
    Create a radar chart with `num_vars` axes.

    This function creates a RadarAxes projection and registers it.

    Parameters
    ----------
    num_vars : int
        Number of variables for radar chart.
    frame : {'circle', 'polygon'}
        Shape of frame surrounding axes.

    """
    # calculate evenly-spaced axis angles
    theta = np.linspace(0, 2*np.pi, num_vars, endpoint=False)

    class RadarTransform(PolarAxes.PolarTransform):

        def transform_path_non_affine(self, path):
            # Paths with non-unit interpolation steps correspond to gridlines,
            # in which case we force interpolation (to defeat PolarTransform's
            # autoconversion to circular arcs).
            if path._interpolation_steps > 1:
                path = path.interpolated(num_vars)
            return Path(self.transform(path.vertices), path.codes)

    class RadarAxes(PolarAxes):

        name = 'radar'
        PolarTransform = RadarTransform

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            # rotate plot such that the first axis is at the top
            # self.set_theta_zero_location('N')

        def fill(self, *args, closed=True, **kwargs):
            """Override fill so that line is closed by default"""
            return super().fill(closed=closed, *args, **kwargs)

        def plot(self, *args, **kwargs):
            """Override plot so that line is closed by default"""
            lines = super().plot(*args, **kwargs)
            for line in lines:
                self._close_line(line)

        def _close_line(self, line):
            x, y = line.get_data()
            # FIXME: markers at x[0], y[0] get doubled-up
            if x[0] != x[-1]:
                x = np.append(x, x[0])
                y = np.append(y, y[0])
                line.set_data(x, y)

        def set_varlabels(self, labels, **kwargs):
            _lines, texts = self.set_thetagrids(np.degrees(theta), labels, **kwargs)
            half = (len(texts) - 1) // 2
            for t in texts:
                if 'leader' in t._text or 'Laptime' in t._text or 'Collision' in t._text:
                    t.set_horizontalalignment('left')
                else:  # 'pursuer' in t._text or 'track' in t._text:
                    t.set_horizontalalignment('right')
                # else:
                #     t.set_horizontalalignment('center')
                if 'Wins' in t._text:
                    t.set_verticalalignment('top')
                elif 'track' in t._text or 'Collision' in t._text:
                    t.set_verticalalignment('bottom')

        def _gen_axes_patch(self):
            # The Axes patch must be centered at (0.5, 0.5) and of radius 0.5
            # in axes coordinates.
            if frame == 'circle':
                return Circle((0.5, 0.5), 0.5)
            elif frame == 'polygon':
                return RegularPolygon((0.5, 0.5), num_vars, orientation=-np.pi/num_vars,  # NOTE: added orientation
                                      radius=.5, edgecolor="k")
            else:
                raise ValueError("Unknown value for 'frame': %s" % frame)

        def _gen_axes_spines(self):
            if frame == 'circle':
                return super()._gen_axes_spines()
            elif frame == 'polygon':
                # spine_type must be 'left'/'right'/'top'/'bottom'/'circle'.
                spine = Spine(axes=self,
                              spine_type='circle',
                              path=Path.unit_regular_polygon(num_vars))
                # unit_regular_polygon gives a polygon of radius 1 centered at
                # (0, 0) but we want a polygon of radius 0.5 centered at (0.5,
                # 0.5) in axes coordinates.
                spine.set_transform(Affine2D().scale(.5).rotate(-np.pi/num_vars).translate(.5, .5)  # NOTE: added orientation
                                    + self.transAxes)
                return {'polar': spine}
            else:
                raise ValueError("Unknown value for 'frame': %s" % frame)

    register_projection(RadarAxes)
    return theta

logdir = './logs/tri_2v2_vhc_rear/eval/23_05_06_08_54_00_bilevel_2v2/'

### Experiment 1 - Final checkpoint vs Previous
configs_exp1 = [
    'stats_ckpt1000_vs_ckpt500.csv',
    'stats_ckpt1000_vs_ckpt600.csv',
    'stats_ckpt1000_vs_ckpt700.csv',
    'stats_ckpt1000_vs_ckpt800.csv',
    'stats_ckpt1000_vs_ckpt900.csv',
    'stats_ckpt1000_vs_ckpt1000.csv',
]

exp1_winrate_mean = []
exp1_winrate_stdv = []
exp1_winrate_behind = []
exp1_winrate_ahead = []
exp1_collisions = []
exp1_offtrack = []
exp1_overtakes = []
exp1_laptimes = []
exp1_leadtimes = []
for config_exp1 in configs_exp1:  
    data = np.genfromtxt(logdir + config_exp1, dtype=float, delimiter=',', names=True) 

    ### Winrate
    # did_win = data['teamrank']==0  # Ego=1000
    did_win = data['teamrank']!=0  # Ego=checkpoint
    winrate_mean = did_win.mean()
    winrate_stdv = did_win.std()

    exp1_winrate_mean.append(winrate_mean)
    exp1_winrate_stdv.append(winrate_stdv)

    ### Winrate vs initial position
    initial_teamrank = data['startrank']
    final_teamrank = data['teamrank']
    frac_win_from_behind = np.sum((initial_teamrank>0) & (final_teamrank==0))/np.sum((initial_teamrank>0))
    frac_win_from_ahead = np.sum((initial_teamrank==0) & (final_teamrank==0))/np.sum((initial_teamrank==0))
    exp1_winrate_behind.append(frac_win_from_behind)
    exp1_winrate_ahead.append(frac_win_from_ahead)

    ### Collisions
    avg_collisions = data['collisions'].mean()
    exp1_collisions.append(avg_collisions)

    ### Offtrack
    avg_offtrack = data['offtrack'].mean()
    exp1_offtrack.append(avg_offtrack)

    ### Overtakes
    avg_overtakes = data['overtakes'].mean()
    exp1_overtakes.append(avg_overtakes)

    ### Laptime
    laptimes = data['laptime']
    outliers = laptimes > 1e3
    avg_laptime = laptimes[~outliers].mean()
    exp1_laptimes.append(avg_laptime)

    ### Leadtime
    avg_leadtime = data['leadtime'].mean()
    exp1_leadtimes.append(avg_leadtime)
    

### Experiment 1 - Ego checkpoints vs PPC
configs_exp2 = [
    'stats_ckpt500_vs_ppc.csv',
    'stats_ckpt600_vs_ppc.csv',
    'stats_ckpt700_vs_ppc.csv',
    'stats_ckpt800_vs_ppc.csv',
    'stats_ckpt900_vs_ppc.csv',
    'stats_ckpt1000_vs_ppc.csv',
]

exp2_winrate_mean = []
exp2_winrate_stdv = []
exp2_winrate_behind = []
exp2_winrate_ahead = []
exp2_collisions = []
exp2_offtrack = []
exp2_overtakes = []
exp2_laptimes = []
exp2_leadtimes = []
for config_exp2 in configs_exp2:  
    data = np.genfromtxt(logdir + config_exp2, dtype=float, delimiter=',', names=True) 

    ## Winrate
    # did_win = data['teamrank']!=0  # NOTE: PPC=ego
    did_win = data['teamrank']==0  # NOTE: RL=ego
    winrate_mean = did_win.mean()
    winrate_stdv = did_win.std()

    exp2_winrate_mean.append(winrate_mean)
    exp2_winrate_stdv.append(winrate_stdv)

    ### Winrate vs initial position
    initial_teamrank = data['startrank']
    final_teamrank = data['teamrank']
    frac_win_from_behind = np.sum((initial_teamrank>0) & (final_teamrank==0))/np.sum((initial_teamrank>0))
    frac_win_from_ahead = np.sum((initial_teamrank==0) & (final_teamrank==0))/np.sum((initial_teamrank==0))
    exp2_winrate_behind.append(frac_win_from_behind)
    exp2_winrate_ahead.append(frac_win_from_ahead)

    ### Collisions
    avg_collisions = data['collisions'].mean()
    exp2_collisions.append(avg_collisions)

    ### Offtrack
    avg_offtrack = data['offtrack'].mean()
    exp2_offtrack.append(avg_offtrack)

    ### Overtakes
    avg_overtakes = data['overtakes'].mean()
    exp2_overtakes.append(avg_overtakes)

    ### Laptime
    laptimes = data['laptime']
    outliers = laptimes > 1e3
    avg_laptime = laptimes[~outliers].mean()
    exp2_laptimes.append(avg_laptime)

    ### Leadtime
    avg_leadtime = data['leadtime'].mean()
    exp2_leadtimes.append(avg_leadtime)


logdir = './logs/tri_2v2_vhc_rear/eval/23_05_06_08_54_00_bilevel_2v2_testing/'

### Experiment 3 - Ego checkpoints vs Action centralization
configs_exp3 = [
    'stats_afs1em3_ckpt500_vs_ckpt500_23_05_25_22_03_05.csv',
    'stats_afs1em3_ckpt600_vs_ckpt600_23_05_25_22_03_05.csv',
    'stats_afs1em3_ckpt700_vs_ckpt700_23_05_25_22_03_05.csv',
    'stats_afs1em3_ckpt800_vs_ckpt800_23_05_25_22_03_05.csv',
    'stats_afs1em3_ckpt900_vs_ckpt900_23_05_25_22_03_05.csv',
    'stats_afs1em3_ckpt1000_vs_ckpt1000_23_05_25_22_03_05.csv',
]

exp3_winrate_mean = []
exp3_winrate_stdv = []
exp3_winrate_behind = []
exp3_winrate_ahead = []
exp3_collisions = []
exp3_offtrack = []
exp3_overtakes = []
exp3_laptimes = []
exp3_leadtimes = []
for config_exp3 in configs_exp3:  
    data = np.genfromtxt(logdir + config_exp3, dtype=float, delimiter=',', names=True) 

    ### Winrate
    did_win = data['teamrank']==0
    winrate_mean = did_win.mean()
    winrate_stdv = did_win.std()

    exp3_winrate_mean.append(winrate_mean)
    exp3_winrate_stdv.append(winrate_stdv)

    ### Winrate vs initial position
    initial_teamrank = data['startrank']
    final_teamrank = data['teamrank']
    frac_win_from_behind = np.sum((initial_teamrank>0) & (final_teamrank==0))/np.sum((initial_teamrank>0))
    frac_win_from_ahead = np.sum((initial_teamrank==0) & (final_teamrank==0))/np.sum((initial_teamrank==0))
    exp3_winrate_behind.append(frac_win_from_behind)
    exp3_winrate_ahead.append(frac_win_from_ahead)

    ### Collisions
    avg_collisions = data['collisions'].mean()
    exp3_collisions.append(avg_collisions)

    ### Offtrack
    avg_offtrack = data['offtrack'].mean()
    exp3_offtrack.append(avg_offtrack)

    ### Overtakes
    avg_overtakes = data['overtakes'].mean()
    exp3_overtakes.append(avg_overtakes)

    ### Laptime
    laptimes = data['laptime']
    outliers = laptimes > 1e3
    avg_laptime = laptimes[~outliers].mean()
    exp3_laptimes.append(avg_laptime)

    ### Leadtime
    avg_leadtime = data['leadtime'].mean()
    exp3_leadtimes.append(avg_leadtime)


steps = [500, 600, 700, 800, 900, 1000]
exp1_winrate_mean = np.array(exp1_winrate_mean) * 100.0
exp1_winrate_stdv = np.array(exp1_winrate_stdv) * 100.0
exp2_winrate_mean = np.array(exp2_winrate_mean) * 100.0
exp2_winrate_stdv = np.array(exp2_winrate_stdv) * 100.0
exp3_winrate_mean = np.array(exp3_winrate_mean) * 100.0
exp3_winrate_stdv = np.array(exp3_winrate_stdv) * 100.0

colors = sns.color_palette("husl", 3)
text_size = 16
label_size = 14

with sns.axes_style("darkgrid"):
    # figure, axes = plt.subplots(nrows=1, ncols=2, figsize=(24, 5))
    # figure = plt.figure(figsize=(15, 4))
    # gridspec = figure.add_gridspec(1, 6)
    figure = plt.figure(figsize=(12, 8), constrained_layout = True)
    gridspec = figure.add_gridspec(7, 2)

    alpha = 0.1
    linewidth = 3

    axis = figure.add_subplot(gridspec[:4, :])
    # axis.fill_between(steps, exp1_winrate_mean - exp1_winrate_stdv, exp1_winrate_mean + exp1_winrate_stdv, color=list(colors[0] + (alpha,)))
    axis.plot(steps, exp2_winrate_mean, linestyle='--', color='k', linewidth=linewidth, label=r'Pure Pursuit')
    axis.plot(steps, exp3_winrate_mean, linestyle='-', color=list(colors[1]), linewidth=linewidth, label=r'RL Independent')
    axis.plot(steps, exp1_winrate_mean, linestyle='-', color=list(colors[0]), linewidth=linewidth, label=r'RL Team [at 1k]')
    axis.plot(np.NaN, np.NaN, '-', color='none', label=r'Opponent:')

    axis.set_xlim([500, 1000])
    axis.set_ylim([0.0, 100.0])
    axis.set_xlabel(r"Iteration", fontsize=text_size)
    axis.set_ylabel(r"Win rate [%]", fontsize=text_size)
    axis.set_title(r"Win Rate of RL Team over Training Iterations" + '\n', fontsize=text_size)

    axis.tick_params(axis='both', which='major', labelsize=label_size)

    legend_size = 16
    # offset = (0.5, -0.4)
    offset = (0.5, 0.0)
    axis.legend(*map(reversed, axis.get_legend_handles_labels()), loc='lower center', prop={'size': legend_size}, ncol=7, bbox_to_anchor=offset, handletextpad=0.5)

# # PNG
# fig_name = './figures/sim_eval.png'
# figure.savefig(fig_name, bbox_inches='tight', format='png')

offset = (0.5, -0.3)

### Radar chart data
stats_radar2 = [
    exp2_laptimes,
    exp2_winrate_ahead, 
    exp2_winrate_behind,
    exp2_overtakes,
    exp2_offtrack,
    exp2_collisions,
    # exp2_leadtimes,
]
stats_radar2 = np.stack(stats_radar2, axis=-1)
stats_radar2 = stats_radar2[[1,3,5]]

stats_radar3 = [
    exp3_laptimes,
    exp3_winrate_ahead, 
    exp3_winrate_behind,
    exp3_overtakes,
    exp3_offtrack,
    exp3_collisions,
    # exp3_leadtimes,
]
stats_radar3 = np.stack(stats_radar3, axis=-1)
stats_radar3 = stats_radar3[[1,3,5]]
stats_radar_max = np.concatenate((stats_radar2, stats_radar3), axis=0).max(axis=0)

### Radar chart vs PPC
# stats_radar2 = stats_radar2 / stats_radar2.max(axis=0)
stats_radar2[:, 0] = stats_radar2[:, 0] / stats_radar2[:, 0].max(axis=0)
stats_radar2[:, 3] = stats_radar2[:, 3] / stats_radar2[:, 3].max(axis=0)
stats_radar2[:, 4] = stats_radar2[:, 4] / stats_radar2[:, 4].max(axis=0)
stats_radar2[:, 5] = stats_radar2[:, 5] / stats_radar2[:, 5].max(axis=0)
data_radar = [[r'Laptime', r'Wins as' + '\n' + r'leader', r'Wins as' + '\n' + r'pursuer', r'Overtakes', r'Off-track', r'Collision'],  # , 'Leadtime'],
        (r'Behavior vs Pure Pursuit' + '\n', stats_radar2)]

N = len(data_radar[0])
theta = radar_factory(N, frame='polygon')

title, case_data = data_radar[1]

# fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(projection='radar'))
# fig.subplots_adjust(top=0.85, bottom=0.05)

with sns.axes_style("darkgrid"):
    axis = figure.add_subplot(gridspec[4:, 0], projection='radar')
    axis.plot(np.NaN, np.NaN, '-', color='none', label='')

    axis.set_rgrids([0.0, 0.2, 0.4, 0.6, 0.8])
    axis.set_title(title,  position=(0.5, 1.1), ha='center', fontdict={'fontsize': text_size})

    for idx, d in enumerate(case_data):
        line = axis.plot(theta, d, linewidth=linewidth, color=colors[::-1][idx])
        axis.fill(theta, d,  alpha=0.25, label='_nolegend_', color=colors[::-1][idx])
    axis.set_varlabels(data_radar[0], fontsize=text_size)
    # axis.set_ylim([0.4, 1.1])
    axis.set_ylim([0.0, 1.1])
    axis.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    axis.set_yticklabels(['', '0.2', '', '0.6', '', '1.0'])

    axis.tick_params(axis='both', which='major', labelsize=label_size)

    labels = ('Iteration:', '600', '800', '1000')
    legend = axis.legend(labels, loc='lower center', prop={'size': legend_size}, ncol=7, bbox_to_anchor=offset, handletextpad=0.5)

    # legend_size = 16
    # offset = (0.5, -0.4)
    # axis.legend(*map(reversed, axis.get_legend_handles_labels()), loc='lower center', prop={'size': legend_size}, ncol=7, bbox_to_anchor=offset, handletextpad=0.5)


### Radar chart vs No teaming
# stats_radar3 = stats_radar3 / stats_radar3.max(axis=0)
stats_radar3[:, 0] = stats_radar3[:, 0] / stats_radar3[:, 0].max(axis=0)
stats_radar3[:, 3] = stats_radar3[:, 3] / stats_radar3[:, 3].max(axis=0)
stats_radar3[:, 4] = stats_radar3[:, 4] / stats_radar3[:, 4].max(axis=0)
stats_radar3[:, 5] = stats_radar3[:, 5] / stats_radar3[:, 5].max(axis=0)
data_radar = [[r'Laptime', r'Wins as' + '\n' + r'leader', r'Wins as' + '\n' + r'pursuer', r'Overtakes', r'Off-track', r'Collision'],  # , 'Leadtime'],
        (r'Behavior vs RL Independent' + '\n', stats_radar3)]

N = len(data_radar[0])
theta = radar_factory(N, frame='polygon')

title, case_data = data_radar[1]

# fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(projection='radar'))
# fig.subplots_adjust(top=0.85, bottom=0.05)

# from matplotlib.lines import Line2D
# l = Line2D([0],[0],color="w")

with sns.axes_style("darkgrid"):
    axis = figure.add_subplot(gridspec[4:, 1], projection='radar')
    axis.plot(np.NaN, np.NaN, '-', color='none', label='')

    axis.set_rgrids([0.0, 0.2, 0.4, 0.6, 0.8])
    axis.set_title(title,  position=(0.5, 1.1), ha='center', fontdict={'fontsize': text_size})

    for idx, d in enumerate(case_data):
        line = axis.plot(theta, d, linewidth=linewidth, color=colors[::-1][idx])
        axis.fill(theta, d,  alpha=0.25, label='_nolegend_', color=colors[::-1][idx])
    axis.set_varlabels(data_radar[0], fontsize=text_size)
    # axis.set_ylim([0.4, 1.1])
    axis.set_ylim([0.0, 1.1])
    axis.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    axis.set_yticklabels(['', '0.2', '', '0.6', '', '1.0'])

    axis.tick_params(axis='both', which='major', labelsize=label_size)

    labels = ('Iteration:', '600', '800', '1000')
    legend = axis.legend(labels, loc='lower center', prop={'size': legend_size}, ncol=7, bbox_to_anchor=offset, handletextpad=0.5)

    # handles, labels = axis.get_legend_handles_labels()
    
    # import matplotlib.patches as mpatches
    # label1 = r"Policy"
    # empty_patch1 = mpatches.Patch(color='none', label=label1)

    # handles = [handles[2], handles[1], handles[0], empty_patch1]
    # labels = [labels[2], labels[1], labels[0], label1]

    # legend1 = figure.legend(*map(reversed, (handles, labels)), loc='lower center', prop={'size': legend_size}, ncol=4, bbox_to_anchor=offset, handletextpad=0.5)  # For 2x3 plot
    # figure.add_artist(legend1)


fig_name = './figures/sim_eval_radar.png'
figure.savefig(fig_name, bbox_inches='tight', format='png')

pass