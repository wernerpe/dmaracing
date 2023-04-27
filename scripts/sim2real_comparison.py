from dmaracing.utils.rl_helpers import get_bima_ppo_runner
from dmaracing.env.dmar_bilevel import DmarEnvBilevel
from dmaracing.utils.helpers import *
from datetime import date, datetime
import os
import sys
import shutil
import numpy as np
import time
import pickle 
import matplotlib.pyplot as plt 

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def eval():
    env = DmarEnvBilevel(cfg, args)
    runner = get_bima_ppo_runner(env, cfg_train, logdir, args.device)

    #load active policies
    model_paths_hl, model_paths_ll = [], []
    for run_hl, chkpt_hl, run_ll, chkpt_ll in zip(runs_hl, chkpts_hl, runs_ll, chkpts_ll):
        dir_hl, modelnr_hl = get_run(logdir_root, run=run_hl + '/hl_model', chkpt=chkpt_hl)
        model_paths_hl.append("{}/model_{}.pt".format(dir_hl, modelnr_hl))
        print("Loading HL model" + model_paths_hl[-1])
        dir_ll, modelnr_ll = get_run(logdir_root, run=run_ll + '/ll_model', chkpt=chkpt_ll)
        model_paths_ll.append("{}/model_{}.pt".format(dir_ll, modelnr_ll))
        print("Loading LL model" + model_paths_ll[-1])
    runner.load_multi_path(model_paths_hl, model_paths_ll, load_optimizer=True)

    policy_eval_hl = runner.get_inference_policy_hl(device=env.device)
    policy_eval_ll = runner.get_inference_policy_ll(device=env.device)

    run_eval2(env, policy_eval_hl, policy_eval_ll)
    # run_eval_sensitivity(env, policy_eval_hl, policy_eval_ll)
    #run_eval_winrate(env, policy_eval_hl, policy_eval_ll)

def wrap(ang):
    return np.mod(ang + np.pi, 2*np.pi) - np.pi

def run_eval(env, policy_hl, policy_ll):

    obs, _ = env.reset()
    start = 37
    env.states[:,1:, 0:3] += 20000
    env.states[:, 0, 0] = pose_aligned[start, 0]
    env.states[:, 0, 1] = pose_aligned[start, 1]
    env.states[:, 0, 2] = pose_aligned[start, 2]
    env.states[:, 0, 3] = pose_aligned[start, 3]
    env.states[:, 0, 4] = pose_aligned[start, 4]
    env.states[:, 0, 5] = pose_aligned[start, 5]
    env.states[:, 0, -1] = 0

    sim_state = []
    for step in range(start, ll_cmds_aligned.shape[0]-start): #
        #compute actions
        cmds = torch.tensor(ll_cmds_aligned[step, :], device=env.device).view(1,1,2).tile(1,4,1)
        actions_ll = cmds
        actions_ll[..., 0] = ll_cmds_aligned[step, 0] - wrap(env.states[0,0,2].item()) 
        actions_ll[..., 0] = (cmds[..., 0] - env.default_actions[0])/env.action_scales[0]
        actions_ll[..., 1] = (cmds[..., 1] - env.default_actions[1])/env.action_scales[1]
        obs, privileged_obs, rewards, dones, infos = env.step(actions_ll)
        dones *=False
        sim_state.append([env.states[0,0,0].item(),
                           env.states[0,0,1].item(),
                           env.states[0,0,2].item(),
                           env.states[0,0,3].item(),
                           env.states[0,0,4].item(), 
                           env.states[0,0,5].item()])


    sim_state_np = np.array(sim_state)
    x = sim_state_np[:,0]
    y = sim_state_np[:,1]

    #20s
    plt.figure(figsize=(7,7))
    plt.plot(pose_aligned[start:start+200,0], pose_aligned[start:start+200,1], c = 'k', label = 'Hardware')
    plt.plot(x[:200],y[:200], c = 'r', label = 'Simulation')
    plt.scatter(x[0], y[0], c='k', s = 30)
    # plt.scatter(x[49], y[49], c='k', s = 30)
    # plt.scatter(x[99], y[99], c='k', s = 30)
    plt.scatter(x[199], y[199], c='k', s = 30)
    #plt.scatter(pose_aligned[start+49,0], pose_aligned[start+49,1], c='k', s = 30)
    # plt.plot([x[49],pose_aligned[start+49,0]], [y[49],pose_aligned[start+49,1]], linestyle = 'dotted', linewidth = 2, c = 'm', label='Simulation Error 5s')
    # plt.scatter(pose_aligned[start+99,0], pose_aligned[start+99,1], c='k', s = 30)
    #plt.plot([x[99],pose_aligned[start+99,0]], [y[99],pose_aligned[start+99,1]], linestyle = 'dotted', linewidth = 2, c = 'g', label='Simulation Error 10s')
    plt.scatter(pose_aligned[start+199,0], pose_aligned[start+199,1], c='k', s = 30)
    plt.plot([x[199],pose_aligned[start+199,0]], [y[199],pose_aligned[start+199,1]], linestyle = 'dotted', linewidth = 2, c = 'k', label='Simulation Error 20s')
    plt.legend()
    plt.xlabel('x [m]')
    plt.ylabel('y [m]')
    plt.axis('equal')
    print('error norm 200', np.linalg.norm(sim_state_np[199, :2]-pose_aligned[start+199,:2]))
    print('error norm 100', np.linalg.norm(sim_state_np[99,:2]-pose_aligned[start+99,:2]))
    print('error norm 50', np.linalg.norm(sim_state_np[49,:2]-pose_aligned[start+49,:2]))
    plt.show()
    
    num_episodes = 5

def run_eval2(env, policy_hl, policy_ll):

    obs, _ = env.reset()
    start = 37
    env.states[:,1:, 0:3] += 20000
    env.states[:, 0, 0] = pose_aligned[start, 0]
    env.states[:, 0, 1] = pose_aligned[start, 1]
    env.states[:, 0, 2] = pose_aligned[start, 2]
    env.states[:, 0, 3] = pose_aligned[start, 3]
    env.states[:, 0, 4] = pose_aligned[start, 4]
    env.states[:, 0, 5] = pose_aligned[start, 5]
    env.states[:, 0, -1] = 0

    interval = 50

    sim_state = []
    curr_state = []
    for step in range(start, start+100): #
        #compute actions
        cmds = torch.tensor(ll_cmds_aligned[step, :], device=env.device).view(1,1,2).tile(1,4,1)
        actions_ll = cmds
        actions_ll[..., 0] = wrap(ll_cmds_aligned[step, 0] - env.states[0,0,2].item()) 
        actions_ll[..., 0] = (cmds[..., 0] - env.default_actions[0])/env.action_scales[0]
        actions_ll[..., 1] = (cmds[..., 1] - env.default_actions[1])/env.action_scales[1]
        obs, privileged_obs, rewards, dones, infos = env.step(actions_ll)
        dones *=False

        if (step-start+1) % interval ==0:
            env.states[:, 0, 0] = pose_aligned[step, 0]
            env.states[:, 0, 1] = pose_aligned[step, 1]
            env.states[:, 0, 2] = yaw_vals[step]# wrap(pose_aligned[step, 2])
            c = np.cos(yaw_vals[step])
            s = np.sin(yaw_vals[step])
            env.states[:, 0, 3] = c*pose_aligned[step, 3] +s*pose_aligned[step, 4]
            env.states[:, 0, 4] = -s*pose_aligned[step, 3] +c*pose_aligned[step, 4]
            env.states[:, 0, 5] = pose_aligned[step, 5]
            sim_state.append(curr_state)
            curr_state = []
        curr_state.append([env.states[0,0,0].item(),
                           env.states[0,0,1].item(),
                           env.states[0,0,2].item(),
                           env.states[0,0,3].item(),
                           env.states[0,0,4].item(), 
                           env.states[0,0,5].item()])


    #2s
    plt.figure(figsize=(7,7))
    plt.plot(pose_aligned[start:start+200,0], pose_aligned[start:start+200,1], c = 'k', label = 'Hardware')
    for i, sim_s in enumerate(sim_state):
        sim_state_np = np.array(sim_s)
        x = sim_state_np[:,0]
        y = sim_state_np[:,1]
        if i ==0:
            plt.plot(x,y, c = 'r', label = 'Simulation')
        else:
            plt.plot(x,y, c = 'r')
        plt.scatter(x[0], y[0], c='k', s = 30)
        #plt.scatter(x[199], y[199], c='k', s = 30)
        #plt.scatter(pose_aligned[start+199,0], pose_aligned[start+199,1], c='k', s = 30)
        #plt.plot([x[199],pose_aligned[start+199,0]], [y[199],pose_aligned[start+199,1]], linestyle = 'dotted', linewidth = 2, c = 'k', label='Simulation Error 20s')
    plt.legend()
    plt.xlabel('x [m]')
    plt.ylabel('y [m]')
    plt.axis('equal')
        #print('error norm 200', np.linalg.norm(sim_state_np[199, :2]-pose_aligned[start+199,:2]))
    #print('error norm 100', np.linalg.norm(sim_state_np[99,:2]-pose_aligned[start+99,:2]))
    #print('error norm 50', np.linalg.norm(sim_state_np[49,:2]-pose_aligned[start+49,:2]))
    plt.show()


if __name__ == "__main__":
    args = CmdLineArguments()
    args.parse(sys.argv[1:])
    args.device = 'cuda:0'
    args.headless =  False 


    # ### Run information
    exp_name = 'tri_2v2_vel_heading_control'  # 'tri_single_blr_hierarchical'
    timestamp = '23_04_12_17_40_35_bilevel_2v2'  #'23_03_23_11_34_55_bilevel_2v2'  # '23_03_20_19_06_44_bilevel_2v2'  # '23_02_21_17_16_07_bilevel_2v2'  # '23_01_31_14_30_58_bilevel_2v2'  # '23_01_31_11_54_24_bilevel_2v2'
    checkpoint = 1300  # 500  # 1300
    #active policies
    runs_hl = [timestamp, '23_04_12_17_40_35_bilevel_2v2']  # '23_02_22_21_18_03_bilevel_2v2'
    chkpts_hl = [checkpoint, 1300]
    runs_ll = [timestamp, '23_04_12_17_40_35_bilevel_2v2']
    chkpts_ll = [checkpoint, 1300]
    ##policies to populate adversary buffer
    adv_runs = ['23_04_12_17_40_35_bilevel_2v2']
    adv_chkpts = [checkpoint]


    path_cfg = os.getcwd() + '/logs/' + exp_name + '/' + runs_hl[0]
    cfg, cfg_train, logdir_root = getcfg(path_cfg, postfix='', postfix_train='')
    cfg['viewer']['logEvery'] = -1
    cfg['sim']['numEnv'] = 1

    # cfg_train['policy']['numteams'] = 2
    # cfg_train['policy']['teamsize'] = 2
    # cfg['learn']['agent_dropout_prob'] = 0.0
    cfg['learn']['agent_dropout_prob_ini'] = 0.0
    cfg['learn']['agent_dropout_prob_end'] = 0.0
    cfg['learn']['steer_ado_with_PPC'] = True  # False
    # cfg['learn']['resetrand'] = [0.05, 0.05, 0.0, 0.0, 0.0,  0.0, 0.0]
    cfg["sim"]["reset_timeout_only"] = True
    cfg["sim"]["filtercollisionstoego"] = False

    cfg['teams'] = dict()
    cfg['teams']['numteams'] = cfg_train['policy']['numteams']
    cfg['teams']['teamsize'] = cfg_train['policy']['teamsize']
    cfg['learn']['timeout'] = 50
    args.override_cfg_with_args(cfg, cfg_train)
    set_dependent_cfg_entries(cfg, cfg_train)

    # logdir = logdir_root +'/'+timestamp+'_no_dist_to_go_' + cfg_train['runner']['algorithm_class_name']+'_'+str(cfg_train['runner']['num_steps_per_env'])
    # logdir = logdir_root +'/'+timestamp + cfg_train['runner']['algorithm_class_name']+'_'+str(cfg_train['runner']['num_steps_per_env'])
    logdir = logdir_root +'/eval/'+timestamp
    cfg["logdir"] = logdir


    #load real data
    path_data = "/home/peter/git/tri_dynamics_data/sim2realcomparsion/sim2real_data_2023-04-14-11-29-28.pkl"
    with open(path_data, 'rb') as experiment_data_file:
        experiment_data = pickle.load(experiment_data_file)        
    names = experiment_data[0]
    msg_dicts = experiment_data[1]
    ll_cmd_msgs = msg_dicts[0] 
    pose_msgs = msg_dicts[2]
    ll_cmd_data = np.array(list(ll_cmd_msgs.values()))
    pose_data = np.array(list(pose_msgs.values()))    
    
    times_commands = list(ll_cmd_msgs.keys())
    times_poses = list(pose_msgs.keys())
    #use commands to align times
    times_zero_commands = np.array([float(t) for t in times_commands]) - float(list(times_commands)[0])
    times_zero_poses = np.array([float(t) for t in times_poses]) - float(list(times_commands)[0])
    dur = times_zero_commands[-1] - times_zero_commands[0]
    
    times_to_command = 1/10.*np.arange(int(dur*10))
    cmd_idx = np.array([np.argmin(np.abs(times_zero_commands-t)) for t in times_to_command])
    pose_idx = np.array([np.argmin(np.abs(times_zero_poses-t)) for t in times_to_command])

    ll_cmds_aligned = ll_cmd_data[cmd_idx]
    pose_aligned = pose_data[pose_idx]
    #unwrap yaw
    wrap5 = 0
    yaw_vals =[]
    for i in range(pose_aligned.shape[0]-1):
        yaw_next = pose_aligned[i+1, 2]
        yaw = pose_aligned[i, 2]
        if np.abs(yaw_next-yaw)>5.0:
            if yaw_next<yaw:
                wrap5 += 2*np.pi
            else:
                wrap5 -= 2*np.pi
        yaw_vals.append(yaw+wrap5)
    yaw_vals = np.array(yaw_vals+[pose_aligned[-1, 2]+wrap5])
    #pose_aligned[:,2] = np.array(yaw_vals+[pose_aligned[-1, 2]+wrap])
    eval()