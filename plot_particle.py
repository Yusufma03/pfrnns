import torch
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Rectangle
from matplotlib.collections import PatchCollection
import matplotlib.patches as patches
import os
import numpy as np
import sys
import json 
import pickle
import configargparse

"""
Reference from https://matplotlib.org/3.3.2/gallery/animation/double_pendulum.html#sphx-glr-gallery-animation-double-pendulum-py 
"""

def run(time_step):

    # run particles
    try:
        x = particle_pred[:, traj_num, time_step, 0].cpu().data.numpy() * 10
        y = particle_pred[:, traj_num, time_step, 1].cpu().data.numpy() * 10
    except:
        print("Incorrect trajectory number! Should be smaller than {}. ".format(len(particle_pred[0])))
        x = particle_pred[:, 0, time_step, 0].cpu().data.numpy() * 10
        y = particle_pred[:, 0, time_step, 1].cpu().data.numpy() * 10
    
    particle_ax.set_data(x, y)
    step_text.set_text(step_template % (time_step))

    # run robot reference
    try:
        x = robot_traj[traj_num][time_step][0]
        y = robot_traj[traj_num][time_step][1]
    except:
        print("Incorrect trajectory number! Should be smaller than {}. ".format(len(robot_traj)))
        x = robot_traj[0][time_step][0]
        y = robot_traj[0][time_step][1]
    
    robot_ax.set_data(x, y)

    return particle_ax, robot_ax


#### this function is not ready ####
# if trying the save the video with multiple animation
# please combine all animation updates in one run function. 
def run_robot_arrow(time_step):

    robot_orient_ax.remove()

    arrow_length = 0.5
    try:
        x = robot_traj[traj_num][time_step][0]
        y = robot_traj[traj_num][time_step][1]
        angle = robot_traj[traj_num][time_step][2] / 360.0 * 2 * np.pi
        dx = arrow_length * np.cos(angle) 
        dy = arrow_length * np.sin(angle) 
    except:
        print("Incorrect trajectory number! Should be smaller than {}. ".format(len(robot_traj)))
        x = robot_traj[0][time_step][0]
        y = robot_traj[0][time_step][1]
        angle = robot_traj[traj_num][time_step][2] / 360.0 * 2 * np.pi
        dx = arrow_length * np.cos(angle) 
        dy = arrow_length * np.sin(angle) 
    
    # arrow.set_data(x, y, dx, dy)
    arrow_x = x
    arrow_y = y
    arrow_dx = dx
    arrow_dy = dy
    
    arrow = ax.arrow(arrow_x, arrow_y, arrow_dx, arrow_dy, fc='k', ec='k')
    robot_orient_ax = arrow

    return robot_orient_ax


def plot_obstacles(facecolor='r', edgecolor='None'):

    obstacles = []
    y_dim = len(maze_data)
    x_dim = 0 if y_dim == 0 else len(maze_data[0])

    for j in range(0, y_dim):
        for i in range(0, x_dim):
            if maze_data[i][j] == 0:
                continue
            obstacles.append([y_dim - j - 1, i, maze_data[i][j]])

    # Loop over data points; create box from errors at each point
    for x, y, obs in obstacles:
        """
        #738595 : grey
        #fc5a50 : coral
        """
        facecolor = '#738595' if obs == 1 else '#fc5a50'
        rect = Rectangle((x, y), 1, 1, linewidth=1, edgecolor=edgecolor, facecolor=facecolor)
        ax.add_patch(rect)


def plot_robot_and_particles(save_video=False):

    config = get_config()
    sequence_length = config['sl']

    ani = animation.FuncAnimation(fig, run, sequence_length, interval=500)
    # ani_orient = animation.FuncAnimation(fig, run_robot_arrow, sequence_length, interval=500)
    if save_video:
        ani.save('particles.mp4', writer=writer)
    else:
        plt.show()



def get_data_name(config, train):
    """
    get the dataset name

    :param config: experiment args
    :param train: train / eval
    :return: fname: the name of the file
    """
    # number of trajs
    num_trajs = config["num_trajs"]
    # number of sequence_length
    traj_len = config["sl"]

    mode = 'train' if train else 'eval'

    fname = '{}_data_trajs{}_sl{}.pkl'.format(mode, num_trajs, traj_len)
    return fname


def load_data():
    # load predicted particles 
    particle_pred = torch.load(get_particle_fname())

    # load robot trajectories
    config = get_config()
    eval_fname = get_data_name(config, False)
    eval_fpath = os.path.join('data', eval_fname)

    eval_data = pickle.load(open(eval_fpath, 'rb'))
    robot_traj = eval_data['trajs']

    maze_data = eval_data['map']
    
    return particle_pred, robot_traj, maze_data


def get_config():
    config = json.load(open(get_config_fname(), 'r'))
    return config


def get_particle_fname():
    return os.path.join('eval', str(eval_num), 'particle_pred')


def get_config_fname():
    return os.path.join('eval', str(eval_num), 'args.conf')


if __name__ == "__main__":
    parser = configargparse.ArgumentParser(default_config_files=[])
    parser.add_argument('--traj_num', type=int, default=0, help='the number of trajectory')
    parser.add_argument('--eval_num', type=int, default=0, help='the number of evaluate folder')
    plot_args = parser.parse_args()
    traj_num = plot_args.traj_num
    eval_num = plot_args.eval_num

    #### Set up formatting for the movie files ####
    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=15, metadata=dict(artist='Me'), bitrate=1800)

    #### Create plot figure ####
    fig = plt.figure(figsize=(5,4))
    ax = fig.add_subplot(111, xlim=(0, 10), ylim=(0, 10))
    ax.grid()

    particle_ax, = ax.plot([], [], '.')
    robot_ax, = ax.plot([], [], 'r.')
    #### robot orientation arrow haven't succeed yet. ####
    # robot_orient_ax = ax.add_patch(arrow)
    step_template = 'step = %d'
    step_text = ax.text(0.05, 0.9, '', transform=ax.transAxes)

    #### global variables for plotting particles and robot position ####
    particle_pred, robot_traj, maze_data = load_data()

    plot_obstacles()
    plot_robot_and_particles()