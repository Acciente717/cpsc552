import math
import numpy as np
import matplotlib.pyplot as plt
from PathPlanningEnv import PathPlanningEnv

def MovingAveragePlot(input_list, window_size):
    plt.subplots(figsize=(14, 7))

    plt.subplot(1, 2, 1)
    plt.plot(range(len(input_list)), input_list)

    plt.subplot(1, 2, 2)
    window = np.ones(int(window_size))/float(window_size)
    ave_values = np.convolve(input_list, window, 'valid')
    plt.plot(range(len(ave_values)), ave_values)

    plt.show()


def Softmax(l):
    exps = [math.e ** i for i in l]
    s = sum(exps)
    return [i/s for i in exps]


def VectorFieldPlot(network, env: PathPlanningEnv, width, height):

    fig = plt.subplots(figsize=(width, height))
    ax = plt.axes(xlim=(-1, width - 1 + 1), ylim=(-1, height - 1 + 1))
    tgtPoint, = ax.plot(env.goal_col, height - 1 - env.goal_row, linestyle='', markersize=20, marker='o', color=(0.5, 0.5, 1.0, 0.75))

    # obstacles
    for i in range(height):
        for j in range(width):
            if env.grid[2, i, j] == 1:
                obs_row = height - 1 - i
                obs_col = j
                obsstaclePoint, = ax.plot(obs_col, obs_row, linestyle='', markersize=15, marker='x', markeredgewidth=2, color='r')

    # vector field
    for i in range(height):
        for j in range(width):
            old_obs = env.grid[2,:,:]
            env._init_from_grid(old_obs, i, j, env.goal_row, env.goal_col)
            state = env.grid
            if env.grid[2, i, j] == 0 and (i, j) != (env.goal_row, env.goal_col):
                preds = []
                state = env.grid.clone().detach()
                state = state.view(1, *state.shape)
                for action in env.actions:
                    action = action.view(1, *action.shape)
                    pred = network(state, action)
                    preds.append(pred.item())
                preds = Softmax(preds)
                max_index = preds.index(max(preds))
                max_reward = max(preds)

                x = j  # x coordinates
                y = height - 1 - i  # y coordinates
                if max_index == 0: # go up
                    u = [0]  # x component of SVF
                    v = [max_reward]  # y component of SVF
                elif max_index == 1: # go down
                    u = [0]  
                    v= [-max_reward]
                elif max_index == 2: # go left
                    u = [-max_reward]
                    v = [0]  
                else: # go right
                    u = [max_reward]  # 
                    v = [0]
                Q1 = ax.quiver(x, y, u, v, color=(0, 0.2, 0), scale_units='xy', headwidth=0.5);

    plt.xticks([]);
    plt.yticks([]);
