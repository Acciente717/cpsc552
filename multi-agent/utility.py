import numpy as np
import random
import matplotlib.pyplot as plt

def MovingAveragePlot(input_list, window_size):
    '''
    Plot the moving average of a list of rewards or losses
    '''
    plt.subplots(figsize=(14, 7))

    plt.subplot(1, 2, 1)
    plt.plot(range(len(input_list)), input_list)

    plt.subplot(1, 2, 2)
    window = np.ones(int(window_size))/float(window_size)
    ave_values = np.convolve(input_list, window, 'valid')
    plt.plot(range(len(ave_values)), ave_values)

    plt.show()

def RoutePlot(width, height, obs, tgts, locs, movs=None, image_name=None):
    '''
    Plot the current locations, target locations, moving directions, and obstacles
    '''
    plt.figure(figsize=(width, height))
    
    plt.xlim([-0.5, width-0.5])
    plt.ylim([-0.5, height-0.5])
    x_major_ticks = np.arange(-0.5, width+0.5, 1)
    y_major_ticks = np.arange(-0.5, height+0.5, 1)
    plt.xticks(x_major_ticks)
    plt.yticks(y_major_ticks)
    
    ax = plt.gca()
    ax.axes.xaxis.set_ticklabels([])
    ax.axes.yaxis.set_ticklabels([])
    ax.tick_params(axis="x",direction="in")
    ax.tick_params(axis="y",direction="in")
    plt.grid()
    
    plt.gca().invert_yaxis()
    
    # show obs
    plt.scatter(obs[0], obs[1], s=500, marker='x', color='r')
    
    # show tgts
    plt.scatter(tgts[0], tgts[1], s=1000, marker='o', color=(0.5, 0.5, 1.0, 0.75))
    for x,y,l in zip(tgts[0], tgts[1], tgts[2]):
        label = "{}".format(l)
        plt.annotate(label, (x,y), size=20, textcoords="offset points", xytext=(0,-7), ha='center')
    
    # show mov direction
    if movs is not None:
        for i in range(len(locs[0])-len(movs[0])):
            movs[0].append(0)
            movs[1].append(0)
        plt.quiver(locs[0], locs[1], movs[0], movs[1], color=(0, 0.2, 0), angles='xy', scale_units='xy', scale=1)
    
    # show locs
    plt.scatter(locs[0], locs[1], s=1000, marker='o', color=(0.0, 0.5, 0.0))
    for x,y,l in zip(locs[0], locs[1], locs[2]):
        label = "{}".format(l)
        plt.annotate(label, (x,y), size=20, textcoords="offset points", xytext=(0,-7), ha='center')
    
    if image_name is not None:
        plt.savefig(image_name)

def get_optimal_movs(env, model):
    '''
    Return the optimal move directions for each agent
    '''
    movs = [[],[]]
    choices = []
    actor_number = env.get_actor_number()
    if (actor_number==0): return movs, choices

    for actor_id in range(actor_number):
        preds = []
        state = env.get_state(actor_id)
        for action in env.actions:
            pred = model(state, action)
            preds.append(pred)
        choice = -1
        list_pred = [x.item() for x in preds]
        max_pred = np.amax(list_pred)
        max_positions = np.argwhere(list_pred == max_pred).flatten().tolist()
        choice = random.choice(max_positions)
        choices.append(choice)
        if choice == 0:
            movs[0].append(0)
            movs[1].append(-1)
        elif choice == 1:
            movs[0].append(0)
            movs[1].append(1)
        elif choice == 2:
            movs[0].append(-1)
            movs[1].append(0)
        elif choice == 3:
            movs[0].append(1)
            movs[1].append(0)
        elif choice == 4:
            movs[0].append(0)
            movs[1].append(0)
        else:
            assert False, "Unknow operation"
    return movs, choices

def PlayOnce(env, model, config):
    '''
    Play a whole episode of the game, and draw pictures for every step
    '''
    env.reset()

    width = env.width
    height = env.height

    counter = 0
    max_play_length = config.max_play_length
    model.eval()
    while (counter <= max_play_length and env.get_actor_number() > 0):
        actor_number = env.get_actor_number()
        if (actor_number==0): break            
        
        obs = env.get_obs()
        tgts = env.get_tgts()
        locs = env.get_locs()
        movs, choices = get_optimal_movs(env, model)
        RoutePlot(width, height, obs, tgts, locs, movs, str(counter)+".png")
        
        env.step(choices)
        env.remove_dones()
        counter += 1
        
    obs = env.get_obs()
    tgts = env.get_tgts()
    locs = env.get_locs()
    RoutePlot(width, height, obs, tgts, locs, None, str(counter)+".png")
