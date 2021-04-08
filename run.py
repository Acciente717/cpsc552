import random
import torch
from torch import optim
from torch import nn
import numpy as np
from PathPlanningEnv import PathPlanningEnv
from FCNN import FCNN

import settings


def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.00)


def weights_init_uniform_rule(m):
    classname = m.__class__.__name__
    # for every Linear layer in a model..
    if classname.find('Linear') != -1:
        # get the number of the inputs
        n = m.in_features
        y = 1.0/np.sqrt(n)
        m.weight.data.uniform_(-y, y)
        m.bias.data.fill_(0)


def init_seed(torch_seed=100, random_seed=100, np_random_seed=100):
    torch.manual_seed(torch_seed)
    random.seed(random_seed)
    np.random.seed(np_random_seed)


def TrainMemorize(model: nn.Module, env: PathPlanningEnv, config: settings.Config, loss: torch.nn.modules.loss = nn.MSELoss()):
    '''
    This function trains the model to memorize rewards for each (state, action) pair
    '''
    device = config.device
    if "cuda" in device:
        model.to(device)

    epsilon, lr = config.epsilon, config.learning_rate
    num_of_play, max_play_length = config.epochs, config.max_play_length
    optimizer = optim.SGD(model.parameters(), lr=lr)
    loss_func = loss

    init_seed()
    model.apply(init_weights)
    model.train()

    rewards = []
    losses = []

    for i in range(1, num_of_play+1):
        done = False
        counter = 0
        env.reset()
        if i % 100 == 0:
            print("play round: {}, ave reward (last {}): {:.4f}".format(
                i, 100, sum(rewards[-100:])/100))
        while (counter <= max_play_length and not done):
            preds = []
            state = env.grid
            for action in env.actions:
                pred = model(state, action)
                preds.append(pred)

            p = np.random.uniform(0, 1)
            choice = -1
            if p < epsilon:
                choice = np.random.randint(0, 4)
            else:
                list_pred = [x.item() for x in preds]
                max_pred = np.amax(list_pred)
                max_positions = np.argwhere(
                    list_pred == max_pred).flatten().tolist()
                choice = random.choice(max_positions)

            # if early_stop = True, the agent is likely to commit suicide
            _, reward, done, _ = env.step(choice, early_stop=False)
            rewards.append(reward)
            counter += 1

            loss = loss_func(preds[choice], torch.Tensor([reward]))
            losses.append(loss.item())

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

    return rewards, losses


def TrainQlearning(model: nn.Module, env: PathPlanningEnv, config: settings.Config, loss: torch.nn.modules.loss = nn.MSELoss()):
    '''
    This function trains the model using the Q-learning algorithm
    '''
    device = config.device
    if "cuda" in device:
        model.to(device)

    epsilon, lr = config.epsilon, config.learning_rate
    num_of_play, max_play_length = config.epochs, config.max_play_length
    gamma, epsilon, epsilon_low, epsilon_step = config.gamma, config.epsilon, config.epsilon_low, config.epsilon_step
    optimizer = optim.SGD(model.parameters(), lr=lr)
    loss_func = loss

    init_seed()
    model.apply(init_weights)
    model.train()

    rewards = []
    losses = []

    for i in range(1, num_of_play+1):
        done = False
        counter = 0
        env.reset()
        if i % 100 == 0:
            print("play round: {}, ave reward (last {}): {:.4f}".format(
                i, 100, sum(rewards[-100:])/100))
            if epsilon > epsilon_low:
                epsilon -= epsilon_step
        while (counter <= max_play_length and not done):
            # find Q(s_{t},a) for all actions
            preds = []
            state = env.grid
            for action in env.actions:
                pred = model(state, action)
                preds.append(pred)

            p = np.random.uniform(0, 1)
            choice = -1
            if p < epsilon:
                choice = np.random.randint(0, 4)
            else:
                list_pred = [x.item() for x in preds]
                max_pred = np.amax(list_pred)
                max_positions = np.argwhere(
                    list_pred == max_pred).flatten().tolist()
                choice = random.choice(max_positions)

            # take the action, s_{t},a -> s_{t+1}
            # get the immediate reward
            _, imm_reward, done, _ = env.step(choice, early_stop=False, q_learning=True)

            future_reward = 0
            # find Q(s_{t+1},a) for all actions
            if not done:
                with torch.no_grad():
                    next_preds = []
                    state = env.grid
                    for action in env.actions:
                        next_pred = model(state, action)
                        next_preds.append(next_pred.item())
                    future_reward = max(next_preds)
                    #print(next_preds, future_reward)

            tot_reward = imm_reward + gamma * future_reward
            rewards.append(tot_reward)
            counter += 1

            # Q(s,a|t) = r + gamma*max[Q(s,a|t+1)]
            loss = loss_func(preds[choice], torch.Tensor([tot_reward]))
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

    return rewards, losses


def PlayOnce(model: nn.Module, env: PathPlanningEnv, config: settings.Config):
    env.reset()
    model.eval()

    max_play_length = config.max_play_length
    counter = 0
    print("Step {}".format(counter))
    counter += 1
    env.display()

    done = False
    while counter <= max_play_length and not done:
        preds = []
        state = env.grid
        print("state-action rewards: ", end=" ")
        for action in env.actions:
            pred = model(state, action)
            preds.append(pred)
            print("{:.4f}".format(pred.item()), end=" ")
        print(" ")

        choice = np.argmax(np.array(preds))
        _, _, done, _ = env.step(choice)

        print("Step {}".format(counter))
        counter += 1
        env.display()
