import random
import torch
from torch import optim
from torch import nn
import numpy as np
from MultiActorEnv import MultiActorEnv
from FCNN import FCNN

def init_seed(torch_seed=100, random_seed=100, np_random_seed=100):
    '''
    Initialize all random seeds
    '''
    torch.manual_seed(torch_seed)
    random.seed(random_seed)
    np.random.seed(np_random_seed)
    torch.use_deterministic_algorithms(True)

def init_weights(m):
    '''
    Initialize net weights
    '''
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.00)

def TrainQlearning(model: nn.Module, env: MultiActorEnv, config, loss: torch.nn.modules.loss = nn.MSELoss()):
    '''
    This function trains the model using the Q-learning algorithm
    '''
    lr = config.learning_rate
    num_of_play, max_play_length = config.epochs, config.max_play_length
    gamma, epsilon, epsilon_low, epsilon_step = config.gamma, config.epsilon, config.epsilon_low, config.epsilon_step
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_func = loss

    init_seed()
    model.apply(init_weights)
    model.train()

    rewards = []

    for i in range(1, num_of_play+1):
        done = False
        counter = 0
        env.reset()
            
        if i % 100 == 0:
            print("play round: {}, ave reward (last {}): {:.4f}".format(
                i, 100, sum(rewards[-100:])/100))
            if epsilon > epsilon_low:
                epsilon -= epsilon_step
        while (counter <= max_play_length):
            counter += 1
            actor_number = env.get_actor_number()
            if (actor_number==0): break
            
            inputs = []            
            choices = []
            for actor_id in range(actor_number):
                # find Q(s_{t},a) for all actions
                preds = []
                state = env.get_state(actor_id)
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
                
                choices.append(choice)
                inputs.append(preds[choice])
                
            # take the action, s_{t},a -> s_{t+1}
            # get the immediate reward
            imm_reward, dones = env.step(choices)
            
            targets = []
            for actor_id in range(actor_number):
                future_reward = 0
                # find Q(s_{t+1},a) for all actions
                if not dones[actor_id]:
                    with torch.no_grad():
                        next_preds = []
                        state = env.get_state(actor_id)
                        for action in env.actions:
                            next_pred = model(state, action)
                            next_preds.append(next_pred.item())
                        future_reward = max(next_preds)
                        
                # Q(s,a|t) = r + gamma*max[Q(s,a|t+1)]
                tot_reward = imm_reward[actor_id] + gamma * future_reward
                rewards.append(tot_reward)
                targets.append(torch.Tensor([tot_reward]))

            inputs = torch.stack(inputs, dim=0)
            targets = torch.stack(targets, dim=0)
            loss = loss_func(inputs, targets)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            env.remove_dones()

    return rewards
