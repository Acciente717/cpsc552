import math
import torch
import numpy as np
from torch import nn, optim
from AStar import AStar
from PathPlanningEnv import PathPlanningEnv

class Loc():
    def __init__(self, row, col):
        self.row = row
        self.col = col

    def copy(self):
        return Loc(self.row, self.col)

    def copy_from(self, loc):
        self.row = loc.row
        self.col = loc.col
    
    def move(self, direction):
        row = self.row
        col = self.col
        if direction == 'u' or direction == 0:
            row -= 1
        elif direction == 'd' or direction == 1:
            row += 1
        elif direction == 'l' or direction == 2:
            col -= 1
        elif direction == 'r' or direction == 3:
            col += 1
        else:
            raise RuntimeError("Error: unknown move")
        return Loc(row, col)

    def __eq__(self, other):
        return (self.row == other.row) and (self.col == other.col)


def ReverseAction(action):
    '''
    Input -> Return:
    1 -> 0
    0 -> 1
    2 -> 3
    3 -> 2
    '''
    parity = action % 2
    change = parity * (-2) + 1
    return action + change

def Softmax(l):
    exps = [math.e ** i for i in l]
    s = sum(exps)
    return [i/s for i in exps]

class ProgressiveTrainer:
    
    def __init__(self, model, height=10, width=10, num_obstacle=10, max_play_length=500,
                 epsilon_high=0.9, epsilon_low=0.1, gamma=0.9, lr=0.01,
                 init_env_num=1, max_env_num=20, env_inc_acc=0.9, env_final_acc=0.98, seed=42,
                 loss_func=nn.MSELoss(), device='cpu'):
        '''
        Parameters:
        model: the action-value function network
        height: map height
        width: map width
        num_obstacle: number of obstacles in the map
        max_play_length: maximum number of steps in each epoch
        epsilon_high: high epsilon value
        epsilon_low: low epsilon value
        gamma: discounting factor
        lr: learning rate
        max_env_num: the maximum number of envs
        env_inc_acc: if all existing envs reach this accuracy, a new env will be added
        env_final_acc: if all existing envs reach this accuracy and max_env_num is reached, the training stops
        loss_func: loss function
        device: running device
        '''

        self.model = model
        self.loss_func = loss_func
        self.device = device
        self.optimizer = optim.SGD(model.parameters(), lr=lr)
        self.height = height
        self.width = width
        self.num_obstacle = num_obstacle
        self.epsilon_high = epsilon_high
        self.epsilon_low = epsilon_low
        self.gamma = gamma
        self.max_play_length = max_play_length
        
        
        self.model.to(device)
        self.device = device
        self.envs = []
        self.init_env_num = init_env_num
        self.max_env_num = max_env_num
        self.env_inc_acc = env_inc_acc
        self.env_final_acc = env_final_acc

        self.seed = seed
        
        self.train_state = {}
        
        while len(self.envs) < self.init_env_num:
            self._add_new_env()
            
    
    def _add_new_env(self):
        '''
        Add a new environment to the environment list.
        '''
        
        # Generate a random map and make sure that all non-obstacle
        # positions can reach the goal.
        while True:
            model_settings = {
                'height' : self.height,
                'width' : self.width,
                'obs_count' : self.num_obstacle,
                'random_seed': self.seed,
                'device': self.device
            }
            self.seed += 1
            new_env = PathPlanningEnv(**model_settings)
            astar = AStar(new_env.grid[2,:,:], (new_env.goal_row, new_env.goal_col))

            all_reachable = True
            for row in range(self.height):
                for col in range(self.width):
                    if new_env.grid[2,row,col] == 0 \
                    and new_env.grid[1,row,col] != 1 \
                    and not astar.plan(row, col):
                        all_reachable = False
            
            if all_reachable: break

        self.envs.append(new_env)
    
    def _gen_vec_map(self, env):
        '''
        Generate the vector map of an environment. Each entry in the vector map
        represents the agent's action (next step direction).
        '''
        
        # Vector map value: -2: obstacle, -1: goal, 0-3: directions
        vec_map = torch.full((env.height, env.width), -2, dtype=torch.int8, requires_grad=False).to(self.device)

#         for i in range(env.height):
#             for j in range(env.width):
#                 old_obs = env.grid[2,:,:]
#                 env._init_from_grid(old_obs, i, j, env.goal_row, env.goal_col, self.device)
#                 state = env.grid
#                 if env.grid[2, i, j] == 0 and (i, j) != (env.goal_row, env.goal_col):
#                     preds = []
#                     state = env.grid.clone().detach()
#                     state = state.view(1, *state.shape)
#                     for action in env.actions:
#                         action = action.view(1, *action.shape)
#                         pred = self.model(state, action)
#                         preds.append(pred.item())
#                     preds = Softmax(preds)
#                     max_index = preds.index(max(preds))
#                     vec_map[i, j] = max_index
        
#         vec_map[env.goal_row, env.goal_col] = -1
#         return vec_map

#         for i in range(env.height):
#             for j in range(env.width):
#                 old_obs = env.grid[2,:,:]
#                 env._init_from_grid(old_obs, i, j, env.goal_row, env.goal_col, self.device)
#                 state = env.grid
#                 if env.grid[2, i, j] == 0 and (i, j) != (env.goal_row, env.goal_col):
#                     state = env.grid.clone().detach()
#                     states = torch.stack((state, state, state, state))
#                     actions = torch.stack(env.actions)
#                     preds = self.model(states, actions)
#                     preds = list(preds.flatten())
#                     max_index = preds.index(max(preds))
#                     vec_map[i, j] = max_index

#         vec_map[env.goal_row, env.goal_col] = -1
#         return vec_map
    
        for i in range(env.height):
            packed_states = []
            packed_actions = []
            for j in range(env.width):
                old_obs = env.grid[2,:,:]
                env._init_from_grid(old_obs, i, j, env.goal_row, env.goal_col, self.device)
                state = env.grid
                if env.grid[2, i, j] == 0 and (i, j) != (env.goal_row, env.goal_col):
                    state = env.grid.clone().detach()
                    states = torch.stack((state, state, state, state))
                    actions = torch.stack(env.actions)
                    packed_states.append(states)
                    packed_actions.append(actions)
            packed_states = torch.cat(packed_states, dim=0)
            packed_actions = torch.cat(packed_actions, dim=0)
            packed_preds = self.model(packed_states, packed_actions)
            idx = 0
            for j in range(env.width):
                if env.grid[2, i, j] == 0 and (i, j) != (env.goal_row, env.goal_col):
                    preds = list(packed_preds[idx:idx+4].flatten())
                    max_index = preds.index(max(preds))
                    vec_map[i, j] = max_index
                    idx += 4
        return vec_map

    def _is_out_of_boundary(self, env, loc):
        '''
        Returns True if the location is out of the boundaries.
        '''
        if (loc.row >= env.height) or (loc.row < 0):
            return True
        if (loc.col >= env.width) or (loc.col < 0):
            return True
        return False

    def _is_obstacle(self, env, loc):
        '''
        Returns True if the location is an obstacle.
        '''
        return env.grid[2, loc.row, loc.col] == 1
    
    def _is_goal(self, env, loc):
        '''
        Returns True if the location is the goal.
        '''
        return env.grid[1, loc.row, loc.col] == 1

    def _find_env_failure(self, env):
        '''
        Find the positions on the map from which the agent can't reach the goal.
        '''
        vec_map = self._gen_vec_map(env)
        
        # Result map value: -1: can't reach goal, 0: not tested yet, 1: can reach goal
        results = torch.full((env.height, env.width), 0, dtype=torch.int8, requires_grad=False)

        failed_cnt = 0
        failed_locs = []

        for i in range(env.height):
            for j in range(env.width):
                cur_loc = Loc(i, j)

                # This position has known result.
                if results[i, j]: continue
                    
                # This position is an obstacle.
                if self._is_obstacle(env, cur_loc): continue
                    
                # This position is the goal.
                if self._is_goal(env, cur_loc): continue
                
                # A visited map, used to detect looping.
                visited = torch.full((env.height, env.width), False, dtype=torch.bool, requires_grad=False)

                # Agent trajectory.
                path = []

                while not self._is_out_of_boundary(env, cur_loc) \
                and not self._is_obstacle(env, cur_loc) \
                and not self._is_goal(env, cur_loc) \
                and not visited[cur_loc.row, cur_loc.col] \
                and not results[cur_loc.row, cur_loc.col]:
                    visited[cur_loc.row, cur_loc.col] = True
                    path.append(cur_loc)
                    cur_loc = cur_loc.move(vec_map[cur_loc.row, cur_loc.col])

                # If the agent runs into a boundary or an obstacle or forms a loop
                # or a known position that will lead to the above in the future,
                # mark all the positions in the trajectory as fail (-1).
                if self._is_out_of_boundary(env, cur_loc) \
                or self._is_obstacle(env, cur_loc) \
                or visited[cur_loc.row, cur_loc.col] \
                or results[cur_loc.row, cur_loc.col] < 0:
                    for loc in path:
                        results[loc.row, loc.col] = -1
                # Otherwise, the agent must have reached the goal. Mark all the
                # positions in the trajectory as success (1).
                else:
                    for loc in path:
                        results[loc.row, loc.col] = 1
        
        # Count the number of failed positions and record them.
        for i in range(env.height):
            for j in range(env.width):
                if results[i, j].item() < 0:
                    failed_cnt += 1
                    failed_locs.append(Loc(i, j))
                
        return failed_cnt / (env.height * env.width), failed_locs

    def _choose_train_env(self):
        '''
        Choose the environment that has the lowest accuracy in the environment list.
        '''
        if not self.envs:
            self._add_new_env()

        fail_rates = []
        fail_poses = []
        
        # Check the fail rate and fail positions of all existing environments.
        for env in self.envs:
            fail_rate, fail_pos = self._find_env_failure(env)
            fail_rates.append(fail_rate)
            fail_poses.append(fail_pos)

        worst_idx = np.argmax(fail_rates)
        worst_rate = fail_rates[worst_idx]
        
        # Check whether it's time to add more environments.
        if 1 - worst_rate >= self.env_inc_acc:
            
            # If we can still add more environments, add one and try again.
            if len(self.envs) < self.max_env_num:
                self._add_new_env()
                self.train_state['inc_env'] = True
                return self._choose_train_env()
            
            # If we have reached the maximum number of environments, if
            # all the environments have their accuracy above the threshold,
            # stop the training.
            if 1 - worst_rate >= self.env_final_acc:
                return None
            
        # Pick the worst performance environment.
        worst_env = self.envs[worst_idx]
        start_pos = np.random.choice(fail_poses[worst_idx])
        
        return worst_env, start_pos, worst_rate

    def test_envs(self):
        '''
        Return the fail rates of all environments.
        '''
        fail_rates = []
        for env in self.envs:
            fail_rate, _ = self._find_env_failure(env)
            fail_rates.append(fail_rate)
        return fail_rates

    def train_once(self):
        self.train_state = {
            'finish': False,
            'inc_env': False
        }
        
        # Choose the worst environment to train.
        chosen_env_tup = self._choose_train_env()
        if chosen_env_tup is None:
            self.train_state['finish'] = True
            return self.train_state
        env, start_pos, fail_rate = chosen_env_tup
        self.train_state['fail_rate'] = fail_rate
        env.reset_to(start_pos.row, start_pos.col)

        # Set epsilon based on the fail rate. Epsilon is larger when
        # the fail rate is higher.
        fail_rate **= 1/4
        epsilon = self.epsilon_high * fail_rate + self.epsilon_low * (1 - fail_rate)

        done = False
        counter = 0
        moves = []
        imm_rewards = []
        losses = []
        
        # Walk through the map to reach the goal without updating the
        # network. Record the path.
        self.model.eval()
        while counter < self.max_play_length and not done:
            # For probability epsilon, choose a random action, otherwise
            # choose the best action predicted by the network.
            p = np.random.uniform(0, 1)
            if p < epsilon:
                choice = np.random.randint(0, 4)
            else:
                preds = []
                state = env.grid.clone().detach()
                states = torch.stack((state, state, state, state))
                actions = torch.stack(env.actions)
                preds = self.model(states, actions)
                list_pred = [x.item() for x in preds]
                max_pred = np.amax(list_pred)
                max_positions = np.argwhere(
                    list_pred == max_pred).flatten().tolist()
                choice = np.random.choice(max_positions)
            
            # Take the step and save the immediate reward value.
            _, imm_reward, done, _ = env.step(
                choice, early_stop=False, q_learning=True)
            moves.append(choice)
            imm_rewards.append(imm_reward)

            counter += 1

        # If the agent successfully reached the goal, reversely update
        # the network from the goal to the starting position.
        if done:
            self.model.train()

            # Spetial treatment for the very last step, because reaching
            # the goal only gives a positive immediate reward and no
            # future reward.
            cur_row, cur_col = env.goal_row, env.goal_col
            cur_action, cur_reward = moves.pop(), imm_rewards.pop()
            rev_action = ReverseAction(cur_action)
            env.step(rev_action, early_stop=False, q_learning=True)

            state = env.grid.clone().detach()
            state = state.view(1, *state.shape)

            action_vec = env.actions[cur_action]
            action_vec = action_vec.view(1, *action_vec.shape)
            self.optimizer.zero_grad()
            pred_reward = self.model(state, action_vec)

            real_reward = torch.Tensor([cur_reward]).to(self.device)
            real_reward = real_reward.view(1, *real_reward.shape)
            loss = self.loss_func(pred_reward, real_reward)

            losses.append(loss)

            loss.backward()
            self.optimizer.step()

            # Calculate new predicted future reward for the updating
            # of the previous step.
            with torch.no_grad():
                next_preds = []
                state = env.grid.clone().detach()
                state = state.view(1, *state.shape)
                for action in env.actions:
                    action = action.view(1, *action.shape)
                    next_pred = self.model(state, action)
                    next_preds.append(next_pred.item())
                future_reward = max(next_preds)

            # Go reversely through the path and update the network
            # along the way.
            while len(moves) > 0 and len(imm_rewards) > 0:
                # Take a step backward and update the network.
                cur_row, cur_col = env.goal_row, env.goal_col
                cur_action, cur_reward = moves.pop(), imm_rewards.pop()
                rev_action = ReverseAction(cur_action)
                
                # If the reward was 0, it means that it wasn't stepping into
                # the boundaries or obstacles, so we should perform a reverse
                # step. Otherwise, we should stay at the same position.
                if cur_reward == 0:
                    env.step(rev_action, early_stop=False, q_learning=True)

                state = env.grid.clone().detach()
                state = state.view(1, *state.shape)

                action_vec = env.actions[cur_action]
                action_vec = action_vec.view(1, *action_vec.shape)
                self.optimizer.zero_grad()
                pred_reward = self.model(state, action_vec)

                real_reward = cur_reward + self.gamma * future_reward
                real_reward = torch.Tensor([real_reward]).to(self.device)
                real_reward = real_reward.view(1, *real_reward.shape)
                loss = self.loss_func(pred_reward, real_reward)

                losses.append(loss)

                loss.backward()
                self.optimizer.step()

                # Calculate new predicted future reward for the updating
                # of the previous step.
                with torch.no_grad():
                    next_preds = []
                    state = env.grid.clone().detach()
                    state = state.view(1, *state.shape)
                    for action in env.actions:
                        action = action.view(1, *action.shape)
                        next_pred = self.model(state, action)
                        next_preds.append(next_pred.item())
                    future_reward = max(next_preds)

        return self.train_state
