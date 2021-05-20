import random
import torch
import numpy as np

class Loc():
    '''
    A class for agent location
    '''
    def __init__(self, row, col):
        self.row = row
        self.col = col

    def copy(self):
        return Loc(self.row, self.col)

    def set_from(self, loc):
        self.row = loc.row
        self.col = loc.col

    def __eq__(self, other):
        return (self.row == other.row) and (self.col == other.col)

    def __str__(self):
        return "(" + str(self.row) + "," + str(self.col) + ")"

    def __repr__(self):
        return str(self)

class MultiActorEnv():
    def __init__(self, *,
                 actor_number,
                 height, width, obs_count, grid=None,
                 srcs=None, tgts=None,
                 random_seed=None):

        # map height, width, and number of obstacles
        self.height = height
        self.width = width
        self.obs_count = obs_count

        # list of source-target pairs and associated information
        self.actor_number = actor_number
        self.srcs = list()
        self.tgts = list()
        self.locs = list()
        self.idx = list()
        self.srcs_init = list()
        self.tgts_init = list()

        # list of agents that have been in there target location
        self.tgt_dones = list()
        self.idx_dones = list()


        if random_seed is None:
            random_seed = 100
        
        # randomly initialize the map
        self._init_random_grid(self.actor_number, self.height, self.width, self.obs_count, random_seed)

        # action space
        self.actions = [
            torch.Tensor().new_tensor([1, 0, 0, 0, 0], dtype=torch.float32, requires_grad=False),  # up
            torch.Tensor().new_tensor([0, 1, 0, 0, 0], dtype=torch.float32, requires_grad=False),  # down
            torch.Tensor().new_tensor([0, 0, 1, 0, 0], dtype=torch.float32, requires_grad=False),  # left
            torch.Tensor().new_tensor([0, 0, 0, 1, 0], dtype=torch.float32, requires_grad=False),  # right
            torch.Tensor().new_tensor([0, 0, 0, 0, 1], dtype=torch.float32, requires_grad=False)   # stay still
        ]

        # reward and penalty
        self.target_reward = 1
        self.out_bound_penalty = -0.5
        self.obstacle_penalty = -0.5
        self.collision_penalty = -0.5
        self.stationary_penalty = -0.01

    def _init_random_grid(self, actor_number: int, height: int, width: int, obs_count: int, random_seed: int):
        '''
        Randomly initialize the map, including source-target pairs. obstacles
        '''
        # check total number of grid points
        total_size = height*width
        assert total_size >= 2, 'Error: expect height * width >= 2'

        # initialize the grid
        self.grid_initial = torch.zeros(size=(3, height, width), requires_grad=False)

        # randomly select part of grid points as obstacles, and actor_number (source, target) pair
        random.seed(random_seed)
        random_selections = random.sample(range(total_size), k=min(obs_count+2*actor_number, total_size))

        # randomly assign source and target locations for each object
        for i in range(actor_number):
            num = random_selections.pop()
            source = Loc(num//width, num % width)
            self.srcs_init.append(source)
            self.grid_initial[0, source.row, source.col] = i + 1

            num = random_selections.pop()
            target = Loc(num//width, num % width)
            self.tgts_init.append(target)
            self.grid_initial[1, target.row, target.col] = i + 1

        # randomly assign locations for obstacles
        self.obs_list = [[],[],[]]
        for num in random_selections:
            row = num // width
            col = num % width
            self.grid_initial[2, row, col] = 1
            self.obs_list[0].append(col)
            self.obs_list[1].append(row)

        self.reset()

    def reset(self):
        '''
        Reset the grid map to its initial state
        '''
        self.srcs = [x.copy() for x in self.srcs_init]
        self.tgts = [x.copy() for x in self.tgts_init]
        self.actor_number = len(self.srcs_init)
        self.grid = self.grid_initial.detach().clone()
        self.locs = [x.copy() for x in self.srcs]
        self.idx = list(range(1, self.actor_number+1))
        self.tgt_dones.clear()
        self.idx_dones.clear()

    def display(self, grid=None):
        '''
        A simply funtion to show things on the map
        '''
        if grid is None:
            grid = self.grid
        assert grid.shape[0] != 0, 'Error: expect a non-empty grid'

        displ_board = [['  ' for _ in range(self.width)]
                       for _ in range(self.height)]
        for i in range(self.height):
            for j in range(self.width):
                if grid[0, i, j] != 0:
                    displ_board[i][j] = 'P' + str(int(grid[0, i, j].item()))
                elif grid[1, i, j] != 0:
                    displ_board[i][j] = 'T' + str(int(grid[1, i, j].item()))
                elif grid[2, i, j] == 1:
                    displ_board[i][j] = 'OO'
        for row in displ_board:
            print(row)

    def get_obs(self):
        '''
        Returns the list of obstacles
        '''
        return self.obs_list

    def get_tgts(self):
        '''
        Returns  the list of targets
        '''
        tgts_list = [[],[],[]]
        for i in range(self.actor_number):
            tgts_list[0].append(self.tgts[i].col)
            tgts_list[1].append(self.tgts[i].row)
            tgts_list[2].append(self.idx[i])
        for i in range(len(self.idx_dones)):
            tgts_list[0].append(self.tgt_dones[i].col)
            tgts_list[1].append(self.tgt_dones[i].row)
            tgts_list[2].append(self.idx_dones[i])
        return tgts_list

    def get_locs(self):
        '''
        Returns the list of current locations of agents
        '''
        locs_list = [[],[],[]]
        for i in range(self.actor_number):
            locs_list[0].append(self.locs[i].col)
            locs_list[1].append(self.locs[i].row)
            locs_list[2].append(self.idx[i])
        for i in range(len(self.idx_dones)):
            locs_list[0].append(self.tgt_dones[i].col)
            locs_list[1].append(self.tgt_dones[i].row)
            locs_list[2].append(self.idx_dones[i])
        return locs_list
    
    def __out_of_boundary(self, loc: Loc):
        '''
        Returns True if the new location is out of the boundary
        '''
        if (loc.row >= self.height) or (loc.row < 0):
            return True
        if (loc.col >= self.width) or (loc.col < 0):
            return True
        return False
    
    def __is_obstacle(self, loc: Loc):
        '''
        Returns True if the new location is an obstacle
        '''
        return self.grid[2, loc.row, loc.col] == 1
    
    def __is_occupied(self, loc: Loc, agent_index: int):
        '''
        Returns True if the new location is occupied by a different agent
            1. this new location is not empty, and
            2. the agent in this new location is not the current agent
        '''
        return (self.grid[0, loc.row, loc.col] != 0) and (self.grid[0, loc.row, loc.col] != agent_index)

    def get_actor_number(self):
        return self.actor_number

    def get_state(self, actor_index: int):
        '''
        Return [4*height*width] tensor
        4*height*width comes from:
            0th layer: current agent location
            1st layer: agent target location
            2nd layer: obstacles
            3rd layer: other agents
        '''
        assert actor_index < self.actor_number, "Invalid actor index"
        state = torch.zeros(size=(4, self.height, self.width))
        state[0, self.locs[actor_index].row, self.locs[actor_index].col] = 1
        state[1, self.tgts[actor_index].row, self.tgts[actor_index].col] = 1
        state[2] = self.grid[2].detach().clone()
        for i in range(len(self.locs)):
            if i != actor_index:
                state[3, self.locs[i].row, self.locs[i].col] = 1
        return state.view(4*self.height*self.width)

    def remove_dones(self):
        '''
        Remove agents from the list if they are at their target location
        '''
        hi_index = self.actor_number-1
        for i in range(hi_index, -1, -1):
            if (self.locs[i] == self.tgts[i]):
                self.grid[0, self.tgts[i].row, self.tgts[i].col] = 0
                self.grid[1, self.tgts[i].row, self.tgts[i].col] = 0
                self.grid[2, self.tgts[i].row, self.tgts[i].col] = 1

                self.idx_dones.append(self.idx[i])
                self.tgt_dones.append(self.tgts[i].copy())
                self.actor_number -= 1

                self.srcs.pop(i)
                self.tgts.pop(i)
                self.locs.pop(i)
                self.idx.pop(i)

    def step(self, actions: list):
        '''
        Take a step based on given actions.
        Returns
            rewards: [batch_size, 1] tensor, an agent can move to a new location only when
                1. new location is not out of bouondary, otherwise, return out_bound_penalty
                2. new location is not a obstacle, otherwise, return obstacle_penalty
                3. new location will not be occupied by another agent, otherwise, return collision_penalty
            dones: a list of boolean values indicate whether agents have reached their targers
        Miscellaneous
            if an agent has been on its target lcoation before taking the action, then this agent will be ignored
        '''
        rewards = [0]*len(actions)
        dones = [False]*len(actions)

        # for each agent, find its new location based on its current location and the action
        new_locs = []
        for i in range(self.actor_number):
            action = actions[i]
            new_loc = self.locs[i].copy()
            if action == 'u' or action == 0:
                new_loc.row -= 1
            elif action == 'd' or action == 1:
                new_loc.row += 1
            elif action == 'l' or action == 2:
                new_loc.col -= 1
            elif action == 'r' or action == 3:
                new_loc.col += 1
            elif action == 's' or action == 4:
                pass
            else:
                raise RuntimeError("Error: unknown move")
                
            new_locs.append(new_loc)

        for i in range(self.actor_number):
            if self.__out_of_boundary(new_locs[i]): # if out of boundary, then do not move this agent, and give a penalty
                new_locs[i].set_from(self.locs[i])
                rewards[i] = self.out_bound_penalty
            elif self.__is_obstacle(new_locs[i]): # if this location is an obstacle, then do not move this agent, and give a penalty
                new_locs[i].set_from(self.locs[i])
                rewards[i] = self.obstacle_penalty
            elif (new_locs[i] == self.locs[i]): # if this agent is stationary, then give a small penalty
                rewards[i] = self.stationary_penalty
        
        is_collsions = [False]*len(actions)
        for i in range(self.actor_number): # check collision
            for j in range(self.actor_number):
                if i==j: continue
                if (new_locs[i]==new_locs[j]) and (new_locs[i]!=self.locs[i]):
                    # type1: move to a location, which is also a new location of a different agent
                    is_collsions[i] = True
                    break
                elif (new_locs[i]==self.locs[j]) and (new_locs[j]!=self.locs[i]):
                    # type2: location swap
                    is_collsions[i] = True
        
        for i in range(self.actor_number): # if collision happens, do not move agents
            if is_collsions[i]:
                new_locs[i].set_from(self.locs[i])
                rewards[i] = self.collision_penalty
                

        for i in range(self.actor_number):
            if (new_locs[i] == self.tgts[i]): # if reach target, then give a reward
                rewards[i] = self.target_reward
        
        # clean old locations on the gird
        for i in range(self.actor_number):
            self.grid[0, self.locs[i].row, self.locs[i].col] = 0
        # add new locations to the grid
        for i in range(self.actor_number):
            self.grid[0, new_locs[i].row, new_locs[i].col] = self.idx[i]
            self.locs[i].set_from(new_locs[i])

        # check done status
        for i in range(self.actor_number):
            dones[i] = (self.locs[i] == self.tgts[i])

        return rewards, dones


def main():
    '''
    Simple test
    '''
    env = MultiActorEnv(actor_number=4, height=10, width=10, obs_count=5, random_seed=100)
    print(env.grid)
    env.display()
    print("-------------------------------------------------------------")
    rewards, dones = env.step(['r', 'l', 'r', 'l'])
    print(env.grid)
    env.display()
    print(rewards)
    print("-------------------------------------------------------------")
    rewards, dones = env.step(['u', 'd', 'u', 'd'])
    print(env.grid)
    env.display()
    print(rewards)
    print("-------------------------------------------------------------")
    rewards, dones = env.step(['u', 'd', 'u', 'd'])
    print(env.grid)
    env.display()
    print(rewards)
    print("-------------------------------------------------------------")
    rewards, dones = env.step(['d', 'u', 'd', 'u'])
    print(env.grid)
    env.display()
    print(rewards)

if __name__ == "__main__":
    main()
