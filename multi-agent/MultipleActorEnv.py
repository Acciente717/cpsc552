import random
import torch
import numpy as np

class Loc():
    def __init__(self, row, col):
        self.row = row
        self.col = col

    def copy(self):
        return Loc(self.row, self.col)

    def copy_from(self, loc):
        self.row = loc.row
        self.col = loc.col

    def __eq__(self, other):
        return (self.row == other.row) and (self.col == other.col)

class MultiActorEnv():
    def __init__(self, *,
                 agent_count,
                 height, width, obs_count, grid=None,
                 srcs=None, tgts=None,
                 random_seed=None):

        self.agent_count = agent_count
        self.height = height
        self.width = width
        self.obs_count = obs_count

        self.srcs = list()
        self.tgts = list()
        self.locs = list()
        if random_seed is None:
            random_seed = 100
        
        self._init_random_grid(self.agent_count, self.height, self.width, self.obs_count, random_seed)

        self.actions = [
            torch.Tensor().new_tensor([1, 0, 0, 0, 0], dtype=torch.float32, requires_grad=False),  # up
            torch.Tensor().new_tensor([0, 1, 0, 0, 0], dtype=torch.float32, requires_grad=False),  # down
            torch.Tensor().new_tensor([0, 0, 1, 0, 0], dtype=torch.float32, requires_grad=False),  # left
            torch.Tensor().new_tensor([0, 0, 0, 1, 0], dtype=torch.float32, requires_grad=False),  # right
            torch.Tensor().new_tensor([0, 0, 0, 0, 1], dtype=torch.float32, requires_grad=False)   # stay still
        ]
        self.target_reward = 1
        self.out_bound_penalty = -0.5
        self.obstacle_penalty = -0.5
        self.collision_penalty = -0.5

    def _init_random_grid(self, agent_count: int, height: int, width: int, obs_count: int, random_seed: int):
        # check total number of grid points
        total_size = height*width
        assert total_size >= 2, 'Error: expect height * width >= 2'

        # initialize the grid
        self.grid_initial = torch.zeros(size=(3, height, width), requires_grad=False)

        # randomly select part of grid points as obstacles, and agent_count (source, target) pair
        random.seed(random_seed)
        random_selections = random.sample(range(total_size), k=min(obs_count+2*agent_count, total_size))

        for i in range(agent_count):
            num = random_selections.pop()
            source = Loc(num//width, num % width)
            self.srcs.append(source)
            self.grid_initial[0, source.row, source.col] = i + 1

            num = random_selections.pop()
            target = Loc(num//width, num % width)
            self.tgts.append(target)
            self.grid_initial[1, target.row, target.col] = i + 1

        for num in random_selections:
            row = num // width
            col = num % width
            self.grid_initial[2, row, col] = 1

        self.reset()

    def reset(self):
        self.grid = self.grid_initial.detach().clone()
        self.locs = self.srcs.copy()

    def display(self, grid=None):
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

    def step(self, actions: list):
        '''
        Take a step based on given actions.
        Returns
            observations: a list of [3, height, weight] tensors
                0th layer: current agent location
                1st layer: current agent location
                2nd layer: obstacles and other agents
            rewards: a list of reward, an agent can move to a new location only when
                1. new location is not out of bouondary, otherwise, return out_bound_penalty
                2. new location is not a obstacle, otherwise, return obstacle_penalty
                3. new location is currently not occupied by another agent, otherwise, return collision_penalty
            dones: a list of boolean values indicate whether agents have reached their targers
        Miscellaneous
            if an agent has been on its target lcoation before taking the action, then this agent will be ignored
        '''
        assert len(actions) == self.agent_count, "Error: #actions is incorrect"
        rewards = [0]*self.agent_count
        dones = [False]*self.agent_count

        # for each agent, find its new location based on its current location and the action
        new_locs = []
        for i in range(self.agent_count):
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

        for i in range(self.agent_count):
            if self.__out_of_boundary(new_locs[i]): # if out of boundary, then do not move this agent, and give a penalty
                new_locs[i].copy_from(self.locs[i])
                rewards[i] = self.out_bound_penalty
            elif self.__is_obstacle(new_locs[i]): # if this location is an obstacle, then do not move this agent, and give a penalty
                new_locs[i].copy_from(self.locs[i])
                rewards[i] = self.obstacle_penalty
            elif self.__is_occupied(new_locs[i], i+1): # if this location is occupised, then do not move this agent, and give a penalty
                new_locs[i].copy_from(self.locs[i])
                rewards[i] = self.obstacle_penalty
            else:
                if (new_locs[i] == self.tgts[i]):
                    rewards[i] = self.target_reward
        
        # clean old locations on the gird
        for i in range(self.agent_count):
            self.grid[0, self.locs[i].row, self.locs[i].col] = 0
        # add new locations to the grid
        for i in range(self.agent_count):
            self.grid[0, new_locs[i].row, new_locs[i].col] = i + 1 
            self.locs[i].copy_from(new_locs[i])

        observations = self.grid

        info = ""

        return observations, rewards, dones, info


def main():
    env = MultiActorEnv(agent_count=2, height=5, width=10, obs_count=5, random_seed=100)
    print(env.grid)
    env.display()
    print("-------------------------------------------------------------")
    env.step(['l', 'l'])
    print(env.grid)
    env.display()
    print("-------------------------------------------------------------")
    env.step(['u', 'd'])
    print(env.grid)
    env.display()
    print("-------------------------------------------------------------")
    env.step(['u', 'd'])
    print(env.grid)
    env.display()

if __name__ == "__main__":
    main()
