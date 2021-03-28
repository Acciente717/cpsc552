import numpy as np
import random
import gym
from gym import spaces

from AStar import AStar

class PathPlanningEnv(gym.Env):
    def __init__(self, discrete_space_size = 4):
        super(PathPlanningEnv, self).__init__()
        self.action_space = {'u':0, 'd':1, 'l':2, 'r':3}
        
        self.grid_initial = np.empty(shape=(0, 0, 0))
        self.init_row = None
        self.init_col = None
        self.goal = None
        
        self.grid = np.empty(shape=(0, 0, 0))
        self.cur_row = None
        self.cur_col = None
    
    def init_random_grid(self, height: int, width: int, obs_count: int, random_seed: int):
        # check total number of grid points
        total_size = height*width

        assert total_size >= 2, 'Error: expect height * width >= 2'
        
        
        # initialize the grid
        self.height = height
        self.width = width
        self.grid_initial = np.empty(shape=(3, height, width))
        self.grid_initial.fill(0)
        # randomly select part of grid points as obstacles, and 1 point as source, 1 as target
        random.seed(random_seed)
        random_selections = random.sample(range(total_size), k=min(obs_count+2, total_size))
        
        source = random_selections.pop()
        self.init_row, self.init_col  = source//width, source%width
        self.grid_initial[0, self.init_row, self.init_col] = 1
        
        target = random_selections.pop()
        self.grid_initial[1, target//width, target%width] = 1
        self.goal = (target//width, target%width)
        
        for i in random_selections:
            row = i // width
            col = i % width
            #print("{}: ({}, {})".format(i, row, col))
            self.grid_initial[2,row,col] = 1
            
        self.reset()

    def init_from_grid(self, obs_grid, p_init_row, p_init_col, goal_row, goal_col):
        self.height = obs_grid.shape[0]
        self.width = obs_grid.shape[1]
        self.grid_initial = np.empty(shape=(3, self.height, self.width))
        self.grid_initial.fill(0)
        for i in range(self.height):
            for j in range(self.width):
                self.grid_initial[2,i,j] = obs_grid[i,j]

        self.init_row, self.init_col = p_init_row, p_init_col
        self.grid_initial[0, self.init_row, self.init_col] = 1

        self.goal = (goal_row, goal_col)
        self.grid_initial[1, self.goal[0], self.goal[1]] = 1

        self.reset()

        
    def reset(self):
        assert self.grid_initial.shape[0] != 0, 'Error: expect an initial grid'
        assert self.init_row is not None and self.init_col is not None, 'Error: expect a initial position'

        self.grid = np.copy(self.grid_initial)
        self.cur_row, self.cur_col = self.init_row, self.init_col
    
    def display(self, grid=None):
        if grid is None:
            grid = self.grid
        assert grid.shape[0] != 0, 'Error: expect a non-empty grid'

        displ_board = np.zeros((self.height, self.width), dtype='<U2')
        displ_board.fill(' ')
        for i in range(self.height):
            for j in range(self.width):
                if grid[0, i, j]==1:
                    displ_board[i,j] = 'P'
                elif grid[1, i, j]==1:
                    displ_board[i,j] = 'T'
                elif grid[2, i, j]==1:
                    displ_board[i,j] = 'O'
        print(displ_board)
    
    def ComputeAllDistance(self, silent=True):
        assert self.grid_initial.shape[0] != 0, 'Error: expect a non-empty grid'

        planner = AStar(self.grid_initial[2,:,:], self.goal, False)
        if not silent: print(self.grid_initial[2,:,:])
        self.distances = np.empty(shape=(self.height, self.width))
        for i in range(self.height):
            for j in range(self.width):
                if not silent: print("Evaluate distance from ({}, {}) to ({}, {})".format(i, j, self.goal[0], self.goal[1]))
                distance = 100
                if (self.grid_initial[2, i, j] == 0):
                    path = planner.plan(i, j)
                    distance = len(path)
                if not silent: print("  Distance {}".format(distance))
                self.distances[i,j] = distance

    
    def step(self, action, early_stop=True):
        done = False
        old_distance = self.distances[self.cur_row, self.cur_col]
        new_row = self.cur_row
        new_col = self.cur_col
        if action=='u' or action==0:
            new_row -= 1
        elif action=='d' or action==1:
            new_row += 1
        elif action=='l' or action==2:
            new_col -= 1
        elif action=='r' or action==3:
            new_col += 1
        else:
            raise RuntimeError("Error: unknown move")

        if not 0 <= new_row < self.height or not 0 <= new_col < self.width:
            reward = -10
            done = early_stop
        elif self.grid[2, new_row, new_col] == 1:
            reward = -10
            done = early_stop
        else:
            new_distance = self.distances[new_row, new_col]
            reward = old_distance - new_distance
            
            self.grid[0, self.cur_row, self.cur_col] = 0
            self.grid[0, new_row, new_col] = 1

            self.cur_row = new_row
            self.cur_col = new_col

        if (new_row == self.goal[0] and new_col == self.goal[1]): # reach the target
            done = True
        
        observation = self.grid
        
        info = ""

        return observation, reward, done, info


def main():
    env = PathPlanningEnv()
    env.init_random_grid(5, 10, 5, 100)
    print(env.grid)
    env.display()
    env.ComputeAllDistance()
    print(env.distances)
    for _ in range(10):
        _, reward, done, _ = env.step('r')
        env.display()
        print("reward: {}, done: {}\n".format(reward, done))
        

if __name__ == "__main__":
    main()
