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
        if (total_size < 2):
            print("Error: expect height * width >= 2")
            return
        
        # initialize the grid
        self.height = height
        self.width = width
        self.grid_initial = np.empty(shape=(3, height, width))
        self.grid_initial.fill(0)
        # randomly select part of grid points as obstacles, and 1 point as source, 1 as target
        random.seed(100)
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
        
    def reset(self):
        if self.grid_initial.shape[0] == 0:
            print("Error: expect an initial grid")
            return
        if self.init_row == None or self.init_col == None:
            print("Error: expect a initial position")
            return
        self.grid = np.copy(self.grid_initial)
        self.cur_row, self.cur_col = self.init_row, self.init_col
    
    def display(self, grid=None):
        if grid is None:
            grid = self.grid
        if grid.shape[0] == 0:
            return
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
    
    def ComputeAllDistance(self):
        if self.grid_initial.shape[0] == 0: return
        planner = AStar(self.grid_initial[2,:,:], self.goal, False)
        print(self.grid_initial[2,:,:])
        self.distances = np.empty(shape=(self.height, self.width))
        for i in range(self.height):
            for j in range(self.width):
                print("Evaluate distance from ({}, {}) to ({}, {})".format(i, j, self.goal[0], self.goal[1]))
                distance = 100000
                if (self.grid_initial[2, i, j] == 0):
                    path = planner.plan(i, j)
                    distance = len(path)
                print("  Distance {}".format(distance))
                self.distances[i,j] = distance

    
    def step(self, action):
        if action=='u' or action==0:
            if self.cur_row > 0:
                self.grid[0, self.cur_row, self.cur_col] = 0
                self.cur_row -= 1
                self.grid[0, self.cur_row, self.cur_col] = 1
        elif action=='d' or action==1:
            if self.cur_row < self.height - 1:
                self.grid[0, self.cur_row, self.cur_col] = 0
                self.cur_row += 1
                self.grid[0, self.cur_row, self.cur_col] = 1
        elif action=='l' or action==2:
            if self.cur_col > 0:
                self.grid[0, self.cur_row, self.cur_col] = 0
                self.cur_col -= 1
                self.grid[0, self.cur_row, self.cur_col] = 1
        elif action=='r' or action==3:
            if self.cur_col < self.width - 1:
                self.grid[0, self.cur_row, self.cur_col] = 0
                self.cur_col += 1
                self.grid[0, self.cur_row, self.cur_col] = 1
        else:
            print("ERROR: unknown move")
            return
        
        
        observation = self.grid
        
        reward = 0
        
        done = False
        
        info = ""
        return observation, reward, done, info


def main():
    env = PathPlanningEnv()
    env.init_random_grid(5, 10, 5, 100)
    print(env.grid)
    env.display()
    env.ComputeAllDistance()
    print(env.distances)

if __name__ == "__main__":
    main()
