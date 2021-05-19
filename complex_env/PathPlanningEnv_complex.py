import random
import torch
import numpy as np

class PathPlanningEnv():
    def __init__(self, *, height=None, width=None, obs_count=None, random_seed=None,
                 grid=None, init_row=None, init_col=None, goal1_row=None, goal1_col=None, goal2_row=None, goal2_col=None):

        # Initialize from an existing grid is supplied.
        if grid is not None and init_row is not None and init_col is not None\
                and goal1_col is not None:
            self._init_from_grid(grid, init_row, init_col, goal1_row, goal1_col, goal2_row, goal2_col)
            #print("init_from_grid")
        # Initialize a random grid.
        elif height is not None and width is not None and obs_count is not None:
            random_seed = 142 if random_seed is None else random_seed
            self._init_random_grid(height, width, obs_count, random_seed)
            #print("init_random_grid")
        else:
            raise RuntimeError('Error: insufficient arguments.')

        self.actions = [
            torch.Tensor().new_tensor([1, 0, 0, 0], dtype=torch.float32, requires_grad=False), 
            torch.Tensor().new_tensor([0, 1, 0, 0], dtype=torch.float32, requires_grad=False),
            torch.Tensor().new_tensor([0, 0, 1, 0], dtype=torch.float32, requires_grad=False),
            torch.Tensor().new_tensor([0, 0, 0, 1], dtype=torch.float32, requires_grad=False)
        ]
        self.foot_prints = np.zeros((self.height, self.width), dtype=int)
        
    def _init_random_grid(self, height: int, width: int, obs_count: int, random_seed: int):
        # check total number of grid points
        total_size = height*width
        assert total_size >= 2, 'Error: expect height * width >= 2'
        #print("random env")
        # initialize the grid
        self.height = height
        self.width = width
        self.grid_initial = torch.zeros(size=(3, height, width), requires_grad=False)

        # randomly select part of grid points as obstacles with different harzards, and 1 point as source, 1 as target with different rewards
        random.seed(random_seed)
        random_selections = random.sample(range(total_size), k=min(obs_count+2, total_size))

        # source agent location 
        source = random_selections.pop()
        self.init_row, self.init_col = source//width, source % width
        self.grid_initial[0, self.init_row, self.init_col] = 1

        # target locations with differet rewards (3 locations, hard-coded now)
       
        target = random_selections.pop()
        self.grid_initial[1, target//width, target % width] = 1
        self.goal1_row = target//width
        self.goal1_col = target % width
        
        target = random_selections.pop()
        self.grid_initial[1, target//width, target % width] = 2
        self.goal2_row = target//width
        self.goal2_col = target % width
        
        
        # obstacles with different harzard degrees
        for i in random_selections:
            if i % 3 == 0:
                row = i // width
                col = i % width
                self.grid_initial[2, row, col] = 1
            elif i % 3 == 1:
                row = i // width
                col = i % width
                self.grid_initial[2, row, col] = 2
            else:
                row = i // width
                col = i % width
                self.grid_initial[2, row, col] = 3

        self.reset()

    def _init_from_grid(self, obs_grid, p_init_row, p_init_col, goal1_row, goal1_col, goal2_row, goal2_col):
        #print("fixed env")
        self.height = obs_grid.shape[0]
        self.width = obs_grid.shape[1]
        self.grid_initial = torch.zeros(size=(3, self.height, self.width), requires_grad=False)
        for i in range(self.height):
            for j in range(self.width):
                self.grid_initial[2, i, j] = obs_grid[i, j]

        self.init_row, self.init_col = p_init_row, p_init_col
        self.grid_initial[0, self.init_row, self.init_col] = 1

        self.goal1_row = goal1_row
        self.goal1_col = goal1_col
        self.grid_initial[1, self.goal1_row, self.goal1_col] = 1
        
        self.goal2_row = goal2_row
        self.goal2_col = goal2_col
        self.grid_initial[1, self.goal2_row, self.goal2_col] = 2
        
        self.reset()

    def reset(self):
        assert self.grid_initial.shape[0] != 0, 'Error: expect an initial grid'
        assert self.init_row is not None and self.init_col is not None, 'Error: expect a initial position'

        self.grid = self.grid_initial.detach().clone()
        self.cur_row, self.cur_col = self.init_row, self.init_col
        self.done = False

    def display(self, grid=None):
        if grid is None:
            grid = self.grid
        assert grid.shape[0] != 0, 'Error: expect a non-empty grid'

        displ_board = [['  ' for _ in range(self.width)]
                       for _ in range(self.height)]
        for i in range(self.height):
            for j in range(self.width):
                if grid[0, i, j] == 1:
                    displ_board[i][j] = 'P '
                elif grid[1, i, j] == 1:
                    #displ_board[i][j] = grid[1, i, j].item()
                    displ_board[i][j] = 'T1'
                elif grid[1, i, j] == 2:
                    #displ_board[i][j] = grid[1, i, j].item()
                    displ_board[i][j] = 'T2'

                elif grid[2, i, j] == 1:
                    displ_board[i][j] = 'O '
                elif grid[2, i, j] == 2:
                    displ_board[i][j] = 'H1'
                elif grid[2, i, j] == 3:
                    displ_board[i][j] = 'H2'
        for row in displ_board:
            print(row)


    def step(self, action, early_stop=True, q_learning=True):
        #print(q_learning)
        self.done = False
        new_row = self.cur_row
        new_col = self.cur_col

        self.foot_prints[self.cur_row, self.cur_col] += 1

        if action == 'u' or action == 0:
            new_row -= 1
        elif action == 'd' or action == 1:
            new_row += 1
        elif action == 'l' or action == 2:
            new_col -= 1
        elif action == 'r' or action == 3:
            new_col += 1
        else:
            raise RuntimeError("Error: unknown move")

        if not 0 <= new_row < self.height or not 0 <= new_col < self.width:
            reward = -0.6 if q_learning else -1
            self.done = early_stop
        elif self.grid[2, new_row, new_col] == 1:
            reward = -0.6 if q_learning else -1
            self.done = early_stop
            
        # harzards
        elif self.grid[2, new_row, new_col] == 2: # obs
            reward = -0.15 if q_learning else -1
            self.done = early_stop
        elif self.grid[2, new_row, new_col] == 3: # obs
            reward = -0.25 if q_learning else -1
            self.done = early_stop
            
        else:
            if q_learning:
                reward = -0.1


            self.grid[0, self.cur_row, self.cur_col] = 0
            self.grid[0, new_row, new_col] = 1

            self.cur_row = new_row
            self.cur_col = new_col

        if (new_row == self.goal1_row and new_col == self.goal1_col):  # reach the target
            #self.foot_prints[self.goal1_row, self.goal1_col] += 1
            reward = 1 if q_learning else reward
            self.done = True
        if (new_row == self.goal2_row and new_col == self.goal2_col):  # reach the target
            reward = 1.5 if q_learning else reward
            self.done = True

            
        observation = self.grid

        info = ""

        return observation, reward, self.done, info


def main():
    env = PathPlanningEnv(height=5, width=10, obs_count=5, random_seed=100)
    print(env.grid)
    env.display()
    print(env.distances)
    for _ in range(10):
        _, reward, done, _ = env.step('r')
        env.display()
        print("reward: {}, done: {}\n".format(reward, done))


if __name__ == "__main__":
    main()
