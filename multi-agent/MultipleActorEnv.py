import random
import torch
import numpy as np


class MultiActorEnv():
    def __init__(self, *,
                 agent_count,
                 height, width, obs_count, grid=None,
                 sources=None, goals=None,
                 random_seed=None):

        self.agent_count = agent_count
        self.height = height
        self.width = width
        self.obs_count = obs_count

        self.sources = list()
        self.targets = list()
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
            source = [num//width, num % width]
            self.sources.append(source)
            self.grid_initial[0, source[0], source[1]] = 1

            num = random_selections.pop()
            target = [num//width, num % width]
            self.targets.append(target)
            self.grid_initial[1, target[0], target[1]] = 1

        for i in random_selections:
            row = i // width
            col = i % width
            self.grid_initial[2, row, col] = 1

        self.reset()

    def _init_from_grid(self, obs_grid, p_init_row, p_init_col, goal_row, goal_col):
        self.height = obs_grid.shape[0]
        self.width = obs_grid.shape[1]
        self.grid_initial = torch.zeros(
            size=(3, self.height, self.width), requires_grad=False)
        for i in range(self.height):
            for j in range(self.width):
                self.grid_initial[2, i, j] = obs_grid[i, j]

        self.init_row, self.init_col = p_init_row, p_init_col
        self.grid_initial[0, self.init_row, self.init_col] = 1

        self.goal_row = goal_row
        self.goal_col = goal_col
        self.grid_initial[1, self.goal_row, self.goal_col] = 1

        self.reset()

    def reset(self):
        self.grid = self.grid_initial.detach().clone()
        self.done = False

    def random_reset(self):
        assert self.grid_initial.shape[0] != 0, 'Error: expect an initial grid'
        assert self.init_row is not None and self.init_col is not None, 'Error: expect a initial position'

        initials = [(0, 0), (0, self.width-1), (self.height-1, 0),
                    (self.height-1, self.width-1)]

        self.grid = self.grid_initial.detach().clone()
        self.cur_row, self.cur_col = random.choice(initials)
        self.done = False
        self.trace = [(self.cur_row, self.cur_row)]

    def display(self, grid=None):
        if grid is None:
            grid = self.grid
        assert grid.shape[0] != 0, 'Error: expect a non-empty grid'

        displ_board = [[' ' for _ in range(self.width)]
                       for _ in range(self.height)]
        for i in range(self.height):
            for j in range(self.width):
                if grid[0, i, j] == 1:
                    displ_board[i][j] = 'P'
                elif grid[1, i, j] == 1:
                    displ_board[i][j] = 'T'
                elif grid[2, i, j] == 1:
                    displ_board[i][j] = 'O'
        for row in displ_board:
            print(row)

    def step(self, action, early_stop=True, q_learning=False):
        return 1


def main():
    env = MultiActorEnv(agent_count=2, height=5, width=10, obs_count=5, random_seed=100)
    print(env.grid)
    env.display()

if __name__ == "__main__":
    main()
