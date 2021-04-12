import random
import torch
import numpy as np

from AStar import AStar


class PathPlanningEnv():
    def __init__(self, *, height=None, width=None, obs_count=None, random_seed=None,
                 grid=None, init_row=None, init_col=None, goal_row=None, goal_col=None):

        # Initialize from an existing grid is supplied.
        if grid is not None and init_row is not None and init_col is not None\
                and goal_col is not None:
            self._init_from_grid(grid, init_row, init_col, goal_row, goal_col)
        # Initialize a random grid.
        elif height is not None and width is not None and obs_count is not None:
            random_seed = 42 if random_seed is None else random_seed
            self._init_random_grid(height, width, obs_count, random_seed)
        else:
            raise RuntimeError('Error: insufficient arguments.')

        self._compute_all_distances()
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

        # initialize the grid
        self.height = height
        self.width = width
        self.grid_initial = torch.zeros(size=(3, height, width), requires_grad=False)

        # randomly select part of grid points as obstacles, and 1 point as source, 1 as target
        random.seed(random_seed)
        random_selections = random.sample(range(total_size), k=min(obs_count+2, total_size))

        source = random_selections.pop()
        self.init_row, self.init_col = source//width, source % width
        self.grid_initial[0, self.init_row, self.init_col] = 1

        target = random_selections.pop()
        self.grid_initial[1, target//width, target % width] = 1
        self.goal_row = target//width
        self.goal_col = target % width

        for i in random_selections:
            row = i // width
            col = i % width
            self.grid_initial[2, row, col] = 1

        self.reset()

    def _init_from_grid(self, obs_grid, p_init_row, p_init_col, goal_row, goal_col):
        self.height = obs_grid.shape[0]
        self.width = obs_grid.shape[1]
        self.grid_initial = torch.zeros(size=(3, self.height, self.width), requires_grad=False)
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
        assert self.grid_initial.shape[0] != 0, 'Error: expect an initial grid'
        assert self.init_row is not None and self.init_col is not None, 'Error: expect a initial position'

        self.grid = self.grid_initial.detach().clone()
        self.cur_row, self.cur_col = self.init_row, self.init_col
        self.done = False
        self.trace = [(self.cur_row, self.cur_row)]

    def random_reset(self):
        assert self.grid_initial.shape[0] != 0, 'Error: expect an initial grid'
        assert self.init_row is not None and self.init_col is not None, 'Error: expect a initial position'
        
        initials = [(0, 0), (0, self.width-1), (self.height-1, 0), (self.height-1, self.width-1)]

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

    def _compute_all_distances(self, silent=True):
        assert self.grid_initial.shape[0] != 0, 'Error: expect a non-empty grid'

        planner = AStar(
            self.grid_initial[2, :, :], (self.goal_row, self.goal_col), False)
        if not silent:
            print(self.grid_initial[2, :, :])
        self.distances = torch.empty(size=(self.height, self.width))
        for i in range(self.height):
            for j in range(self.width):
                if not silent:
                    print("Evaluate distance from ({}, {}) to ({}, {})".format(i, j, self.goal_row, self.goal_col))
                distance = 100
                if (self.grid_initial[2, i, j] == 0):
                    path = planner.plan(i, j)
                    distance = len(path)
                if not silent:
                    print("  Distance {}".format(distance))
                self.distances[i, j] = distance

    def step(self, action, early_stop=True, q_learning=False):
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
            reward = -0.5 if q_learning else -1
            self.done = early_stop
        elif self.grid[2, new_row, new_col] == 1:
            reward = -0.5 if q_learning else -1
            self.done = early_stop
        else:
            if q_learning:
                reward = 0
            else:
                old_distance = self.distances[self.cur_row, self.cur_col]
                new_distance = self.distances[new_row, new_col]
                reward = old_distance - new_distance

            self.grid[0, self.cur_row, self.cur_col] = 0
            self.grid[0, new_row, new_col] = 1

            self.cur_row = new_row
            self.cur_col = new_col

        if (new_row == self.goal_row and new_col == self.goal_col):  # reach the target
            self.foot_prints[self.goal_row, self.goal_col] += 1
            reward = 1 if q_learning else reward
            self.done = True

        self.trace.append((self.cur_row, self.cur_col))

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
