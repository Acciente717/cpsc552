import heapq

class AStar:
    def __init__(self, grid, goal, allow_diagonal=False):
        '''
        Create an object to perform A* search.
        
        Parameters:
        grid: an 2D array representing the map. 1 represents obstacles, 0 free block.
        goal: a tuple (x, y) denoting the coordinate of the goal.
        allow_diagonal: whether diagonal movement is allowed. Diagonal movements have cost 1.
        '''
        self.grid = grid
        self.height = len(grid)
        self.width = len(grid[0])
        self.goalx, self.goaly = goal
        self.allow_diagonal = allow_diagonal
        if allow_diagonal:
            self.heauristic_func = lambda curx, cury: min(abs(curx - self.goalx), abs(cury- self.goaly))
        else:
            self.heauristic_func = lambda curx, cury: abs(curx - self.goalx) + abs(cury - self.goaly)

    def plan(self, startx, starty):
        '''
        Run a path searching.
        
        Parameters:
        startx, starty: the starting coordinate.
        
        Return:
        A list containing the path from the starting point to the goal.
        '''
        if self.grid[startx][starty] == 1:
            return []

        push_count = 0
        predecessors = [[None for _ in range(self.width)] for _ in range(self.height)]
        costs = [[None for _ in range(self.width)] for _ in range(self.height)]

        heap = []
        heapq.heappush(heap, (self.heauristic_func(startx, starty), push_count, (startx, starty), 0, None))

        while heap:
            current = heapq.heappop(heap)
            cost, _, (x, y), distance, predecessor = current

            if (x, y) == (self.goalx, self.goaly):
                predecessors[x][y] = predecessor
                break

            if costs[x][y] is None or cost < costs[x][y]:
                predecessors[x][y] = predecessor
                costs[x][y] = cost
                distance += 1

                if x >= 1 and self.grid[x - 1][y] == 0:
                    push_count += 1
                    h = self.heauristic_func(x - 1, y)
                    cost = distance + h
                    heapq.heappush(heap, (cost, push_count, (x - 1, y), distance, (x, y)))
                if y >= 1 and self.grid[x][y - 1] == 0:
                    push_count += 1
                    h = self.heauristic_func(x, y - 1)
                    cost = distance + h
                    heapq.heappush(heap, (cost, push_count, (x, y - 1), distance, (x, y)))
                if x <= self.width - 2 and self.grid[x + 1][y] == 0:
                    push_count += 1
                    h = self.heauristic_func(x + 1, y)
                    cost = distance + h
                    heapq.heappush(heap, (cost, push_count, (x + 1, y), distance, (x, y)))
                if y <= self.width - 2 and self.grid[x][y + 1] == 0:
                    push_count += 1
                    h = self.heauristic_func(x, y + 1)
                    cost = distance + h
                    heapq.heappush(heap, (cost, push_count, (x, y + 1), distance, (x, y)))

                if self.allow_diagonal:
                    if x >= 1 and y >= 1 and self.grid[x - 1][y - 1] == 0:
                        push_count += 1
                        h = self.heauristic_func(x - 1, y - 1)
                        cost = distance + h
                        heapq.heappush(heap, (cost, push_count, (x - 1, y - 1), distance, (x, y)))
                    if x >= 1 and y <= self.height - 2 and self.grid[x - 1][y + 1] == 0:
                        push_count += 1
                        h = self.heauristic_func(x - 1, y + 1)
                        cost = distance + h
                        heapq.heappush(heap, (cost, push_count, (x - 1, y + 1), distance, (x, y)))
                    if x <= self.width - 2 and y >= 1 and self.grid[x + 1][y - 1] == 0:
                        push_count += 1
                        h = self.heauristic_func(x + 1, y - 1)
                        cost = distance + h
                        heapq.heappush(heap, (cost, push_count, (x + 1, y - 1), distance, (x, y)))
                    if x <= self.width - 2 and y <= self.height - 2 and self.grid[x + 1][y + 1] == 0:
                        push_count += 1
                        h = self.heauristic_func(x + 1, y + 1)
                        cost = distance + h
                        heapq.heappush(heap, (cost, push_count, (x + 1, y + 1), distance, (x, y)))

        path = []
        curx, cury = self.goalx, self.goaly
        while predecessors[curx][cury] is not None:
            path.append((curx, cury))
            curx, cury = predecessors[curx][cury]
        path.reverse()
        return path
