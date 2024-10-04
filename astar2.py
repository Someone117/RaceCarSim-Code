def get_angle_change(from_pos: tuple[int, int], int_pos: tuple[int, int], to_pos: tuple[int, int]) -> float:
    """Get the angle change from one position to another."""
    v1 = (int_pos[0] - from_pos[0], int_pos[1] - from_pos[1])
    v2 = (to_pos[0] - int_pos[0], to_pos[1] - int_pos[1])

    cross_product = v1[0] * v2[1] - v1[1] * v2[0]
    dot_product = v1[0] * v2[0] + v1[1] * v2[1]

    angle_radians = math.atan2(cross_product, dot_product)
    angle_degrees = math.degrees(angle_radians)

    return angle_degrees

def astar2(start_pos: tuple[int, int], goal_pos: tuple[int, int], grid) -> list[tuple[int, int]]:
    """A* algorithm to find the shortest path from start to goal."""

    INF = float('inf')

    def h(a, b):
        """ A* Heuristic """
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def visit_cost(past_stack: list[tuple[int, int]], curr: tuple[int, int], next: tuple[int, int]) -> float:
        pxy_last = past_stack[-1] if past_stack else None
        pxy_mid = past_stack[len(past_stack) // 2] if past_stack else None
        cx, cy = curr
        nx, ny = next
        downwards = ny > cy

        angle = abs(get_angle_change(pxy_last, curr, next) if pxy_last and pxy_mid else 0)
        direct_cost = 100 if downwards else (1 + angle / 10)

        c = grid[cx, cy]
        n = grid[nx, ny]

        if c == 1 or c == 2:
            if n == 2:
                return 1000 * direct_cost
            if n == 0 or n is None:
                return direct_cost

        if c == 0 or c is None:
            if n == 0 or n is None:
                return direct_cost

        return -1

    def get_neighbours(past, pos):
        x, y = pos
        neighbours = [
            (x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1),
            (x + 1, y + 1), (x - 1, y - 1),
            (x + 1, y - 1), (x - 1, y + 1)
        ]
        neighbours = map(lambda n: (n[0], n[1], visit_cost(past, pos, n)), neighbours)
        neighbours = filter(lambda n: n[2] >= 0, neighbours)
        return list(neighbours)

    def reconstruct_path_stack(came_from, current, limit = None):
        path = []
        while (current in came_from) and (limit is None or len(path) < limit):
            path.append(current)
            current = came_from[current]
        return path

    open_set = {start_pos}
    came_from = {}
    g_score = {start_pos: 0.0}
    f_score = {start_pos: h(start_pos, goal_pos)}

    while open_set:
        current = min(open_set, key=lambda x: f_score[x])

        if current == goal_pos or not (current[0] > len(grid) or current[1] > len(grid[0])):
            print("Path found!")

            path_stack = reconstruct_path_stack(came_from, current)
            path_stack.reverse()
            return path_stack

        open_set.remove(current)

        past_path = reconstruct_path_stack(came_from, current, ANGLE_BACKTRACK_UNITS)
        for neighbour in get_neighbours(past_path, current):
            nx, ny, cost = neighbour
            tentative_g_score = g_score[current] + cost
            if tentative_g_score < g_score.get((nx, ny), INF):
                came_from[nx, ny] = current
                g_score[nx, ny] = tentative_g_score
                f_score[nx, ny] = tentative_g_score + h((nx, ny), goal_pos)
                if (nx, ny) not in open_set:
                    open_set.add((nx, ny))

    print("No path found!")
    return []
