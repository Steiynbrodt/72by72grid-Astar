import pygame
import heapq

# -----------------------
# Grid settings
# -----------------------
GRID_SIZE = 72
CELL_SIZE = 12  # GUI pixel size (50 mm irrelevant for rendering)
WIDTH = GRID_SIZE * CELL_SIZE
HEIGHT = GRID_SIZE * CELL_SIZE

pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("A* Pathfinding Interactive Grid")

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (180, 180, 180)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
BLUE = (0, 128, 255)
YELLOW = (255, 255, 0)

# Grid structure
grid = [[0 for _ in range(GRID_SIZE)] for _ in range(GRID_SIZE)]
start = None
goal = None
waypoints = []
path = []

# -----------------------
# Helper Functions
# -----------------------
def heuristic(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def a_star(start, goal, obstacles):
    """Basic A* implementation on grid."""
    open_set = [(0, start)]
    came_from = {}
    g_score = {start: 0}

    while open_set:
        _, current = heapq.heappop(open_set)

        if current == goal:
            # Reconstruct path
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            return path[::-1]

        x, y = current
        neighbors = [(x+1,y), (x-1,y), (x,y+1), (x,y-1)]

        for nx, ny in neighbors:
            if 0 <= nx < GRID_SIZE and 0 <= ny < GRID_SIZE:
                if (nx, ny) in obstacles:
                    continue

                tentative_g = g_score[current] + 1

                if (nx, ny) not in g_score or tentative_g < g_score[(nx, ny)]:
                    g_score[(nx, ny)] = tentative_g
                    priority = tentative_g + heuristic((nx, ny), goal)
                    heapq.heappush(open_set, (priority, (nx, ny)))
                    came_from[(nx, ny)] = current

    return None  # No path found

def compute_total_path():
    global path
    if not start or not goal:
        path = []
        return

    obstacles = {(x, y) for x in range(GRID_SIZE) for y in range(GRID_SIZE) if grid[y][x] == 1}

    full_path = []
    current = start

    for wp in waypoints + [goal]:
        segment = a_star(current, wp, obstacles)
        if not segment:
            full_path = []
            return
        if full_path:
            segment = segment[1:]
        full_path.extend(segment)
        current = wp

    path = full_path

# -----------------------
# Main drawing function
# -----------------------
def draw():
    screen.fill(WHITE)

    # Obstacles
    for y in range(GRID_SIZE):
        for x in range(GRID_SIZE):
            if grid[y][x] == 1:
                pygame.draw.rect(screen, BLACK, (x*CELL_SIZE, y*CELL_SIZE, CELL_SIZE, CELL_SIZE))

    # Path
    for (x, y) in path:
        pygame.draw.rect(screen, YELLOW, (x*CELL_SIZE, y*CELL_SIZE, CELL_SIZE, CELL_SIZE))

    # Start
    if start:
        pygame.draw.rect(screen, GREEN, (start[0]*CELL_SIZE, start[1]*CELL_SIZE, CELL_SIZE, CELL_SIZE))

    # Goal
    if goal:
        pygame.draw.rect(screen, RED, (goal[0]*CELL_SIZE, goal[1]*CELL_SIZE, CELL_SIZE, CELL_SIZE))

    # Waypoints
    for (x, y) in waypoints:
        pygame.draw.rect(screen, BLUE, (x*CELL_SIZE, y*CELL_SIZE, CELL_SIZE, CELL_SIZE))

    # Grid lines
    for x in range(GRID_SIZE):
        pygame.draw.line(screen, GRAY, (x*CELL_SIZE, 0), (x*CELL_SIZE, HEIGHT))
    for y in range(GRID_SIZE):
        pygame.draw.line(screen, GRAY, (0, y*CELL_SIZE), (WIDTH, y*CELL_SIZE))

    pygame.display.flip()

# -----------------------
# Main Loop
# -----------------------
running = True
while running:
    draw()

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        if pygame.mouse.get_pressed()[0]:  # Left click
            x, y = pygame.mouse.get_pos()
            gx, gy = x // CELL_SIZE, y // CELL_SIZE

            mods = pygame.key.get_mods()
            if mods & pygame.KMOD_SHIFT:  # Set start
                start = (gx, gy)
                compute_total_path()
            else:
                grid[gy][gx] = 1  # obstacle
                compute_total_path()

        if pygame.mouse.get_pressed()[2]:  # Right click
            x, y = pygame.mouse.get_pos()
            gx, gy = x // CELL_SIZE, y // CELL_SIZE
            mods = pygame.key.get_mods()
            if mods & pygame.KMOD_SHIFT:  # Set goal
                goal = (gx, gy)
                compute_total_path()
            else:
                grid[gy][gx] = 0  # remove obstacle
                compute_total_path()

        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 2:  # Middle click
            x, y = event.pos
            gx, gy = x // CELL_SIZE, y // CELL_SIZE
            waypoints.append((gx, gy))
            compute_total_path()

pygame.quit()
