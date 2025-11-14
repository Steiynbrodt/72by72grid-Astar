import pygame
import heapq

# -----------------------
# Field / Grid / GPS settings
# -----------------------
GRID_SIZE = 72
CELL_SIZE = 12  # GUI pixel size

# VEX field in mm (approx. -1800 .. +1800)
FIELD_SIZE_MM = 3600
FIELD_HALF_MM = FIELD_SIZE_MM / 2
CELL_MM = 50  # each cell represents 50 mm

WIDTH = GRID_SIZE * CELL_SIZE
HEIGHT = GRID_SIZE * CELL_SIZE

pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("A* Pathfinding with VEX GPS Mapping")

pygame.font.init()
font = pygame.font.SysFont(None, 18)

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
# GPS <-> Grid conversion
# -----------------------
def clamp(val, lo, hi):
    return max(lo, min(hi, val))

def gps_to_grid(x_mm, y_mm):
    """
    Convert VEX GPS coordinates (mm, origin center, +Y up)
    to grid indices (gx, gy) 0..71 with gy downwards.
    """
    gx = int((x_mm + FIELD_HALF_MM) // CELL_MM)
    gy = int((FIELD_HALF_MM - y_mm) // CELL_MM)

    gx = clamp(gx, 0, GRID_SIZE - 1)
    gy = clamp(gy, 0, GRID_SIZE - 1)
    return gx, gy

def grid_to_gps(gx, gy):
    """
    Convert grid indices (gx, gy) 0..71 to GPS mm coordinates.
    Returns center of the cell.
    """
    x_mm = (gx + 0.5) * CELL_MM - FIELD_HALF_MM
    y_mm = FIELD_HALF_MM - (gy + 0.5) * CELL_MM
    return x_mm, y_mm

# -----------------------
# A* Pathfinding
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
# Drawing
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

    # Mouse position overlay: grid + GPS mm
    mx, my = pygame.mouse.get_pos()
    gx = mx // CELL_SIZE
    gy = my // CELL_SIZE
    if 0 <= gx < GRID_SIZE and 0 <= gy < GRID_SIZE:
        x_mm, y_mm = grid_to_gps(gx, gy)
        text = f"Grid: ({gx},{gy})  GPS: ({int(x_mm)} mm, {int(y_mm)} mm)"
        surf = font.render(text, True, (0, 0, 0))
        screen.blit(surf, (5, 5))

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

        # Left click: obstacle or start (with Shift)
        if pygame.mouse.get_pressed()[0]:
            x, y = pygame.mouse.get_pos()
            gx, gy = x // CELL_SIZE, y // CELL_SIZE

            if 0 <= gx < GRID_SIZE and 0 <= gy < GRID_SIZE:
                mods = pygame.key.get_mods()
                if mods & pygame.KMOD_SHIFT:  # Set start
                    start = (gx, gy)
                    compute_total_path()
                else:
                    grid[gy][gx] = 1  # obstacle
                    compute_total_path()

        # Right click: remove obstacle or set goal (with Shift)
        if pygame.mouse.get_pressed()[2]:
            x, y = pygame.mouse.get_pos()
            gx, gy = x // CELL_SIZE, y // CELL_SIZE

            if 0 <= gx < GRID_SIZE and 0 <= gy < GRID_SIZE:
                mods = pygame.key.get_mods()
                if mods & pygame.KMOD_SHIFT:  # Set goal
                    goal = (gx, gy)
                    compute_total_path()
                else:
                    grid[gy][gx] = 0  # remove obstacle
                    compute_total_path()

        # Middle click: add waypoint
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 2:
            x, y = event.pos
            gx, gy = x // CELL_SIZE, y // CELL_SIZE
            if 0 <= gx < GRID_SIZE and 0 <= gy < GRID_SIZE:
                waypoints.append((gx, gy))
                compute_total_path()

        # Example: press 'G' to print start/goal/waypoints in GPS mm to console
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_g:
                if start:
                    sx_mm, sy_mm = grid_to_gps(*start)
                    print(f"Start GPS: ({sx_mm:.1f} mm, {sy_mm:.1f} mm)")
                if goal:
                    gx_mm, gy_mm = grid_to_gps(*goal)
                    print(f"Goal GPS:  ({gx_mm:.1f} mm, {gy_mm:.1f} mm)")
                for i, wp in enumerate(waypoints):
                    wx_mm, wy_mm = grid_to_gps(*wp)
                    print(f"WP {i} GPS:  ({wx_mm:.1f} mm, {wy_mm:.1f} mm)")

pygame.quit()
