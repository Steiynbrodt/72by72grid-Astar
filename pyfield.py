import pygame
import heapq
import math

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

# Robot / inflation parameters (≈30 cm diameter)
BOT_DIAMETER_MM = 300.0
BOT_RADIUS_MM = BOT_DIAMETER_MM / 2.0
SAFETY_MARGIN_MM = 50.0
# Effective "collision radius"
COLLISION_RADIUS_MM = BOT_RADIUS_MM + SAFETY_MARGIN_MM  # ≈200 mm
DEFAULT_INFLATION_RADIUS_MM = COLLISION_RADIUS_MM

pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("VEX A* Pathfinding – Field + Robot Size")

pygame.font.init()
font = pygame.font.SysFont(None, 18)
small_font = pygame.font.SysFont(None, 16)
input_font = pygame.font.SysFont(None, 18)

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (180, 180, 180)
DARK_GRAY = (80, 80, 80)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
BLUE = (0, 128, 255)
YELLOW = (255, 255, 0)
LIGHT_GRAY = (210, 210, 210)
INFO_BG = (245, 245, 245)

# Grid structure:
#   grid[y][x] = 1  -> base obstacle from field geometry or manual
#   inflated_obstacles -> cells blocked because robot has size
grid = [[0 for _ in range(GRID_SIZE)] for _ in range(GRID_SIZE)]
inflated_obstacles = set()

start = None
goal = None
waypoints = []
path = []

# GUI toggles
show_inflated_overlay = True

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

def build_inflated_obstacles():
    """
    Inflate all base obstacles by robot collision radius.
    Any cell whose center lies within COLLISION_RADIUS_MM of ANY base obstacle cell
    becomes blocked for the robot.
    """
    global inflated_obstacles
    inflated_obstacles = set()

    base_cells = [(x, y) for y in range(GRID_SIZE) for x in range(GRID_SIZE) if grid[y][x] == 1]
    if not base_cells:
        return

    r = COLLISION_RADIUS_MM
    r2 = r * r
    cell_radius = int(math.ceil(r / CELL_MM))

    for (ox, oy) in base_cells:
        ox_mm, oy_mm = grid_to_gps(ox, oy)
        gx_min = clamp(ox - cell_radius, 0, GRID_SIZE - 1)
        gx_max = clamp(ox + cell_radius, 0, GRID_SIZE - 1)
        gy_min = clamp(oy - cell_radius, 0, GRID_SIZE - 1)
        gy_max = clamp(oy + cell_radius, 0, GRID_SIZE - 1)

        for gy in range(gy_min, gy_max + 1):
            for gx in range(gx_min, gx_max + 1):
                x_mm, y_mm = grid_to_gps(gx, gy)
                dx = x_mm - ox_mm
                dy = y_mm - oy_mm
                if dx*dx + dy*dy <= r2:
                    inflated_obstacles.add((gx, gy))

def compute_total_path():
    global path
    if not start or not goal:
        path = []
        return

    build_inflated_obstacles()
    obstacles = set(inflated_obstacles)

    full_path = []
    current = start

    for wp in waypoints + [goal]:
        segment = a_star(current, wp, obstacles)
        if not segment:
            path = []
            return
        if full_path:
            segment = segment[1:]
        full_path.extend(segment)
        current = wp

    path = full_path

def clear_path_and_waypoints():
    global path, waypoints
    path = []
    waypoints = []

def reset_field():
    global grid, path, waypoints, start, goal
    grid = [[0 for _ in range(GRID_SIZE)] for _ in range(GRID_SIZE)]
    path = []
    waypoints = []
    start = None
    goal = None
    add_field_obstacles_with_small_x()
    build_inflated_obstacles()

# -----------------------
# Geometry helpers (mm-based)
# -----------------------
def add_edge_margin(margin_mm: float):
    """Replicates field.addEdgeMargin(margin_mm)."""
    for gy in range(GRID_SIZE):
        for gx in range(GRID_SIZE):
            x_mm, y_mm = grid_to_gps(gx, gy)
            if (abs(x_mm) > FIELD_HALF_MM - margin_mm) or (abs(y_mm) > FIELD_HALF_MM - margin_mm):
                grid[gy][gx] = 1

def add_rect_mm(cx_mm: float, cy_mm: float, w_mm: float, h_mm: float):
    """
    Replicates field.addRectMm(centerX, centerY, width, height).
    Axis-aligned rectangle, centered at (cx_mm, cy_mm).
    """
    x_min = cx_mm - w_mm / 2.0
    x_max = cx_mm + w_mm / 2.0
    y_min = cy_mm - h_mm / 2.0
    y_max = cy_mm + h_mm / 2.0

    for gy in range(GRID_SIZE):
        for gx in range(GRID_SIZE):
            x_mm, y_mm = grid_to_gps(gx, gy)
            if x_min <= x_mm <= x_max and y_min <= y_mm <= y_max:
                grid[gy][gx] = 1

def add_disk_mm(x_mm: float, y_mm: float, radius_mm: float):
    """
    Replicates field.addDiskMm(x, y, R) WITHOUT extra inflation.
    """
    r2 = radius_mm * radius_mm
    center_gx, center_gy = gps_to_grid(x_mm, y_mm)
    radius_cells = int(math.ceil(radius_mm / CELL_MM))

    min_gx = clamp(center_gx - radius_cells, 0, GRID_SIZE - 1)
    max_gx = clamp(center_gx + radius_cells, 0, GRID_SIZE - 1)
    min_gy = clamp(center_gy - radius_cells, 0, GRID_SIZE - 1)
    max_gy = clamp(center_gy + radius_cells, 0, GRID_SIZE - 1)

    for gy in range(min_gy, max_gy + 1):
        for gx in range(min_gx, max_gx + 1):
            cell_x_mm, cell_y_mm = grid_to_gps(gx, gy)
            dx = cell_x_mm - x_mm
            dy = cell_y_mm - y_mm
            if dx*dx + dy*dy <= r2:
                grid[gy][gx] = 1

# Interactive circular obstacles (user-defined zones)
def inflate_obstacle_gps(x_mm, y_mm, radius_mm):
    """
    For user input: add a circular forbidden zone in base grid,
    which is then inflated by robot size for pathfinding.
    """
    add_disk_mm(x_mm, y_mm, radius_mm)
    compute_total_path()

def parse_and_inflate_from_text(text):
    """
    Expected format in input field:
        x_mm y_mm radius_mm
    or:
        x_mm y_mm        (radius defaults to COLLISION_RADIUS_MM)
    Example:
        0 0 200
        0 0
    """
    text = text.strip()
    if not text:
        return

    parts = text.replace(",", " ").split()
    if len(parts) == 2:
        parts.append(str(DEFAULT_INFLATION_RADIUS_MM))
    elif len(parts) != 3:
        print("Input must be: x_mm y_mm [radius_mm]")
        return

    try:
        x_mm = float(parts[0])
        y_mm = float(parts[1])
        r_mm = float(parts[2])
    except ValueError:
        print("Could not parse numbers from input.")
        return

    inflate_obstacle_gps(x_mm, y_mm, r_mm)
    print(f"Inflated obstacle at ({x_mm:.1f} mm, {y_mm:.1f} mm) with radius {r_mm:.1f} mm")

# -----------------------
# fieldparameters.hpp: addFieldObstaclesWithSmallX
# -----------------------
def add_field_obstacles_with_small_x():
    """
    Python equivalent of your addFieldObstaclesWithSmallX:

    field.addEdgeMargin(100.0);
    field.addRectMm(0.0,  1200.0, 1000.0, 125.0);
    field.addRectMm(0.0, -1200.0, 1000.0, 125.0);
    constexpr double R = 250.0;
    centers = { ... };
    field.addDiskMm(c[0], c[1], R);
    """
    # ---- Walls / edge margin ----
    add_edge_margin(100.0)

    # ---- Long-goal bars ----
    add_rect_mm(0.0,  1200.0, 1000.0, 125.0)  # top bar
    add_rect_mm(0.0, -1200.0, 1000.0, 125.0)  # bottom bar

    # ---- Small diagonal X (±300mm) ----
    R = 250.0
    centers = [
        (-150.0, -150.0),
        (-150.0,  150.0),
        (-200.0, -200.0),
        (-200.0,  200.0),
        (   0.0,    0.0),
        ( 150.0,  150.0),
        ( 150.0, -150.0),
        ( 200.0,  200.0),
        ( 200.0, -200.0),
    ]
    for (cx, cy) in centers:
        add_disk_mm(cx, cy, R)

# -----------------------
# Drawing
# -----------------------
INPUT_BOX_HEIGHT = 24
input_box_rect = pygame.Rect(5, HEIGHT - INPUT_BOX_HEIGHT - 2, 420, INPUT_BOX_HEIGHT)
input_text = ""
input_active = True

def draw_info_panel():
    panel_rect = pygame.Rect(5, 5, 420, 90)
    pygame.draw.rect(screen, INFO_BG, panel_rect)
    pygame.draw.rect(screen, DARK_GRAY, panel_rect, 1)

    lines = []

    # Start / goal info
    if start:
        sx, sy = grid_to_gps(*start)
        lines.append(f"Start: grid {start}  GPS ({int(sx)} mm, {int(sy)} mm)")
    else:
        lines.append("Start: not set")

    if goal:
        gx, gy = grid_to_gps(*goal)
        lines.append(f"Goal : grid {goal}  GPS ({int(gx)} mm, {int(gy)} mm)")
    else:
        lines.append("Goal : not set")

    lines.append(f"Robot collision radius: {int(COLLISION_RADIUS_MM)} mm")

    # Controls
    lines.append("LMB: add obstacle   RMB: remove")
    lines.append("Shift+LMB: set start   Shift+RMB: set goal")
    lines.append("MMB: add waypoint   C: clear path/WPs   R: reset field")
    lines.append("B: toggle inflated overlay   G: print GPS   Input: 'x y [r]' + Enter")

    y = panel_rect.y + 4
    for text in lines:
        surf = small_font.render(text, True, DARK_GRAY)
        screen.blit(surf, (panel_rect.x + 4, y))
        y += 12

def draw():
    screen.fill(WHITE)

    # Draw base obstacles (field geometry) and inflated-only cells
    for y in range(GRID_SIZE):
        for x in range(GRID_SIZE):
            if grid[y][x] == 1:
                # Base obstacle
                pygame.draw.rect(screen, BLACK, (x*CELL_SIZE, y*CELL_SIZE, CELL_SIZE, CELL_SIZE))
            elif show_inflated_overlay and (x, y) in inflated_obstacles:
                # Blocking only because of robot radius
                pygame.draw.rect(screen, LIGHT_GRAY, (x*CELL_SIZE, y*CELL_SIZE, CELL_SIZE, CELL_SIZE))

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

    # Info panel
    draw_info_panel()

    # Input label
    label_surf = input_font.render(
        f"Inflate zone (x_mm y_mm [r_mm], default r≈{int(DEFAULT_INFLATION_RADIUS_MM)}):",
        True, DARK_GRAY
    )
    screen.blit(label_surf, (input_box_rect.x, input_box_rect.y - 18))

    # Input box background
    pygame.draw.rect(screen, WHITE, input_box_rect)
    pygame.draw.rect(screen, DARK_GRAY, input_box_rect, 1)

    # Input text
    txt_surf = input_font.render(input_text, True, BLACK)
    screen.blit(txt_surf, (input_box_rect.x + 4, input_box_rect.y + 4))

    pygame.display.flip()

# -----------------------
# Main
# -----------------------
add_field_obstacles_with_small_x()
build_inflated_obstacles()  # initial inflation

running = True
clock = pygame.time.Clock()

while running:
    clock.tick(60)
    draw()

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        # Mouse clicks (only on click, not held)
        if event.type == pygame.MOUSEBUTTONDOWN:
            x, y = event.pos

            # ignore clicks in input box
            if input_box_rect.collidepoint(x, y):
                continue

            gx, gy = x // CELL_SIZE, y // CELL_SIZE
            if 0 <= gx < GRID_SIZE and 0 <= gy < GRID_SIZE:
                mods = pygame.key.get_mods()

                # Left click: obstacle or start (with Shift)
                if event.button == 1:
                    if mods & pygame.KMOD_SHIFT:
                        start = (gx, gy)
                    else:
                        grid[gy][gx] = 1
                    compute_total_path()

                # Right click: remove obstacle or set goal (with Shift)
                elif event.button == 3:
                    if mods & pygame.KMOD_SHIFT:
                        goal = (gx, gy)
                    else:
                        grid[gy][gx] = 0
                    compute_total_path()

                # Middle click: add waypoint
                elif event.button == 2:
                    waypoints.append((gx, gy))
                    compute_total_path()

        # Keyboard handling (input field + controls)
        if event.type == pygame.KEYDOWN:
            # debug print GPS of start/goal/WPs
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

            # Clear path + waypoints
            if event.key == pygame.K_c:
                clear_path_and_waypoints()
                build_inflated_obstacles()

            # Reset full field
            if event.key == pygame.K_r:
                reset_field()

            # Toggle inflated overlay
            if event.key == pygame.K_b:
                show_inflated_overlay = not show_inflated_overlay

            # Text input for obstacle inflation
            if input_active:
                if event.key == pygame.K_RETURN:
                    parse_and_inflate_from_text(input_text)
                    input_text = ""
                elif event.key == pygame.K_BACKSPACE:
                    input_text = input_text[:-1]
                else:
                    if event.unicode and event.unicode.isprintable():
                        input_text += event.unicode

pygame.quit()
