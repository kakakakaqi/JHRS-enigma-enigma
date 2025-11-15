"""
Robust Pygame exploration demo

Features:
- Correct rasterization of walls to an occupancy grid (no inverted interior/exterior)
- Thickened walls to avoid corner leaks
- Robust ray-segment intersection with tolerances
- Rays compute exact closest intersection point; samples along ray mark explored interior cells
- Flood-fill from map boundary marks exterior; interior = free & not exterior
- WASD movement, mouse to aim, stamina, collision checking
- FOV drawn as filled polygon; explored interior cells shown in green

Dependencies:
- pygame
- numpy
"""

import pygame
import math
import numpy as np
from collections import deque

# ---------------- Config ----------------
CELL_SIZE = 0.15        # meters per grid cell
SCALE = 55              # pixels per meter
FOV_DEG = 90.0
FOV = math.radians(FOV_DEG)
VIEW_RANGE = 4.5        # meters
RAY_ANG_STEP_DEG = 0.8  # degree resolution for rays

BASE_SPEED = 2.2        # m/s
STAMINA_INIT = 1.0
MIN_STAMINA = 0.12
STAMINA_LOSS_PER_M = 0.025

AVATAR_RADIUS = 0.18    # meters
WALL_THICKEN_PXD = 1    # how many dilation iterations for wall thickness (integer)

EPS = 1e-9
INTERSECTION_EPS = 1e-8

FPS = 60

# Colors
BG = (12, 12, 12)
COLOR_WALL = (230, 230, 230)
COLOR_FREE = (24, 24, 24)
COLOR_EXPLORED = (60, 150, 70)
COLOR_FOV_FILL = (200, 80, 80, 120)
COLOR_FOV_RAY = (240, 120, 120)
COLOR_AVATAR = (250, 200, 60)
COLOR_PATH = (80, 140, 240)

# ---------------- Floor plan (edit here) ----------------
WALLS = [
    ((0.0,0.0),(10.0,0.0)),
    ((10.0,0.0),(10.0,6.0)),
    ((10.0,6.0),(0.0,6.0)),
    ((0.0,6.0),(0.0,0.0)),
    # interior wall with doorway (gap)
    ((0.0,3.0),(3.9,3.0)),
    ((6.1,3.0),(10.0,3.0)),
    # small pillar
    ((4.5,1.0),(5.5,1.0)),
    ((5.5,1.0),(5.5,2.0)),
    ((5.5,2.0),(4.5,2.0)),
    ((4.5,2.0),(4.5,1.0)),
]

# ---------------- Compute bounds and grid ----------------
xs = [p[0] for seg in WALLS for p in seg]
ys = [p[1] for seg in WALLS for p in seg]
MINX, MINY = min(xs) - 0.6, min(ys) - 0.6
MAXX, MAXY = max(xs) + 0.6, max(ys) + 0.6

GRID_W = int(math.ceil((MAXX - MINX) / CELL_SIZE))
GRID_H = int(math.ceil((MAXY - MINY) / CELL_SIZE))

def grid_to_world(gx, gy):
    x = MINX + (gx + 0.5) * CELL_SIZE
    y = MINY + (gy + 0.5) * CELL_SIZE
    return x, y

def world_to_grid(pt):
    x,y = pt
    gx = int((x - MINX) / CELL_SIZE)
    gy = int((y - MINY) / CELL_SIZE)
    return gx, gy

def world_to_screen(pt):
    x,y = pt
    sx = int((x - MINX) * SCALE)
    sy = HEIGHT - int((y - MINY) * SCALE)
    return sx, sy

# ---------------- Geometry helpers ----------------
def normalize(vx, vy):
    l = math.hypot(vx, vy)
    if l < EPS:
        return 0.0, 0.0
    return vx / l, vy / l

def point_segment_distance(p, a, b):
    px,py = p; ax,ay = a; bx,by = b
    dx,dy = bx-ax, by-ay
    if abs(dx) < 1e-12 and abs(dy) < 1e-12:
        return math.hypot(px-ax, py-ay)
    t = ((px-ax)*dx + (py-ay)*dy) / (dx*dx + dy*dy)
    t = max(0.0, min(1.0, t))
    projx = ax + t*dx
    projy = ay + t*dy
    return math.hypot(px-projx, py-projy)

def ray_segment_intersection(ray_origin, ray_dir, p1, p2):
    """
    Ray (origin + u*dir, u>=0) intersect segment p1->p2.
    dir assumed normalized; u is in meters.
    Returns (ix,iy,u) or None.
    """
    x1,y1 = ray_origin
    dx,dy = ray_dir
    x3,y3 = p1
    x4,y4 = p2
    rx = x4 - x3
    ry = y4 - y3
    denom = rx * dy - ry * dx
    if abs(denom) < INTERSECTION_EPS:
        return None
    num_u = ( (x3 - x1) * ry - (y3 - y1) * rx )
    u = num_u / denom
    num_t = ( (x3 - x1) * dy - (y3 - y1) * dx )
    t = num_t / denom
    if t < -1e-7 or t > 1.0 + 1e-7:
        return None
    if u < -1e-9:
        return None
    ix = x1 + u * dx
    iy = y1 + u * dy
    return ix, iy, u

# ---------------- Rasterize walls into occupancy grid ----------------
occupancy = np.zeros((GRID_H, GRID_W), dtype=bool)
# mark cells near any wall segment
for gy in range(GRID_H):
    for gx in range(GRID_W):
        wx,wy = grid_to_world(gx,gy)
        for a,b in WALLS:
            if point_segment_distance((wx,wy), a, b) < (CELL_SIZE * 0.55):
                occupancy[gy,gx] = True
                break

# Dilate wall cells a few iterations to close corner gaps (integer dilation)
def dilate(mask, iterations=1):
    H,W = mask.shape
    out = mask.copy()
    for _ in range(iterations):
        new = out.copy()
        for y in range(H):
            for x in range(W):
                if out[y,x]:
                    for dy in (-1,0,1):
                        for dx in (-1,0,1):
                            nx,ny = x+dx, y+dy
                            if 0<=nx<W and 0<=ny<H:
                                new[ny,nx] = True
        out = new
    return out

occupancy = dilate(occupancy, iterations=WALL_THICKEN_PXD)

# free = not wall
free = ~occupancy

# ---------------- Exterior detection (flood from boundaries) ----------------
# mark exterior cells reachable from outside (grid boundary)
exterior = np.zeros_like(free, dtype=bool)
dq = deque()
# enqueue boundary free cells
for x in range(GRID_W):
    if free[0,x]:
        exterior[0,x] = True; dq.append((x,0))
    if free[GRID_H-1,x]:
        exterior[GRID_H-1,x] = True; dq.append((x,GRID_H-1))
for y in range(GRID_H):
    if free[y,0]:
        exterior[y,0] = True; dq.append((0,y))
    if free[y,GRID_W-1]:
        exterior[y,GRID_W-1] = True; dq.append((GRID_W-1,y))

dirs4 = [(1,0),(-1,0),(0,1),(0,-1)]
while dq:
    x,y = dq.popleft()
    for dx,dy in dirs4:
        nx,ny = x+dx, y+dy
        if 0<=nx<GRID_W and 0<=ny<GRID_H and free[ny,nx] and not exterior[ny,nx]:
            exterior[ny,nx] = True
            dq.append((nx,ny))

# interior mask: free and not exterior
interior = free & (~exterior)

# If interior is empty (unlikely), fallback to free as interior
if np.sum(interior) == 0:
    interior = free.copy()

# ---------------- Pygame init ----------------
pygame.init()
WIDTH = int((MAXX - MINX) * SCALE)
HEIGHT = int((MAXY - MINY) * SCALE)
screen = pygame.display.set_mode((WIDTH, HEIGHT))
clock = pygame.time.Clock()
pygame.display.set_caption("Exploration (fixed)")

font = pygame.font.SysFont(None, 20)

# ---------------- Simulation state ----------------
# pick a start cell inside interior (find one)
start = None
for gy in range(GRID_H):
    for gx in range(GRID_W):
        if interior[gy,gx]:
            start = (gx,gy); break
    if start: break
if not start:
    start = (int(GRID_W*0.2), int(GRID_H*0.5))

avatar_x, avatar_y = grid_to_world(*start)
avatar_heading = 0.0
stamina = STAMINA_INIT
trajectory = [(avatar_x, avatar_y)]
explored = np.zeros((GRID_H, GRID_W), dtype=bool)

# ensure starting visible cells get explored
def sample_and_mark_along_ray(px,py, dx,dy, max_dist):
    """Sample along ray from (px,py) along (dx,dy) normalized up to max_dist, marking interior cells."""
    steps = max(2, int(max_dist / (CELL_SIZE * 0.4)))
    for i in range(1, steps+1):
        r = (i / steps) * max_dist
        sx = px + dx * r
        sy = py + dy * r
        gx,gy = world_to_grid((sx,sy))
        if 0<=gx<GRID_W and 0<=gy<GRID_H:
            if interior[gy,gx]:
                explored[gy,gx] = True

def compute_fov_hits_and_samples(pos, heading):
    """Return list of hit points (world coords) for FOV polygon, and list of sample lists (for marking)."""
    px,py = pos
    half = FOV / 2.0
    n_rays = max(3, int(FOV_DEG / RAY_ANG_STEP_DEG))
    hits = []
    samples = []
    for i in range(n_rays+1):
        ang = heading - half + (i / n_rays) * FOV
        dx,dy = math.cos(ang), math.sin(ang)
        dxn,dyn = normalize(dx,dy)
        closest_u = None
        closest_hit = None
        # check intersections with all walls
        for a,b in WALLS:
            res = ray_segment_intersection((px,py),(dxn,dyn),a,b)
            if res:
                ix,iy,u = res
                if u < 0: continue
                if u <= VIEW_RANGE + 1e-8:
                    if closest_u is None or u < closest_u:
                        closest_u = u
                        closest_hit = (ix,iy,u)
        if closest_hit is None:
            # endpoint at view range
            ex = px + dxn * VIEW_RANGE
            ey = py + dxn * 0 + dyn * VIEW_RANGE  # careful: dyn in variable, keep consistent
            ex = px + dxn * VIEW_RANGE
            ey = py + dyn * VIEW_RANGE
            hits.append((ex,ey))
            samples.append([ (px + dxn*(VIEW_RANGE * t/4), py + dyn*(VIEW_RANGE * t/4)) for t in range(1,5) ])
        else:
            ex,ey,u = closest_hit
            hits.append((ex,ey))
            # sample along ray up to hit (more samples when hit is farther)
            steps = max(2, int(u / (CELL_SIZE * 0.4)))
            pts = [ (px + dxn*(u * t/steps), py + dyn*(u * t/steps)) for t in range(1, steps+1) ]
            samples.append(pts)
    return hits, samples

# pre-mark starting visibility
hits0, samples0 = compute_fov_hits_and_samples((avatar_x, avatar_y), avatar_heading)
for pts in samples0:
    for sx,sy in pts:
        gx,gy = world_to_grid((sx,sy))
        if 0 <= gx < GRID_W and 0 <= gy < GRID_H and interior[gy,gx]:
            explored[gy,gx] = True

# ---------------- Movement collision checker ----------------
def can_place_avatar_at(pt):
    x,y = pt
    # boundary check
    if x < MINX + AVATAR_RADIUS or x > MAXX - AVATAR_RADIUS or y < MINY + AVATAR_RADIUS or y > MAXY - AVATAR_RADIUS:
        return False
    for a,b in WALLS:
        if point_segment_distance((x,y), a, b) < (AVATAR_RADIUS - 1e-6):
            return False
    return True

# ---------------- Main loop ----------------
running = True
while running:
    dt = clock.tick(FPS) / 1000.0
    for ev in pygame.event.get():
        if ev.type == pygame.QUIT:
            running = False

    # mouse -> heading
    mx,my = pygame.mouse.get_pos()
    av_sx, av_sy = world_to_screen((avatar_x, avatar_y))
    vx = mx - av_sx
    vy = (my - av_sy) * -1
    if abs(vx) > EPS or abs(vy) > EPS:
        avatar_heading = math.atan2(vy, vx)

    # keyboard movement
    keys = pygame.key.get_pressed()
    forward = 0; strafe = 0
    if keys[pygame.K_w] or keys[pygame.K_UP]:
        forward += 1
    if keys[pygame.K_s] or keys[pygame.K_DOWN]:
        forward -= 1
    if keys[pygame.K_a] or keys[pygame.K_LEFT]:
        strafe -= 1
    if keys[pygame.K_d] or keys[pygame.K_RIGHT]:
        strafe += 1

    if forward != 0 or strafe != 0:
        # local to world
        fx,fy = math.cos(avatar_heading), math.sin(avatar_heading)
        sx_v, sy_v = -math.sin(avatar_heading), math.cos(avatar_heading)
        wx = fx*forward + sx_v*strafe
        wy = fy*forward + sy_v*strafe
        nx,ny = normalize(wx,wy)
        speed = BASE_SPEED * stamina
        proposed_x = avatar_x + nx * speed * dt
        proposed_y = avatar_y + ny * speed * dt
        if can_place_avatar_at((proposed_x, proposed_y)):
            dist = math.hypot(proposed_x - avatar_x, proposed_y - avatar_y)
            avatar_x, avatar_y = proposed_x, proposed_y
            trajectory.append((avatar_x, avatar_y))
            stamina = max(MIN_STAMINA, stamina - dist * STAMINA_LOSS_PER_M)
        else:
            # try sliding along x or y
            if can_place_avatar_at((proposed_x, avatar_y)):
                dist = abs(proposed_x - avatar_x)
                avatar_x = proposed_x
                trajectory.append((avatar_x, avatar_y))
                stamina = max(MIN_STAMINA, stamina - dist * STAMINA_LOSS_PER_M)
            elif can_place_avatar_at((avatar_x, proposed_y)):
                dist = abs(proposed_y - avatar_y)
                avatar_y = proposed_y
                trajectory.append((avatar_x, avatar_y))
                stamina = max(MIN_STAMINA, stamina - dist * STAMINA_LOSS_PER_M)

    # compute FOV hits and mark explored interior cells
    hits, samples = compute_fov_hits_and_samples((avatar_x, avatar_y), avatar_heading)
    for pts in samples:
        for sx,sy in pts:
            gx,gy = world_to_grid((sx,sy))
            if 0 <= gx < GRID_W and 0 <= gy < GRID_H and interior[gy,gx]:
                explored[gy,gx] = True

    # ---------------- Draw ----------------
    screen.fill(BG)
    # draw grid (walls & free & explored interior)
    cell_w = int(CELL_SIZE * SCALE) + 1
    for gy in range(GRID_H):
        for gx in range(GRID_W):
            wx,wy = grid_to_world(gx,gy)
            sx,sy = world_to_screen((wx,wy))
            rect = pygame.Rect(sx, sy, cell_w, cell_w)
            if occupancy[gy,gx]:
                pygame.draw.rect(screen, COLOR_WALL, rect)
            else:
                if explored[gy,gx] and interior[gy,gx]:
                    pygame.draw.rect(screen, COLOR_EXPLORED, rect)
                else:
                    pygame.draw.rect(screen, COLOR_FREE, rect)

    # crisp wall lines
    for a,b in WALLS:
        sa = world_to_screen(a); sb = world_to_screen(b)
        pygame.draw.line(screen, COLOR_WALL, sa, sb, 2)

    # draw filled FOV polygon
    av_screen = world_to_screen((avatar_x, avatar_y))
    poly = [ av_screen ] + [ world_to_screen(p) for p in hits ]
    if len(poly) >= 3:
        surf = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
        pygame.draw.polygon(surf, (COLOR_FOV_FILL[0], COLOR_FOV_FILL[1], COLOR_FOV_FILL[2], 110), poly)
        screen.blit(surf, (0,0))
    # draw rays
    for hit in hits:
        pygame.draw.line(screen, COLOR_FOV_RAY, av_screen, world_to_screen(hit), 1)

    # draw path
    if len(trajectory) >= 2:
        pts = [ world_to_screen(p) for p in trajectory ]
        pygame.draw.lines(screen, COLOR_PATH, False, pts, 2)

    # draw avatar
    pygame.draw.circle(screen, COLOR_AVATAR, av_screen, max(4, int(AVATAR_RADIUS * SCALE)))
    hx = av_screen[0] + int(math.cos(avatar_heading) * AVATAR_RADIUS * SCALE * 1.6)
    hy = av_screen[1] - int(math.sin(avatar_heading) * AVATAR_RADIUS * SCALE * 1.6)
    pygame.draw.line(screen, (40,40,40), av_screen, (hx,hy), 3)

    # HUD
    st = font.render(f"Stamina: {stamina:.3f}", True, (220,220,220))
    pos = font.render(f"Pos: {avatar_x:.2f}, {avatar_y:.2f}", True, (220,220,220))
    screen.blit(st, (8,8)); screen.blit(pos, (8,28))

    pygame.display.flip()

pygame.quit()
