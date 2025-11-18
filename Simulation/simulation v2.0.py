"""
Visibility-based Watchman Coverage (Python)

- Produces a near-shortest route that lets an agent with limited FOV and radius
  see every point in a known polygonal environment with polygonal obstacles.
- Approach:
  1. Define environment polygon (outer boundary) and obstacles (list of polygons).
  2. Generate candidate viewpoints (corners + sampled points).
  3. For each candidate, compute its *visibility polygon* limited by FOV and radius
     via casting rays to environment & obstacle vertices + uniform sampling.
  4. Greedy set-cover to select viewpoints that cover the free-space area.
  5. Build a shortest-path graph between selected viewpoints (visibility graph),
     compute pairwise shortest-path distances (Dijkstra).
  6. Solve TSP approximately (nearest-neighbour + 2-opt) on the metric distances.
  7. Output and plot route.

Editable variables (near top):
- FOV_ANGLE_DEG (float): field-of-view angle in degrees
- VIS_RADIUS (float): maximum visibility distance (meters)
- MAP_TYPE (str): 'polygon' or 'grid' (only 'polygon' implemented)
- OUTER (list of (x,y)): vertices of outer polygon boundary
- OBSTACLES (list of lists of (x,y)): each obstacle polygon's vertices

Requires: shapely, numpy, matplotlib, networkx
Install: pip install shapely numpy matplotlib networkx

Notes:
- This implementation trades guaranteed optimality for practicality.
- For large/complex maps you can increase angular sampling or candidate density.

"""

from shapely.geometry import Point, Polygon, LineString
from shapely.ops import unary_union
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import math
import random

# --------------------------- Editable variables ---------------------------
FOV_ANGLE_DEG = 90.0      # field of view angle in degrees
VIS_RADIUS = 5         # visibility radius
MAP_TYPE = 'polygon'      # 'polygon' implemented in this script

# Outer boundary (counter-clockwise) and obstacles (list of CCW polygons)
OUTER = [ (0,0), (24,0), (24,15), (0,15) ]
OBSTACLES = [
    [(3,2), (5,2), (5,4), (3,4)],
    [(7,5), (9,5), (9,7), (7,7)]
]

# Candidate sampling parameters
ANGULAR_SAMPLES = 72       # per viewpoint, number of angular rays to cast (for smoothing)
ADDITIONAL_CANDIDATES_PER_EDGE = 0  # add samples along edges

# TSP params
TSP_RANDOM_RESTARTS = 5

# -------------------------------------------------------------------------

# Utility functions

def angle_between(a, b):
    ax, ay = a; bx, by = b
    return math.atan2(by-ay, bx-ax)


def normalize_angle(a):
    # [-pi, pi)
    return (a + math.pi) % (2*math.pi) - math.pi


# Build shapely polygons
outer_poly = Polygon(OUTER)
obstacle_polys = [Polygon(o) for o in OBSTACLES]
obstacles_union = unary_union(obstacle_polys) if obstacle_polys else None
free_space = outer_poly.difference(obstacles_union) if obstacle_polys else outer_poly

# Candidate viewpoints: outer + obstacle vertices + optional edge samples
candidates = []
# Add vertices of obstacles and outer polygon (inward-facing positions)
for x,y in OUTER:
    candidates.append((x,y))
for obs in OBSTACLES:
    for x,y in obs:
        candidates.append((x,y))

# Optionally add some interior random samples for denser coverage
# You can increase this to get fewer but better viewpoints (tuned by user)
NUM_RANDOM_CANDIDATES = 200
minx, miny, maxx, maxy = free_space.bounds
count = 0
while count < NUM_RANDOM_CANDIDATES:
    rx = random.uniform(minx, maxx)
    ry = random.uniform(miny, maxy)
    p = Point(rx, ry)
    if p.within(free_space):
        candidates.append((rx, ry))
        count += 1

# Deduplicate candidates
candidates = list({(round(x,6), round(y,6)) for x,y in candidates})

# Precompute all vertices we will cast rays toward (outer + obstacles vertices)
ray_targets = [tuple(v) for v in OUTER]
for obs in OBSTACLES:
    ray_targets += [tuple(v) for v in obs]

# Visibility polygon computation for a single viewpoint with FOV and radius

# Visibility polygon computation for a single viewpoint with FOV and radius

def clean_geom(geom):
    """Repair invalid geometries using the buffer(0) trick and return a valid geometry."""
    try:
        if geom is None or geom.is_empty:
            return geom
        if geom.is_valid:
            return geom
        repaired = geom.buffer(0)
        if repaired.is_valid:
            return repaired
        return repaired
    except Exception:
        return geom

EPS_ANGLE = 1e-7  # small jitter to break exact-angle degeneracies

def compute_visibility_polygon(viewpoint, heading_angle=None, fov_deg=FOV_ANGLE_DEG, radius=VIS_RADIUS):
    """
    Robust approximation of visibility polygon limited by FOV and radius.
    Uses small angular jitter, cleans geometries, and clips by radius and free_space
    with repair fallbacks to avoid GEOS TopologyExceptions.
    """
    x0, y0 = viewpoint
    center = Point(x0, y0)

    # build angle list (toward vertices + uniform samples)
    angles = []
    for tx, ty in ray_targets:
        angles.append(math.atan2(ty - y0, tx - x0))
    for k in range(ANGULAR_SAMPLES):
        angles.append((2 * math.pi * k) / ANGULAR_SAMPLES)

    if heading_angle is not None:
        half = math.radians(fov_deg) / 2.0
        angles = [a for a in angles if abs(normalize_angle(a - heading_angle)) <= half]

    # add tiny random jitter to avoid exact overlaps
    angles = sorted(set([round(a + random.uniform(-EPS_ANGLE, EPS_ANGLE), 9) for a in angles]))

    ray_points = []
    for a in angles:
        dx = math.cos(a)
        dy = math.sin(a)
        far_pt = (x0 + dx * radius * 1.0001, y0 + dy * radius * 1.0001)
        ray = LineString([(x0, y0), far_pt])
        try:
            inter = ray.intersection(free_space)
        except Exception:
            # If GEOS fails, skip this ray
            continue
        if inter.is_empty:
            continue
        # choose intersection point farthest from viewpoint (on the ray)
        if isinstance(inter, Point):
            px, py = inter.x, inter.y
            ray_points.append((px, py))
            continue
        # If geometry, extract coordinates safely
        coords = []
        try:
            coords = list(inter.coords)
        except Exception:
            try:
                for g in getattr(inter, 'geoms', []):
                    try:
                        coords += list(g.coords)
                    except Exception:
                        pass
            except Exception:
                coords = []
        if not coords:
            continue
        best = max(coords, key=lambda c: (c[0] - x0) ** 2 + (c[1] - y0) ** 2)
        ray_points.append(best)

    if not ray_points:
        return Polygon()

    # angular order and build polygon
    ray_points = sorted(ray_points, key=lambda p: math.atan2(p[1] - y0, p[0] - x0))
    try:
        raw_poly = Polygon([(x0, y0)] + ray_points)
    except Exception:
        # fallback: create polygon from convex hull of points
        try:
            raw_poly = Polygon(Point(x0, y0).buffer(1e-6).union(LineString(ray_points)).convex_hull)
        except Exception:
            return Polygon()

    raw_poly = clean_geom(raw_poly)

    # Clip by radius (disk) and free_space; do radius clip first to reduce complexity
    try:
        radius_disk = clean_geom(Point(x0, y0).buffer(radius, resolution=32))
        vis_poly = raw_poly.intersection(radius_disk)
        vis_poly = clean_geom(vis_poly)
        vis_poly = vis_poly.intersection(free_space)
        vis_poly = clean_geom(vis_poly)
    except Exception:
        # robust fallback: approximate by taking points on the ray within radius
        clipped_pts = []
        for px, py in ray_points:
            if (px - x0) ** 2 + (py - y0) ** 2 <= (radius + 1e-6) ** 2:
                clipped_pts.append((px, py))
            else:
                # scale back to radius
                ang = math.atan2(py - y0, px - x0)
                clipped_pts.append((x0 + math.cos(ang) * radius, y0 + math.sin(ang) * radius))
        try:
            vis_poly = Polygon([(x0, y0)] + clipped_pts)
            vis_poly = clean_geom(vis_poly)
            vis_poly = vis_poly.intersection(free_space)
            vis_poly = clean_geom(vis_poly)
        except Exception:
            return Polygon()

    if vis_poly.is_empty:
        return Polygon()
    return vis_poly


# For omni-directional with limited FOV (agent can rotate), compute union over sampled headings
def viewpoint_full_visibility(viewpoint, fov_deg=FOV_ANGLE_DEG, radius=VIS_RADIUS):
    # If FOV >= 360 treat as full
    if fov_deg >= 360:
        return compute_visibility_polygon(viewpoint, heading_angle=None, fov_deg=fov_deg, radius=radius)
    headings = []
    x0, y0 = viewpoint
    for tx, ty in ray_targets:
        headings.append(math.atan2(ty - y0, tx - x0))
    # add uniform headings
    for k in range(max(3, ANGULAR_SAMPLES // 6)):
        headings.append(2 * math.pi * k / max(3, ANGULAR_SAMPLES // 6))
    vis_union = None
    for h in set(headings):
        try:
            vp = compute_visibility_polygon(viewpoint, heading_angle=h, fov_deg=fov_deg, radius=radius)
        except Exception:
            continue
        if vp.is_empty:
            continue
        vis_union = vp if vis_union is None else clean_geom(vis_union.union(vp))
    if vis_union is None:
        return Polygon()
    return clean_geom(vis_union)

# Compute visibility polygons for all candidates
visibility_polys = {}
for c in candidates:
    p = Point(c)
    if not p.within(free_space):
        continue
    vis = viewpoint_full_visibility(c)
    if vis.is_empty:
        continue
    visibility_polys[c] = vis

# Build area universe to cover: we will approximate by sampling points inside free_space
NUM_AREA_SAMPLES = 1500
area_samples = []
bx0,by0,bx1,by1 = free_space.bounds
attempts = 0
while len(area_samples) < NUM_AREA_SAMPLES and attempts < NUM_AREA_SAMPLES*10:
    rx = random.uniform(bx0, bx1)
    ry = random.uniform(by0, by1)
    pt = Point(rx, ry)
    if pt.within(free_space):
        area_samples.append((rx,ry))
    attempts += 1

# For each candidate, compute which sample points it sees
coverage_sets = {}
for c,poly in visibility_polys.items():
    covered = []
    for i, (sx,sy) in enumerate(area_samples):
        if poly.covers(Point(sx,sy)):
            covered.append(i)
    coverage_sets[c] = set(covered)

# Greedy set cover to pick viewpoints
remaining = set(range(len(area_samples)))
selected = []
while remaining:
    # choose candidate covering largest number of remaining
    best = None
    best_covers = set()
    for c,s in coverage_sets.items():
        cov = s & remaining
        if len(cov) > len(best_covers):
            best = c
            best_covers = cov
    if best is None:
        print("Warning: some areas are not coverable by any candidate. Stopping.")
        break
    selected.append(best)
    remaining -= best_covers

print(f"Selected {len(selected)} viewpoints from {len(visibility_polys)} candidates")

# Build visibility graph nodes: all polygon vertices + selected viewpoints
nodes = []
node_points = {}
# add outer vertices
for v in OUTER:
    nodes.append(v)
    node_points[v] = Point(v)
# add obstacle vertices
for obs in OBSTACLES:
    for v in obs:
        nodes.append(v)
        node_points[v] = Point(v)
# add selected viewpoints
for v in selected:
    nodes.append(v)
    node_points[v] = Point(v)

# Create visibility graph edges: connect node A and B if segment AB lies inside free_space
G = nx.Graph()
for n in nodes:
    G.add_node(n)

for i,a in enumerate(nodes):
    for b in nodes[i+1:]:
        seg = LineString([a,b])
        # check if segment is within free_space (with small buffer tolerance)
        if seg.relate_pattern(free_space, 'T********') or seg.within(free_space) or seg.buffer(1e-9).within(free_space):
            dist = Point(a).distance(Point(b))
            G.add_edge(a,b,weight=dist)
        else:
            # if straight-line not valid, try to see if path via polygon vertices is possible later when computing shortest paths
            pass

# For pairs of selected viewpoints, compute shortest-path distance on G (if disconnected, use large)
pairwise_dist = {}
for i,a in enumerate(selected):
    for b in selected[i+1:]:
        try:
            d = nx.shortest_path_length(G, source=a, target=b, weight='weight')
            pairwise_dist[(a,b)] = d
            pairwise_dist[(b,a)] = d
        except nx.NetworkXNoPath:
            # fallback: euclidean distance but add high penalty
            d = Point(a).distance(Point(b)) * 10.0
            pairwise_dist[(a,b)] = d
            pairwise_dist[(b,a)] = d

# Solve TSP approximately using nearest neighbour + 2-opt

def tour_length(tour):
    L = 0.0
    for i in range(len(tour)):
        a = tour[i]
        b = tour[(i+1)%len(tour)]
        L += pairwise_dist.get((a,b), Point(a).distance(Point(b))*10.0)
    return L


def nearest_neighbor_tour(start):
    remaining_nodes = set(selected)
    tour = [start]
    remaining_nodes.remove(start)
    cur = start
    while remaining_nodes:
        nxt = min(remaining_nodes, key=lambda x: pairwise_dist.get((cur,x), Point(cur).distance(Point(x))*10.0))
        tour.append(nxt)
        remaining_nodes.remove(nxt)
        cur = nxt
    return tour


def two_opt(tour):
    best = tour[:]
    improved = True
    best_len = tour_length(best)
    while improved:
        improved = False
        for i in range(1, len(tour)-1):
            for j in range(i+1, len(tour)):
                if j - i == 1:
                    continue
                new = best[:i] + best[i:j][::-1] + best[j:]
                nl = tour_length(new)
                if nl + 1e-9 < best_len:
                    best = new
                    best_len = nl
                    improved = True
        tour = best
    return best

# Try multiple restarts
best_tour = None
best_len = float('inf')
if not selected:
    print("No viewpoints selected — nothing to plan.")
else:
    for s in selected[:min(len(selected), TSP_RANDOM_RESTARTS)]:
        t = nearest_neighbor_tour(s)
        t = two_opt(t)
        l = tour_length(t)
        if l < best_len:
            best_len = l
            best_tour = t

print(f"TSP tour length (approx): {best_len:.3f}")

# Visualize
fig, ax = plt.subplots(figsize=(10,7))
# plot free space
xs, ys = outer_poly.exterior.xy
ax.fill(xs, ys, alpha=0.2, edgecolor='black')
for obs in obstacle_polys:
    ox,oy = obs.exterior.xy
    ax.fill(ox, oy, color='gray', alpha=0.8)

# plot sample points
asx = [p[0] for p in area_samples]
asy = [p[1] for p in area_samples]
ax.scatter(asx, asy, s=4, alpha=0.3)

# plot selected viewpoints
for v in selected:
    ax.plot(v[0], v[1], 'ro')

# plot visibility polygons (light)
for v in selected:
    poly = visibility_polys[v]
    try:
        px,py = poly.exterior.xy
        ax.fill(px,py, alpha=0.08)
    except Exception:
        pass

# plot tour
if best_tour:
    tour_coords = best_tour + [best_tour[0]]
    tx = [p[0] for p in tour_coords]
    ty = [p[1] for p in tour_coords]
    ax.plot(tx, ty, '-r', linewidth=2, label='tour')

ax.set_aspect('equal')
ax.set_title('Approximate Watchman Route (visibility-based coverage)')
plt.legend()
plt.show()

# Print tour order
if best_tour:
    print('Tour order:')
    for i,v in enumerate(best_tour):
        print(f"  {i+1}: {v}")

# Save route as list of coordinates for downstream navigation
route = best_tour if best_tour else []

# End of script
