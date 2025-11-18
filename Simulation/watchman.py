def compute_watchman_route(
    room,
    fov_angle: float,
    vis_radius: float,
) -> list[tuple[float, float]]:
    # ------------------------------------------
    # Extract geometry from Room
    # ------------------------------------------
    rx, ry, rl, rw = room.rect.dump_xy()
    OUTER = [
        (rx, ry),
        (rx + rl, ry),
        (rx + rl, ry + rw),
        (rx, ry + rw),
    ]
    OBSTACLES = []  # none for now

    # ------------------------------------------
    # Parameters — kept identical to your script
    # ------------------------------------------
    FOV_ANGLE_DEG = fov_angle
    VIS_RADIUS = vis_radius
    ANGULAR_SAMPLES = 72
    NUM_RANDOM_CANDIDATES = 200
    TSP_RANDOM_RESTARTS = 5
    EPS_ANGLE = 1e-7

    from shapely.geometry import Point, Polygon, LineString
    from shapely.ops import unary_union
    import numpy as np
    import networkx as nx
    import math
    import random

    # ------------------------------------------
    # Utility functions
    # ------------------------------------------
    def normalize_angle(a):
        return (a + math.pi) % (2 * math.pi) - math.pi

    def clean_geom(geom):
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

    # ------------------------------------------
    # Build polygons
    # ------------------------------------------
    outer_poly = Polygon(OUTER)
    obstacle_polys = []
    obstacles_union = None
    free_space = outer_poly

    # ------------------------------------------
    # Candidate viewpoints
    # ------------------------------------------
    candidates = []
    for x, y in OUTER:
        candidates.append((x, y))

    # random interior candidates
    minx, miny, maxx, maxy = free_space.bounds
    count = 0
    while count < NUM_RANDOM_CANDIDATES:
        rx = random.uniform(minx, maxx)
        ry = random.uniform(miny, maxy)
        p = Point(rx, ry)
        if p.within(free_space):
            candidates.append((rx, ry))
            count += 1

    candidates = list({(round(x, 6), round(y, 6)) for x, y in candidates})

    # ------------------------------------------
    # Ray targets = outer polygon vertices
    # ------------------------------------------
    ray_targets = [tuple(v) for v in OUTER]

    # ------------------------------------------
    # Visibility polygon per viewpoint
    # ------------------------------------------
    def compute_visibility_polygon(
        viewpoint, heading_angle=None, fov_deg=FOV_ANGLE_DEG, radius=VIS_RADIUS
    ):
        x0, y0 = viewpoint
        center = Point(x0, y0)

        angles = []
        for tx, ty in ray_targets:
            angles.append(math.atan2(ty - y0, tx - x0))
        for k in range(ANGULAR_SAMPLES):
            angles.append((2 * math.pi * k) / ANGULAR_SAMPLES)

        if heading_angle is not None:
            half = math.radians(fov_deg) / 2.0
            angles = [
                a for a in angles if abs(normalize_angle(a - heading_angle)) <= half
            ]

        # jitter
        angles = sorted(
            set([round(a + random.uniform(-EPS_ANGLE, EPS_ANGLE), 9) for a in angles])
        )

        ray_points = []
        for a in angles:
            dx = math.cos(a)
            dy = math.sin(a)
            far_pt = (x0 + dx * radius * 1.0001, y0 + dy * radius * 1.0001)
            ray = LineString([(x0, y0), far_pt])
            try:
                inter = ray.intersection(free_space)
            except Exception:
                continue
            if inter.is_empty:
                continue

            if isinstance(inter, Point):
                ray_points.append((inter.x, inter.y))
                continue

            coords = []
            try:
                coords = list(inter.coords)
            except:
                for g in getattr(inter, "geoms", []):
                    try:
                        coords += list(g.coords)
                    except:
                        pass
            if not coords:
                continue

            best = max(coords, key=lambda c: (c[0] - x0) ** 2 + (c[1] - y0) ** 2)
            ray_points.append(best)

        if not ray_points:
            return Polygon()

        ray_points = sorted(ray_points, key=lambda p: math.atan2(p[1] - y0, p[0] - x0))
        try:
            raw_poly = Polygon([(x0, y0)] + ray_points)
        except:
            return Polygon()

        raw_poly = clean_geom(raw_poly)

        # radius clip
        try:
            radius_disk = clean_geom(Point(x0, y0).buffer(radius, resolution=32))
            vis_poly = raw_poly.intersection(radius_disk)
            vis_poly = clean_geom(vis_poly)
            vis_poly = vis_poly.intersection(free_space)
            vis_poly = clean_geom(vis_poly)
        except:
            return Polygon()

        if vis_poly.is_empty:
            return Polygon()
        return vis_poly

    # ------------------------------------------
    # Union over headings (FOV < 360)
    # ------------------------------------------
    def viewpoint_full_visibility(viewpoint):
        if FOV_ANGLE_DEG >= 360:
            return compute_visibility_polygon(viewpoint)

        x0, y0 = viewpoint
        headings = [math.atan2(ty - y0, tx - x0) for tx, ty in ray_targets]
        for k in range(max(3, ANGULAR_SAMPLES // 6)):
            headings.append(2 * math.pi * k / max(3, ANGULAR_SAMPLES // 6))

        vis_union = None
        for h in set(headings):
            vp = compute_visibility_polygon(viewpoint, heading_angle=h)
            if vp.is_empty:
                continue
            vis_union = vp if vis_union is None else clean_geom(vis_union.union(vp))
        return vis_union if vis_union else Polygon()

    # ------------------------------------------
    # Compute visibility polygons
    # ------------------------------------------
    visibility_polys = {}
    for c in candidates:
        p = Point(c)
        if not p.within(free_space):
            continue
        vis = viewpoint_full_visibility(c)
        if not vis.is_empty:
            visibility_polys[c] = vis

    # ------------------------------------------
    # Sample area points (universe)
    # ------------------------------------------
    NUM_AREA_SAMPLES = 1500
    area_samples = []
    bx0, by0, bx1, by1 = free_space.bounds
    attempts = 0
    while len(area_samples) < NUM_AREA_SAMPLES and attempts < NUM_AREA_SAMPLES * 10:
        rx = random.uniform(bx0, bx1)
        ry = random.uniform(by0, by1)
        pt = Point(rx, ry)
        if pt.within(free_space):
            area_samples.append((rx, ry))
        attempts += 1

    # ------------------------------------------
    # Compute coverage sets
    # ------------------------------------------
    coverage_sets = {}
    for c, poly in visibility_polys.items():
        covered = []
        for i, (sx, sy) in enumerate(area_samples):
            if poly.covers(Point(sx, sy)):
                covered.append(i)
        coverage_sets[c] = set(covered)

    # ------------------------------------------
    # Greedy set cover
    # ------------------------------------------
    remaining = set(range(len(area_samples)))
    selected = []

    while remaining:
        best, best_cov = None, set()
        for c, s in coverage_sets.items():
            cov = s & remaining
            if len(cov) > len(best_cov):
                best, best_cov = c, cov
        if best is None:
            break
        selected.append(best)
        remaining -= best_cov

    if not selected:
        return []

    # ------------------------------------------
    # Build visibility graph: nodes = corners + selected
    # ------------------------------------------
    nodes = OUTER[:] + selected[:]
    G = nx.Graph()
    for n in nodes:
        G.add_node(n)

    for i, a in enumerate(nodes):
        for b in nodes[i + 1 :]:
            seg = LineString([a, b])
            if seg.within(free_space) or seg.buffer(1e-9).within(free_space):
                G.add_edge(a, b, weight=Point(a).distance(Point(b)))

    # ------------------------------------------
    # Pairwise shortest paths
    # ------------------------------------------
    pairwise_dist = {}
    for i, a in enumerate(selected):
        for b in selected[i + 1 :]:
            try:
                d = nx.shortest_path_length(G, a, b, weight="weight")
            except:
                d = Point(a).distance(Point(b)) * 10.0
            pairwise_dist[(a, b)] = d
            pairwise_dist[(b, a)] = d

    # ------------------------------------------
    # TSP solver
    # ------------------------------------------
    def tour_length(tour):
        L = 0
        for i in range(len(tour)):
            a = tour[i]
            b = tour[(i + 1) % len(tour)]
            L += pairwise_dist.get((a, b), Point(a).distance(Point(b)) * 10)
        return L

    def nearest_neighbor(start):
        rem = set(selected)
        tour = [start]
        rem.remove(start)
        cur = start
        while rem:
            nxt = min(
                rem,
                key=lambda x: pairwise_dist.get(
                    (cur, x), Point(cur).distance(Point(x)) * 10
                ),
            )
            tour.append(nxt)
            rem.remove(nxt)
            cur = nxt
        return tour

    best_tour = None
    best_len = float("inf")

    for s in selected[: min(len(selected), TSP_RANDOM_RESTARTS)]:
        t = nearest_neighbor(s)
        # no 2-opt needed in a small room, but could be added
        L = tour_length(t)
        if L < best_len:
            best_len = L
            best_tour = t

    return best_tour if best_tour else []
