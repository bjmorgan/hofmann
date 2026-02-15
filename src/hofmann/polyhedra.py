"""Coordination polyhedra computation from bonds."""

from collections import defaultdict, deque
from fnmatch import fnmatch

import numpy as np
from scipy.spatial import ConvexHull, Delaunay, QhullError

from hofmann.model import Bond, Polyhedron, PolyhedronSpec


def compute_polyhedra(
    species: list[str],
    coords: np.ndarray,
    bonds: list[Bond],
    polyhedra_specs: list[PolyhedronSpec],
    *,
    n_centre_atoms: int | None = None,
) -> list[Polyhedron]:
    """Compute coordination polyhedra from bonds and declarative specs.

    For each atom whose species matches a :class:`PolyhedronSpec`
    centre pattern, the bonded neighbours are collected and their
    convex hull is computed to produce triangulated faces.

    Specs are applied in order; the first matching spec wins for
    each atom (consistent with :func:`~hofmann.bonds.compute_bonds`).

    Args:
        species: Species labels, one per atom.
        coords: Coordinate array of shape ``(n_atoms, 3)``.
        bonds: Previously computed bonds.
        polyhedra_specs: Declarative polyhedron rules.
        n_centre_atoms: If set, only atoms with index below this
            value are considered as polyhedron centres.  Image atoms
            (from PBC expansion) can still be vertices but not
            centres.  ``None`` means all atoms are candidates.

    Returns:
        List of :class:`Polyhedron` objects.
    """
    if not polyhedra_specs or not bonds:
        return []

    coords = np.asarray(coords, dtype=float)
    max_centre = n_centre_atoms if n_centre_atoms is not None else len(species)

    # Build adjacency from bonds.
    adjacency: dict[int, set[int]] = defaultdict(set)
    for bond in bonds:
        adjacency[bond.index_a].add(bond.index_b)
        adjacency[bond.index_b].add(bond.index_a)

    # Track which atoms have already been claimed by a spec.
    claimed: set[int] = set()
    result: list[Polyhedron] = []

    for spec in polyhedra_specs:
        for i, sp in enumerate(species[:max_centre]):
            if i in claimed:
                continue
            if not fnmatch(sp, spec.centre):
                continue

            neighbours = sorted(adjacency.get(i, set()))
            if len(neighbours) < (spec.min_vertices or 3):
                continue

            claimed.add(i)
            neighbour_coords = coords[neighbours]
            faces = _triangulate(neighbour_coords)
            if faces is None:
                continue

            result.append(Polyhedron(
                centre_index=i,
                neighbour_indices=tuple(neighbours),
                faces=faces,
                spec=spec,
            ))

    return result


def _triangulate(coords: np.ndarray) -> list[np.ndarray] | None:
    """Compute faces for a set of points, merging coplanar triangles.

    Args:
        coords: Array of shape ``(n, 3)`` with n >= 3.

    Returns:
        List of faces (each a 1-D array of vertex indices), or
        ``None`` if triangulation fails.
    """
    n = len(coords)
    if n == 3:
        return [np.array([0, 1, 2])]

    try:
        hull = ConvexHull(coords)
        return _merge_coplanar_faces(coords, hull.simplices)
    except QhullError:
        pass

    # Coplanar fallback: project to 2D and use Delaunay.
    return _triangulate_coplanar(coords)


def _triangulate_coplanar(coords: np.ndarray) -> list[np.ndarray] | None:
    """Triangulate coplanar points by projecting to 2D.

    Finds the best-fit plane via PCA, projects onto it, and runs
    2D Delaunay triangulation.

    Args:
        coords: Array of shape ``(n, 3)``.

    Returns:
        List of faces or ``None`` on failure.
    """
    centroid = coords.mean(axis=0)
    centred = coords - centroid

    # PCA: the two largest principal components span the plane.
    _, _, vt = np.linalg.svd(centred, full_matrices=False)
    projected = centred @ vt[:2].T  # (n, 2)

    try:
        tri = Delaunay(projected)
        return _merge_coplanar_faces(coords, tri.simplices)
    except QhullError:
        return None


def _merge_coplanar_faces(
    coords: np.ndarray,
    simplices: np.ndarray,
    cos_tol: float = 0.999,
) -> list[np.ndarray]:
    """Merge adjacent coplanar triangles into polygonal faces.

    Args:
        coords: Vertex coordinates, shape ``(n, 3)``.
        simplices: Triangle array, shape ``(n_tri, 3)``.
        cos_tol: Cosine threshold for treating normals as parallel.
            Default 0.999 (~2.5 degrees).

    Returns:
        List of faces, each a 1-D array of vertex indices ordered
        as a polygon loop.
    """
    n_tri = len(simplices)
    if n_tri == 0:
        return []

    # Compute face normals.
    v0 = coords[simplices[:, 0]]
    v1 = coords[simplices[:, 1]]
    v2 = coords[simplices[:, 2]]
    normals = np.cross(v1 - v0, v2 - v0)
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    normals = normals / norms

    # Orient normals outward (away from centroid).
    centroid = coords.mean(axis=0)
    face_centres = (v0 + v1 + v2) / 3.0
    outward = face_centres - centroid
    flip = np.sum(normals * outward, axis=1) < 0
    normals[flip] *= -1

    # Build edge-to-triangle adjacency.
    edge_to_tris: dict[tuple[int, int], list[int]] = defaultdict(list)
    for ti in range(n_tri):
        tri = simplices[ti]
        for a, b in [(tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])]:
            edge = (min(a, b), max(a, b))
            edge_to_tris[edge].append(ti)

    # BFS to group coplanar adjacent triangles.
    visited = np.zeros(n_tri, dtype=bool)
    groups: list[list[int]] = []

    for start in range(n_tri):
        if visited[start]:
            continue
        group = [start]
        visited[start] = True
        queue = deque([start])
        while queue:
            current = queue.popleft()
            tri = simplices[current]
            for a, b in [(tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])]:
                edge = (min(a, b), max(a, b))
                for neighbour_tri in edge_to_tris[edge]:
                    if visited[neighbour_tri]:
                        continue
                    dot = np.dot(normals[current], normals[neighbour_tri])
                    if dot > cos_tol:
                        visited[neighbour_tri] = True
                        group.append(neighbour_tri)
                        queue.append(neighbour_tri)
        groups.append(group)

    # For each group, extract boundary edges and order into a polygon.
    faces: list[np.ndarray] = []
    for group in groups:
        if len(group) == 1:
            faces.append(np.array(simplices[group[0]]))
            continue

        # Count edge occurrences within the group.
        edge_count: dict[tuple[int, int], int] = defaultdict(int)
        for ti in group:
            tri = simplices[ti]
            for a, b in [(tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])]:
                edge_count[(min(a, b), max(a, b))] += 1

        # Boundary edges appear exactly once.
        boundary = [(a, b) for (a, b), c in edge_count.items() if c == 1]

        # Order boundary edges into a vertex loop.
        loop = _order_boundary_loop(boundary)
        if loop is not None:
            faces.append(np.array(loop))
        else:
            # Fallback: emit original triangles.
            for ti in group:
                faces.append(np.array(simplices[ti]))

    return faces


def _order_boundary_loop(
    edges: list[tuple[int, int]],
) -> list[int] | None:
    """Order boundary edges into a closed polygon vertex loop.

    Args:
        edges: List of ``(a, b)`` vertex index pairs.

    Returns:
        Ordered list of vertex indices, or ``None`` if the edges
        don't form a single closed loop.
    """
    if not edges:
        return None

    adj: dict[int, list[int]] = defaultdict(list)
    for a, b in edges:
        adj[a].append(b)
        adj[b].append(a)

    start = edges[0][0]
    loop = [start]
    prev = -1
    current = start

    for _ in range(len(edges)):
        neighbours = adj[current]
        next_v = None
        for n in neighbours:
            if n != prev:
                next_v = n
                break
        if next_v is None:
            return None
        if next_v == start:
            break
        loop.append(next_v)
        prev = current
        current = next_v
    else:
        # Didn't close the loop.
        return None

    if len(loop) != len(edges):
        return None

    return loop
