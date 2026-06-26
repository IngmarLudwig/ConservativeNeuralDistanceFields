import numpy as np
 
class SphereNode:
    def __init__(self, center, radius, children=None, leaf_data=None):
        self.center = np.asarray(center, dtype=float)
        self.radius = float(radius)
        self.children = children or []
        self.leaf_data = leaf_data or []  # list of (center, radius)
 
    def contains_points(self, points):
        """
        Vectorized query:
        points: (N,3) array
        Returns: Boolean array (N,) indicating whether each point lies inside any sphere.
        """
        points = np.asarray(points, dtype=float)
        # Distance from points to this node’s bounding sphere
        d = np.linalg.norm(points - self.center, axis=1)
        inside = d <= self.radius
        if not np.any(inside):
            return np.zeros(len(points), dtype=bool)
 
        # Leaf node: check contained spheres directly
        if not self.children:
            if not self.leaf_data:
                return np.zeros(len(points), dtype=bool)
            # Vectorized extraction - avoid list comprehensions
            centers = np.array([sphere[0] for sphere in self.leaf_data])
            radii = np.array([sphere[1] for sphere in self.leaf_data])
            # Compute squared distances for efficiency
            diff = points[:, None, :] - centers[None, :, :]
            d2 = np.sum(diff**2, axis=2)
            r2 = radii[None, :] ** 2
            return np.any(d2 <= r2, axis=1)
 
        # Otherwise recurse into children
        result = np.zeros(len(points), dtype=bool)
        for child in self.children:
            # Only recurse for points inside this node’s sphere
            mask = np.logical_and(inside, ~result)
            if not np.any(mask):
                continue
            result[mask] |= child.contains_points(points[mask])
            if np.all(result):
                break
        return result
 
 
def bounding_sphere(centers, radii):
    centers = np.asarray(centers)
    avg_center = np.mean(centers, axis=0)
    max_dist = max(np.linalg.norm(c - avg_center) + r for c, r in zip(centers, radii))
    return avg_center, max_dist
 
 
def build_sphere_tree(spheres, max_leaf_size=8):
    if len(spheres) <= max_leaf_size:
        centers, radii = zip(*spheres)
        c, r = bounding_sphere(centers, radii)
        return SphereNode(c, r, leaf_data=spheres)
 
    # Vectorized extraction of centers - avoid list comprehension
    centers = np.array([sphere[0] for sphere in spheres])
    spreads = centers.max(axis=0) - centers.min(axis=0)
    axis = np.argmax(spreads)
    
    # Use numpy argsort for faster sorting instead of Python sort
    sort_indices = np.argsort(centers[:, axis])
    spheres_sorted = [spheres[i] for i in sort_indices]
    mid = len(spheres_sorted) // 2
 
    left = build_sphere_tree(spheres_sorted[:mid], max_leaf_size)
    right = build_sphere_tree(spheres_sorted[mid:], max_leaf_size)
 
    c, r = bounding_sphere([left.center, right.center], [left.radius, right.radius])
    return SphereNode(c, r, [left, right])