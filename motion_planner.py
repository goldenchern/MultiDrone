import numpy as np
import random

class TreeNode:
    """
    Represents a node in the RRT tree.
    """
    def __init__(self, configuration, parent=None):
        self.configuration = configuration
        self.parent = parent

def distance(q1, q2):
    """
    Compute Euclidean distance between two multi-drone configurations.
    """
    return np.linalg.norm(q1 - q2)

def nearest(tree_nodes, q_rand):
    """
    Find the nearest node in a tree to a sampled configuration.
    """
    return min(tree_nodes, key=lambda node: distance(node.configuration, q_rand))

def steer(q_near, q_rand, step_size):
    """
    Move from q_near toward q_rand by at most step_size.
    """
    direction = q_rand - q_near
    dist = np.linalg.norm(direction)
    if dist < 1e-6:
        return q_near.copy()
    step = min(step_size, dist)
    q_new = q_near + (direction / dist) * step
    return q_new

def bridge_sample(sim):
    """
    Bridge-test sampling for narrow passages.
    Sample two points in collision and take their midpoint. If midpoint is collision-free, return it.
    """
    for _ in range(10):  # try a few times
        q1 = np.random.uniform(sim._bounds[:,0], sim._bounds[:,1], size=(sim.N, 3))
        q2 = np.random.uniform(sim._bounds[:,0], sim._bounds[:,1], size=(sim.N, 3))
        if not sim.is_valid(q1) and not sim.is_valid(q2):
            q_mid = 0.5 * (q1 + q2)
            if sim.is_valid(q_mid):
                return q_mid
    return None

def reconstruct_path(node_start, node_goal):
    """
    Reconstruct path from start to goal by following parent links of both trees.
    """
    path_start, path_goal = [], []
    node = node_start
    while node:
        path_start.append(node.configuration)
        node = node.parent
    node = node_goal
    while node:
        path_goal.append(node.configuration)
        node = node.parent
    return path_start[::-1] + path_goal  # start->goal

def smooth_path(sim, path, iterations=100):
    """
    Shortcutting path smoothing: attempt to replace multi-segment paths
    with straight-line segments if collision-free.
    """
    if path is None or len(path) < 3:
        return path

    path = path.copy()
    for _ in range(iterations):
        if len(path) < 3:
            break
        # Randomly pick two indices i < j
        i = random.randint(0, len(path) - 2)
        j = random.randint(i + 1, len(path) - 1)

        q_i, q_j = path[i], path[j]

        # If direct motion is valid, remove intermediate nodes
        if sim.motion_valid(q_i, q_j):
            path = path[:i+1] + path[j:]
    return path

def RRT(sim, max_iterations=50000, step_size=2.0, goal_bias=0.1, bridge_prob=0.1, smoothing=True):
    """
    Bidirectional RRT planner for multi-drone motion planning with optional enhancements:
        - Goal biasing (towards the opposite tree)
        - Bridge-test sampling (for narrow passages)
        - Path smoothing (shortcutting)

    Args:
        sim (MultiDrone): Simulator object containing initial/goal configurations and collision checking
        max_iterations (int): Maximum number of RRT iterations
        step_size (float): Maximum extension distance per iteration
        goal_bias (float): Probability of sampling toward the opposite tree
        bridge_prob (float): Probability of performing bridge-test sampling
        smoothing (bool): Whether to perform path smoothing after a solution is found

    Returns:
        list of np.ndarray or None: Collision-free path from start to goal, or None if no path found
    """
    q_start = sim.initial_configuration
    q_goal = sim.goal_positions

    # Initialize trees
    tree_start = [TreeNode(q_start)]
    tree_goal = [TreeNode(q_goal)]

    for i in range(max_iterations):
        # Alternate tree expansion
        if i % 2 == 0:
            tree_from, tree_to = tree_start, tree_goal
        else:
            tree_from, tree_to = tree_goal, tree_start

        # Sampling
        r = random.random()
        if r < goal_bias:
            # Goal bias: sample a node from the opposite tree
            q_rand = random.choice(tree_to).configuration
        elif r < goal_bias + bridge_prob:
            # Bridge-test sampling
            q_rand = bridge_sample(sim)
            if q_rand is None:
                q_rand = np.random.uniform(sim._bounds[:,0], sim._bounds[:,1], size=(sim.N, 3))
        else:
            # Uniform sampling
            q_rand = np.random.uniform(sim._bounds[:,0], sim._bounds[:,1], size=(sim.N, 3))

        # Nearest node in current tree
        node_near = nearest(tree_from, q_rand)
        q_new = steer(node_near.configuration, q_rand, step_size)

        # Check motion validity
        if sim.motion_valid(node_near.configuration, q_new):
            node_new = TreeNode(q_new, parent=node_near)
            tree_from.append(node_new)

            # Try to connect to the other tree
            node_near_other = nearest(tree_to, q_new)
            if sim.motion_valid(q_new, node_near_other.configuration):
                # Connection found
                if tree_from is tree_start:
                    path = reconstruct_path(node_new, node_near_other)
                else:
                    path = reconstruct_path(node_near_other, node_new)

                # Optional path smoothing
                if smoothing:
                    path = smooth_path(sim, path, iterations=200)
                return path

    # No path found
    return None


# RRT wrapper for stats
def RRT_stats(sim, max_iterations=50000, step_size=2.0, goal_bias=0.1, bridge_prob=0.1, smoothing=False):
    """
    Run RRT and return path along with statistics.

    Args:
        sim (MultiDrone): Simulator object with environment and collision checking.
        max_iterations (int): Maximum number of RRT iterations.
        step_size (float): Maximum extension per iteration.
        goal_bias (float): Probability of sampling towards the opposite tree.
        bridge_prob (float): Probability of bridge-test sampling.
        smoothing (bool): Whether to smooth the path after solution.

    Returns:
        tuple: (path or None, nodes_explored, iteration_solution_found)
    """
    q_start = sim.initial_configuration
    q_goal = sim.goal_positions

    tree_start = [TreeNode(q_start)]
    tree_goal = [TreeNode(q_goal)]
    nodes_explored = 2  # initial nodes in both trees

    for i in range(max_iterations):
        tree_from, tree_to = (tree_start, tree_goal) if i % 2 == 0 else (tree_goal, tree_start)

        # Sampling
        r = np.random.rand()
        if r < goal_bias:
            q_rand = np.random.choice(tree_to).configuration
        elif r < goal_bias + bridge_prob:
            q_rand = bridge_sample(sim)
            if q_rand is None:
                q_rand = np.random.uniform(sim._bounds[:,0], sim._bounds[:,1], size=(sim.N,3))
        else:
            q_rand = np.random.uniform(sim._bounds[:,0], sim._bounds[:,1], size=(sim.N,3))

        node_near = nearest(tree_from, q_rand)
        q_new = steer(node_near.configuration, q_rand, step_size)

        if sim.motion_valid(node_near.configuration, q_new):
            node_new = TreeNode(q_new, parent=node_near)
            tree_from.append(node_new)
            nodes_explored += 1

            node_near_other = nearest(tree_to, q_new)
            if sim.motion_valid(q_new, node_near_other.configuration):
                path = reconstruct_path(node_new, node_near_other) if tree_from is tree_start else reconstruct_path(node_near_other, node_new)
                if smoothing:
                    path = smooth_path(sim, path, iterations=200)
                return path, nodes_explored, i+1

    return None, nodes_explored, max_iterations