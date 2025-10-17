from multi_drone import MultiDrone
from motion_planner import RRT


def main():
    # Initialize the MultiDrone environment
    sim = MultiDrone(num_drones=2, environment_file="envs/env2.yaml")

    solution_path = RRT(sim, max_iterations=50000, step_size=2.0, goal_bias=0.1, bridge_prob=0.1, smoothing=False)
    if solution_path is None:
        print("No path found")
    else:
        print(f"Found path with {len(solution_path)} waypoints")
        sim.visualize_paths(solution_path)


if __name__ == "__main__":
    main()
