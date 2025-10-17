import time
import numpy as np
import csv
from motion_planner import RRT_stats
from multi_drone import MultiDrone
import gc


def run_trial(env_file, num_drones=2, max_iterations=50000, step_size=3.0,
              goal_bias=0.1, bridge_prob=0.1, smoothing=False):
    """
    Run a single RRT trial for a given environment and number of drones.
    """
    sim = MultiDrone(num_drones=num_drones, environment_file=env_file, headless=True)
    start_time = time.time()
    path, nodes_explored, iter_found = RRT_stats(sim, max_iterations, step_size, goal_bias, bridge_prob, smoothing)
    duration = time.time() - start_time
    path_length = len(path) if path is not None else 0
    success = path is not None

    # Cleanup
    del sim
    gc.collect()

    return {
        "env_file": env_file,
        "num_drones": num_drones,
        "success": success,
        "path_length": path_length,
        "nodes_explored": nodes_explored,
        "duration": duration,
        "iter_found": iter_found
    }


def mean_ci(x):
    """Compute mean and 95% confidence interval."""
    if len(x) == 0:
        return float('nan'), float('nan')
    m = np.mean(x)
    ci = 1.96 * np.std(x) / np.sqrt(len(x))
    return m, ci


def run_experiments(env_files=None, drone_list=None, n_trials=10):
    """
    Run systematic experiments, handling both environment complexity and drone number studies.
    """
    results = []

    if env_files is not None and drone_list is None:
        # Environment complexity study (fixed num_drones)
        num_drones = 2 if drone_list is None else drone_list[0]
        for env_file in env_files:
            print(f"\nRunning trials for environment: {env_file} with {num_drones} drones")
            trials = [run_trial(env_file, num_drones=num_drones) for _ in range(n_trials)]
            results.extend(_compute_summary(trials, env_file, num_drones))

    if drone_list is not None:
        # Varying number of drones
        for i in range(len(drone_list)):
            print(f"\nRunning trials for {drone_list[i]} drones in environment: {env_files[i]}")
            trials = [run_trial(env_files[i], num_drones=drone_list[i]) for _ in range(n_trials)]
            results.extend(_compute_summary(trials, env_files[i], drone_list[i]))

    return results


def _compute_summary(trials, env_file, num_drones):
    """
    Compute mean, confidence intervals, and success rate from trials.
    """
    successes = [t["success"] for t in trials]
    durations = [t["duration"] for t in trials if t["success"]]
    path_lengths = [t["path_length"] for t in trials if t["success"]]
    nodes_explored_list = [t["nodes_explored"] for t in trials]
    iterations_found = [t["iter_found"] for t in trials if t["success"]]

    mean_duration, ci_duration = mean_ci(durations)
    mean_path, ci_path = mean_ci(path_lengths)
    mean_nodes, ci_nodes = mean_ci(nodes_explored_list)
    mean_iter, ci_iter = mean_ci(iterations_found)
    success_rate = np.mean(successes)

    return [{
        "env_file": env_file,
        "num_drones": num_drones,
        "success_rate": success_rate,
        "mean_duration": mean_duration,
        "ci_duration": ci_duration,
        "mean_path_length": mean_path,
        "ci_path_length": ci_path,
        "mean_nodes_explored": mean_nodes,
        "ci_nodes_explored": ci_nodes,
        "mean_iter_found": mean_iter,
        "ci_iter_found": ci_iter
    }]


def print_and_save_results(results, csv_file="experiment_results.csv"):
    """
    Print summary results and save as CSV.
    """
    headers = ["env_file", "num_drones", "success_rate", "mean_duration", "ci_duration",
               "mean_path_length", "ci_path_length", "mean_nodes_explored", "ci_nodes_explored",
               "mean_iter_found", "ci_iter_found"]

    for r in results:
        print(f"\nEnvironment: {r['env_file']}, Drones: {r['num_drones']}")
        print(f"Success rate: {r['success_rate']*100:.1f}%")
        print(f"Mean duration: {r['mean_duration']:.2f}s ± {r['ci_duration']:.2f}s")
        print(f"Mean path length: {r['mean_path_length']:.1f} ± {r['ci_path_length']:.1f}")
        print(f"Mean nodes explored: {r['mean_nodes_explored']:.1f} ± {r['ci_nodes_explored']:.1f}")
        print(f"Mean iteration found: {r['mean_iter_found']:.1f} ± {r['ci_iter_found']:.1f}")

    with open(csv_file, mode="w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for r in results:
            writer.writerow(r)
    print(f"\nResults saved to {csv_file}")