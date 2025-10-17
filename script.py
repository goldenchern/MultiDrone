from experiments import run_experiments, print_and_save_results


def main():
    # Run environment complexity experiment for q4
    env_files = ["envs/env1.yaml", "envs/env2.yaml", "envs/env3.yaml", "envs/env4.yaml"]
    results = run_experiments(env_files=env_files, n_trials=10)
    print_and_save_results(results, "experiment_env_complexity.csv")

    # Run number of drones experiment for q5
    drone_list = [1, 3, 6, 12]
    env_files = ["envs/env5.yaml", "envs/env6.yaml", "envs/env7.yaml", "envs/env8.yaml"]
    results = run_experiments(env_files=env_files, drone_list=drone_list, n_trials=10)
    print_and_save_results(results, "experiment_num_drones.csv")


if __name__ == "__main__":
    main()
