import yaml

# Generates a high complexity environment containing a staggered wall of spheres (2 drones)
def generate_env(filename="environment.yaml",
                              x_range=(0, 50), y_range=(20, 30), z_range=(0, 48),
                              spacing=5, radius=1.5):
    """
    Generate an environment YAML file with a wall of spheres in the given ranges.
    Middle plane is offset to create an alternating pattern.
    """
    env = {
        "bounds": {
            "x": [0, 50],
            "y": [0, 50],
            "z": [0, 50]
        },
        "initial_configuration": [
            [1, 1, 1],
            [50, 1, 50]
        ],
        "goals": [
            {"position": [50, 50, 50], "radius": 1.0},
            {"position": [1, 50, 1], "radius": 1.0}
        ],
        "obstacles": []
    }

    for y in list(range(y_range[0], y_range[1] + 1, spacing)):
        # Offset middle plane spheres by half the spacing
        x_offset = z_offset = spacing / 2 if y == (y_range[0] + y_range[1]) // 2 else 0

        for x in range(x_range[0], x_range[1] + 1, spacing):
            for z in range(z_range[0], z_range[1] + 1, spacing):
                env["obstacles"].append({
                    "type": "sphere",
                    "position": [x + x_offset, y, z + z_offset],
                    "radius": radius,
                    "color": "red"
                })

    with open(filename, "w") as f:
        yaml.dump(env, f, sort_keys=False)

    print(f"Environment saved to {filename} with {len(env['obstacles'])} spheres (with alternating middle plane).")


if __name__ == "__main__":
    generate_env("envs/environment.yaml")