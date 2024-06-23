# grabs expressed.json from /out and creates visualizations of the board at the saved steps.

import matplotlib.pyplot as plt
import numpy as np
import json
import os

# Load model output
with open("paper/code/replication/out/expressed.json") as j:
    model_output = json.load(j)

# Function to find global min and max of expressed values
def find_global_min_max(data):
    min_val = float('inf')
    max_val = float('-inf')
    
    for step_data in data:
        for point in step_data["state"]:
            expressed_val = point["expressed"]
            if expressed_val < min_val:
                min_val = expressed_val
            if expressed_val > max_val:
                max_val = expressed_val
    
    return min_val, max_val

# Get global min and max
global_min, global_max = find_global_min_max(model_output)

# Function to plot the grid state
def plot_grid_state(step, state, vmin, vmax):
    grid_size = 30
    grid = np.zeros((grid_size, grid_size))
    
    for point in state:
        grid[point["y"], point["x"]] = point["expressed"]
    
    plt.figure(figsize=(10, 10))
    plt.imshow(grid, cmap="viridis", interpolation="nearest", vmin=vmin, vmax=vmax)
    plt.colorbar(label="Expressed Attitude")
    plt.title(f"Step {step}")
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.grid(True)
    
    # Save the figure as a PNG file
    plt.savefig(os.path.join("paper/code/replication/out", f"step_{step}.png"))
    plt.close()

# Plot each step with global min and max for color scale
for step_data in model_output:
    plot_grid_state(step_data["step"], step_data["state"], global_min, global_max)
