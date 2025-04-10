import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation


# Define a 2D function that combines sine and cosine waves
def z_function(x, y):
    return np.sin(5 * x) * np.cos(5 * y) / 5


# Define a function to calculate the gradient of z_function
def calculate_gradient(x, y):
    return np.cos(5 * x) * np.cos(5 * y), -np.sin(5 * x) * np.sin(5 * y)


# Create coordinate arrays for x and y, from -1 to 1 with step 0.05
x = np.arange(-1, 1, 0.05)
y = np.arange(-1, 1, 0.05)

# Create 2D coordinate matrices from the 1D arrays
X, Y = np.meshgrid(x, y)

# Compute z values by applying the function to the coordinate matrices
Z = z_function(X, Y)

# Create a figure and 3D subplot
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection="3d")

# Set initial positions
current_pos1 = (-0.5, -0.4, z_function(0.1, 0.4))
current_pos2 = (0.7, 0.4, z_function(0.7, 0.4))
current_pos3 = (0.6, -0.6, z_function(0.6, 0.6))
learning_rate = 0.01

# Create separate position history lists for each starting point
positions1 = [current_pos1]
positions2 = [current_pos2]
positions3 = [current_pos3]


def update_position(current_pos):
    """Calculate the next position using gradient descent"""
    X_derivative, Y_derivative = calculate_gradient(current_pos[0], current_pos[1])
    X_new = current_pos[0] - learning_rate * X_derivative
    Y_new = current_pos[1] - learning_rate * Y_derivative
    return (X_new, Y_new, z_function(X_new, Y_new))


# Animation function for each frame update
def update(num):
    global current_pos1, current_pos2, current_pos3
    global positions1, positions2, positions3

    ax.clear()

    # Plot the surface
    surf = ax.plot_surface(X, Y, Z, cmap="viridis", alpha=0.2)

    # Update positions if not first frame
    if num > 0:
        current_pos1 = update_position(current_pos1)
        current_pos2 = update_position(current_pos2)
        current_pos3 = update_position(current_pos3)

        # Store updated positions
        positions1.append(current_pos1)
        positions2.append(current_pos2)
        positions3.append(current_pos3)

    # Plot paths and current positions for all three starting points

    # Path 1 (magenta)
    path_x1 = [pos[0] for pos in positions1[: num + 1]]
    path_y1 = [pos[1] for pos in positions1[: num + 1]]
    path_z1 = [pos[2] for pos in positions1[: num + 1]]
    if len(path_x1) > 1:
        ax.plot(path_x1, path_y1, path_z1, "magenta", linewidth=2)
    ax.scatter(
        current_pos1[0],
        current_pos1[1],
        current_pos1[2],
        color="magenta",
        s=100,
        label="Position 1",
    )

    # Path 2 (blue)
    path_x2 = [pos[0] for pos in positions2[: num + 1]]
    path_y2 = [pos[1] for pos in positions2[: num + 1]]
    path_z2 = [pos[2] for pos in positions2[: num + 1]]
    if len(path_x2) > 1:
        ax.plot(path_x2, path_y2, path_z2, "blue", linewidth=2)
    ax.scatter(
        current_pos2[0],
        current_pos2[1],
        current_pos2[2],
        color="blue",
        s=100,
        label="Position 2",
    )

    # Path 3 (orange)
    path_x3 = [pos[0] for pos in positions3[: num + 1]]
    path_y3 = [pos[1] for pos in positions3[: num + 1]]
    path_z3 = [pos[2] for pos in positions3[: num + 1]]
    if len(path_x3) > 1:
        ax.plot(path_x3, path_y3, path_z3, "orange", linewidth=2)
    ax.scatter(
        current_pos3[0],
        current_pos3[1],
        current_pos3[2],
        color="orange",
        s=100,
        label="Position 3",
    )

    # Set consistent view angle
    ax.view_init(elev=30, azim=num / 1.5)  # Slowly rotate as animation progresses
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title(f"Gradient Descent Step {num}")
    ax.legend()

    # Print status for one of the positions (or all if desired)
    print(
        f"Step {num}: Position 1: ({current_pos1[0]:.4f}, {current_pos1[1]:.4f}, {current_pos1[2]:.4f})"
    )

    return (surf,)


# Create animation - adjust frames to control number of steps
ani = animation.FuncAnimation(fig, update, frames=100, interval=100, blit=False)

plt.tight_layout()
plt.show()

# Uncomment to save animation
# ani.save('gradient_descent_multiple.mp4', writer='ffmpeg', fps=10, dpi=200)
