# Gradient Descent Visualization

A 3D visualization tool demonstrating gradient descent optimization on a mathematical function.

## Overview

This project creates an animated visualization of gradient descent algorithm finding local minima on a 3D surface. It displays three different starting positions and tracks their paths as they converge toward local minima.

## Features

- 3D surface plot of a custom mathematical function combining sine and cosine waves
- Real-time visualization of gradient descent from multiple starting points
- Color-coded paths showing the optimization trajectory
- Rotating view providing different perspectives during the animation
- Option to save the animation as an MP4 file

## Mathematical Function

The visualization uses the function:
$z = \sin(5x) \cdot \cos(5y) / 5$

With gradient components:

- $\frac{\partial z}{\partial x} = \cos(5x) \cdot \cos(5y)$
- $\frac{\partial z}{\partial y} = -\sin(5x) \cdot \sin(5y)$

## Usage

Run the script to see the gradient descent animation:

```
python gradient_descent_viz.py
```

To save the animation as an MP4 file, uncomment the last line in the script:

```python
# ani.save('gradient_descent_multiple.mp4', writer='ffmpeg', fps=10, dpi=200)
```

## Parameters

The script includes several parameters that can be modified:

- `learning_rate`: Controls the step size (default: 0.01)
- `current_pos1`, `current_pos2`, `current_pos3`: Initial starting positions
- `frames` parameter in `FuncAnimation`: Number of gradient descent steps (default: 100)
- `interval` parameter in `FuncAnimation`: Time between frames in milliseconds (default: 100)

## Dependencies

- NumPy
- Matplotlib
- ffmpeg (optional, for saving animations)

## How It Works

1. The script defines a 3D surface based on the mathematical function
2. Three different starting points are initialized
3. At each animation step, the gradient is calculated for each point
4. Each point moves in the negative direction of the gradient (scaled by the learning rate)
5. The paths are tracked and visualized with different colors
6. The view angle gradually rotates to provide different perspectives

This visualization helps in understanding how gradient descent navigates a complex function landscape and how different starting points can lead to different local minima.
