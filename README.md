# 2D/3D Shadow Projection and Reconstruction

A Python-based visualization project that demonstrates how 2D and 3D objects can be reconstructed from their projections using support function analysis and rim detection techniques. This project implements Olver's reconstruction algorithm for 2D shapes and shadow-based reconstruction for 3D objects, created as part of an academic assignment for the course "PHY1202H".

## Usage

### 2D Shape Reconstruction (Olver's Algorithm)

Run the 2D reconstruction scripts to see how shapes can be reconstructed from support function measurements:

```bash
# Basic ellipse reconstruction with detailed step-by-step visualization
python 2d-ellipsoid.py

# Complex shapes reconstruction (super-ellipse, star, flower, heart)
python 2d-complex-shapes.py
```

### 3D Shadow Reconstruction

Run any of the three main 3D visualization scripts:

```bash
# Ellipsoid shadow reconstruction
python 3d-ellipsoid.py

# Cube shadow reconstruction  
python 3d-cube.py

# Concave sphere reconstruction (demonstrating limitations)
python 3d-concave.py
```

Each script will:
1. Display an interactive animation
2. Show the reconstruction process in real-time
3. Visualize mathematical concepts step-by-step
4. Save an animated GIF of the process

## Mathematical Foundation

### 2D Shape Reconstruction (Olver's Algorithm)

The 2D reconstruction is based on Olver's method using support functions:

1. **Support Function**
   For a convex shape, the support function $\rho(\theta)$ gives the distance from the origin to the supporting line in direction $\theta$:
   $$
   \rho(\theta) = \max_{(x,y) \in \text{shape}} \{x \cos\theta + y \sin\theta\}
   $$

2. **Olver's Reconstruction Formula**
   Given the support function $\rho(\theta)$ and its derivative $\rho'(\theta)$, the boundary points are reconstructed as:
   $$
   \begin{align}
   x(\theta) &= \rho(\theta) \cos\theta - \rho'(\theta) \sin\theta \\
   y(\theta) &= \rho(\theta) \sin\theta + \rho'(\theta) \cos\theta
   \end{align}
   $$

3. **Geometric Interpretation**
   The reconstruction formula represents the envelope of the family of supporting lines (tangents) to the original shape.

### 3D Shadow Reconstruction

The 3D algorithm is based on rim detection and shadow projection:

1. **Surface Normal Calculation**
    For a surface defined implicitly by $f(x, y, z) = 0$, the unit surface normal at point $(x, y, z)$ is:
    $$
    \hat{n} = \frac{\nabla f(x, y, z)}{\left\| \nabla f(x, y, z) \right\|}
    $$

2. **Rim Condition**
    The rim consists of points where the surface normal is perpendicular to the viewing direction $\hat{v}$:
    $$
    \left| \hat{n} \cdot \hat{v} \right| \approx 0
    $$

3. **Shadow Projection**
    To project a point $\mathbf{p}$ onto a plane with normal $\hat{n}$ at distance $d$ from the origin:
    $$
    \mathbf{p}' = \mathbf{p} - \left[ (\mathbf{p} \cdot \hat{n}) - d \right] \hat{n}
    $$

## Features

### 2D Reconstruction Features
- **Ellipse Reconstruction**: Step-by-step visualization of Olver's algorithm applied to ellipses
- **Complex Shapes**: Reconstruction of super-ellipses, stars, flowers, and heart shapes
- **Support Function Visualization**: Real-time display of support functions and their derivatives
- **Envelope Formation**: Geometric visualization of how tangent lines form the reconstructed boundary
- **Error Analysis**: Quantitative measurement of reconstruction accuracy

### 3D Reconstruction Features
- **Interactive 3D Visualization**: Real-time rim detection and shadow projection
- **Multiple Object Types**: Ellipsoids, cubes, and concave shapes
- **Light Ray Tracing**: Visual representation of projection geometry
- **Limitation Demonstration**: Shows challenges with non-convex objects

## Limitations

### 2D Reconstruction Limitations
- Olver's algorithm works perfectly for convex shapes
- Non-convex shapes may require additional techniques beyond support functions
- Numerical differentiation introduces small errors in the derivative calculation
- Sampling density affects reconstruction smoothness

### 3D Reconstruction Limitations
- Reconstruction quality depends on viewing angle sampling density
- Non-convex shapes cannot be fully reconstructed from silhouettes alone
- Numerical precision affects rim detection accuracy
- Real-world applications require handling of noise and occlusion

## Dependencies

- Python 3.7+
- NumPy
- Matplotlib
- Animation support (for GIF generation)

## License

This project is part of an academic assignment and is intended for educational purposes.

## Contributing

This is an educational project. For questions or suggestions, please refer to the course materials or contact the instructor.