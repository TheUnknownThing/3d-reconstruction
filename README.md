# 3D Shadow Projection and Reconstruction

A Python-based visualization project that demonstrates how 3D objects can be reconstructed from their shadow projections using rim detection and geometric analysis. This project is part of an academic assignment for the course "PHY1202H".

## Usage

Run any of the three main visualization scripts:

```bash
# Ellipsoid shadow reconstruction
python 3d-ellipsoid.py

# Cube shadow reconstruction  
python 3d-cube.py

# Concave sphere reconstruction (demonstrating limitations)
python 3d-concave.py
```

Each script will:
1. Display an interactive 3D animation
2. Show the rim detection process in real-time
3. Visualize light rays and shadow projections
4. Accumulate reconstruction points
5. Save an animated GIF of the process

## Mathematical Foundation

The core algorithm is based on the following steps:

1. Surface Normal Calculation
    For a surface defined implicitly by $f(x, y, z) = 0$, the unit surface normal at point $(x, y, z)$ is:
    $$
    \hat{n} = \frac{\nabla f(x, y, z)}{\left\| \nabla f(x, y, z) \right\|}
    $$

2. Rim Condition
    The rim consists of points where the surface normal is perpendicular to the viewing direction $\hat{v}$:
    $$
    \left| \hat{n} \cdot \hat{v} \right| \approx 0
    $$

3. Shadow Projection
    To project a point $\mathbf{p}$ onto a plane with normal $\hat{n}$ at distance $d$ from the origin:
    $$
    \mathbf{p}' = \mathbf{p} - \left[ (\mathbf{p} \cdot \hat{n}) - d \right] \hat{n}
    $$

## Limitations

- Reconstruction quality depends on viewing angle sampling density
- Non-convex shapes cannot be fully reconstructed from silhouettes alone
- Numerical precision affects rim detection accuracy
- Real-world applications require handling of noise and occlusion

## License

This project is part of an academic assignment and is intended for educational purposes.

## Contributing

This is an educational project. For questions or suggestions, please refer to the course materials or contact the instructor.