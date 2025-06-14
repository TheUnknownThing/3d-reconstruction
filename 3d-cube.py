import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

# --- 1. Define Cube and Helper Functions ---

# Define the parameters for a cube
cube_size = 2.0

def get_cube_points(resolution=50):
    """Generates a mesh of points on the cube surface."""
    points = []
    
    # Generate points for each face of the cube
    t = np.linspace(-cube_size/2, cube_size/2, resolution)
    
    # Front and back faces (z = ±cube_size/2)
    for z in [-cube_size/2, cube_size/2]:
        x, y = np.meshgrid(t, t)
        for i in range(len(t)):
            for j in range(len(t)):
                points.append([x[i,j], y[i,j], z])
    
    # Left and right faces (x = ±cube_size/2)
    for x in [-cube_size/2, cube_size/2]:
        y, z = np.meshgrid(t, t)
        for i in range(len(t)):
            for j in range(len(t)):
                points.append([x, y[i,j], z[i,j]])
    
    # Top and bottom faces (y = ±cube_size/2)
    for y in [-cube_size/2, cube_size/2]:
        x, z = np.meshgrid(t, t)
        for i in range(len(t)):
            for j in range(len(t)):
                points.append([x[i,j], y, z[i,j]])
    
    return np.array(points)

def get_surface_normal(x, y, z):
    """Calculates the normal vector at a point on the cube surface."""
    # For a cube, the normal is along the axis of the face
    abs_x, abs_y, abs_z = abs(x), abs(y), abs(z)
    max_coord = max(abs_x, abs_y, abs_z)
    
    if abs_x == max_coord and abs(abs_x - cube_size/2) < 1e-6:
        return np.array([np.sign(x), 0, 0])
    elif abs_y == max_coord and abs(abs_y - cube_size/2) < 1e-6:
        return np.array([0, np.sign(y), 0])
    elif abs_z == max_coord and abs(abs_z - cube_size/2) < 1e-6:
        return np.array([0, 0, np.sign(z)])
    else:
        # For edge/corner points, use a simple approximation
        return np.array([np.sign(x), np.sign(y), np.sign(z)]) / np.sqrt(3)

def draw_cube_wireframe(ax):
    """Draw the wireframe of the cube."""
    # Define cube vertices
    vertices = np.array([
        [-cube_size/2, -cube_size/2, -cube_size/2],
        [cube_size/2, -cube_size/2, -cube_size/2],
        [cube_size/2, cube_size/2, -cube_size/2],
        [-cube_size/2, cube_size/2, -cube_size/2],
        [-cube_size/2, -cube_size/2, cube_size/2],
        [cube_size/2, -cube_size/2, cube_size/2],
        [cube_size/2, cube_size/2, cube_size/2],
        [-cube_size/2, cube_size/2, cube_size/2]
    ])
    
    # Define cube edges
    edges = [
        [0, 1], [1, 2], [2, 3], [3, 0],  # bottom face
        [4, 5], [5, 6], [6, 7], [7, 4],  # top face
        [0, 4], [1, 5], [2, 6], [3, 7]   # vertical edges
    ]
    
    for edge in edges:
        points = vertices[edge]
        ax.plot3D(*points.T, 'b-', alpha=0.3, linewidth=1)

# --- 2. Setup the 3D Plot ---

fig = plt.figure(figsize=(12, 9))
ax = fig.add_subplot(111, projection='3d')
ax.set_title("Visualizing the Rim and Projection")

draw_cube_wireframe(ax)

cube_points = get_cube_points(resolution=30)

reconstructed_scatter = ax.scatter([], [], [], c='r', s=10, label='Reconstructed Points')
rim_plot, = ax.plot([], [], [], 'k-', linewidth=3, label='Rim on Object')
outline_plot, = ax.plot([], [], [], 'g-', linewidth=3, label='Outline (Shadow)')
projection_plane_plot = None
light_ray_plots = [ax.plot([], [], [], 'y-', alpha=0.5)[0] for _ in range(100)] # For light rays

all_reconstructed_points = []

# --- 3. Animation Update Function ---

def update(frame):
    """This function is called for each frame of the animation."""
    global projection_plane_plot

    # --- a. Calculate projection direction ---
    theta_proj = frame * (2 * np.pi / num_frames)
    phi_proj = np.pi / 3

    proj_normal = np.array([
        np.cos(theta_proj) * np.sin(phi_proj),
        np.sin(theta_proj) * np.sin(phi_proj),
        np.cos(phi_proj)
    ])

    # --- b. Find the rim on the cube ---
    surface_normals = np.apply_along_axis(lambda p: get_surface_normal(p[0], p[1], p[2]), 1, cube_points)
    dot_products = np.dot(surface_normals, proj_normal)
    rim_indices = np.where(np.abs(dot_products) < 0.1)[0]  # Slightly larger threshold for cube
    rim_points = cube_points[rim_indices]
    
    if rim_points.shape[0] < 2: # Need at least 2 points to draw
        return reconstructed_scatter, rim_plot, outline_plot

    # --- c. Calculate the outline (shadow) on the projection plane ---
    plane_dist = -4.0
    outline_points = []
    for p in rim_points:
        p_proj = p - (np.dot(p, proj_normal) - plane_dist) * proj_normal
        outline_points.append(p_proj)
    outline_points = np.array(outline_points)

    # --- d. Draw the Projection Plane ---
    if projection_plane_plot:
        projection_plane_plot.remove()
    
    xx, yy = np.meshgrid(np.linspace(-5, 5, 2), np.linspace(-5, 5, 2))
    # Plane equation: ax + by + cz + d = 0 => z = (-ax - by - d) / c
    if abs(proj_normal[2]) > 1e-6: # Avoid division by zero
        zz = (-proj_normal[0] * xx - proj_normal[1] * yy + plane_dist) / proj_normal[2]
        projection_plane_plot = ax.plot_surface(xx, yy, zz, color='gray', alpha=0.2)
    
    # --- e. Draw the Light Rays ---
    # Draw a subset of rays from the rim to the outline
    num_rays_to_show = min(len(rim_points), len(light_ray_plots))
    for i in range(num_rays_to_show):
        ray_start = rim_points[i]
        ray_end = outline_points[i]
        light_ray_plots[i].set_data([ray_start[0], ray_end[0]], [ray_start[1], ray_end[1]])
        light_ray_plots[i].set_3d_properties([ray_start[2], ray_end[2]])
    # Hide unused ray plots
    for i in range(num_rays_to_show, len(light_ray_plots)):
        light_ray_plots[i].set_data([], [])
        light_ray_plots[i].set_3d_properties([])

    # --- f. Update the main plots (rim, outline, reconstruction) ---
    if len(rim_points) > 0:
        rim_plot.set_data(rim_points[:, 0], rim_points[:, 1])
        rim_plot.set_3d_properties(rim_points[:, 2])

        outline_plot.set_data(outline_points[:, 0], outline_points[:, 1])
        outline_plot.set_3d_properties(outline_points[:, 2])
        
        all_reconstructed_points.extend(rim_points)
        points_array = np.array(all_reconstructed_points)
        reconstructed_scatter._offsets3d = (points_array[:, 0], points_array[:, 1], points_array[:, 2])
    
    ax.set_title(f"Frame {frame+1}/{num_frames}: Finding the Rim and its Shadow")
    
    return reconstructed_scatter, rim_plot, outline_plot, projection_plane_plot

# --- 4. Create and Run the Animation ---

num_frames = 30

ax.set_xlim([-5, 5])
ax.set_ylim([-5, 5])
ax.set_zlim([-5, 5])
ax.set_xlabel("X-axis")
ax.set_ylabel("Y-axis")
ax.set_zlabel("Z-axis")
ax.legend(loc='upper left')
ax.set_aspect('equal')

ani = FuncAnimation(fig, update, frames=num_frames, blit=False, interval=500)

plt.tight_layout()
plt.show()

# generate GIF
from matplotlib.animation import PillowWriter
writer = PillowWriter(fps=15)
ani.save("cube_shadow_reconstruction.gif", writer=writer, dpi=60)