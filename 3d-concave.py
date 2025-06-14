import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

# --- 1. Define a Non-Convex "Dented Sphere" ---

def get_dented_sphere_points(resolution=200):
    """Generates a mesh of points for a sphere with a concave dent."""
    phi = np.linspace(0, np.pi, resolution // 2)
    theta = np.linspace(0, 2 * np.pi, resolution)
    theta, phi = np.meshgrid(theta, phi)

    dent_depth = 0.8
    dent_sharpness = 15
    dent_center_phi = 0
    
    radius = 2.5 - dent_depth * np.exp(-dent_sharpness * (phi - dent_center_phi)**2)
    
    x = radius * np.sin(phi) * np.cos(theta)
    y = radius * np.sin(phi) * np.sin(theta)
    z = radius * np.cos(phi)
    
    return x, y, z

def get_surface_normal_numeric(x, y, z, i, j):
    """Calculates the surface normal numerically from the mesh grid."""
    # Use cross product of vectors along grid lines to find normal
    p_center = np.array([x[i, j], y[i, j], z[i, j]])
    
    # Get neighbors, handling grid boundaries
    p_v = np.array([x[min(i+1, x.shape[0]-1), j], y[min(i+1, x.shape[0]-1), j], z[min(i+1, x.shape[0]-1), j]])
    p_u = np.array([x[i, min(j+1, x.shape[1]-1)], y[i, min(j+1, x.shape[1]-1)], z[i, min(j+1, x.shape[1]-1)]])

    v_u = p_u - p_center
    v_v = p_v - p_center
    
    normal = np.cross(v_u, v_v)
    norm_mag = np.linalg.norm(normal)
    return normal / norm_mag if norm_mag > 0 else np.array([0,0,1])

# --- 2. Setup the 3D Plot ---

fig = plt.figure(figsize=(12, 9))
ax = fig.add_subplot(111, projection='3d')
ax.set_title("Attempting to Reconstruct a Concave Object")

# Plot the original object
x_orig, y_orig, z_orig = get_dented_sphere_points()
ax.plot_surface(x_orig, y_orig, z_orig, color='blue', alpha=0.15, rstride=5, cstride=5)

# Initialize plot elements
reconstructed_scatter = ax.scatter([], [], [], c='r', s=10, label='Reconstructed Points (Convex Hull)')
rim_plot, = ax.plot([], [], [], 'k-', linewidth=3, label='Rim on Object')
outline_plot, = ax.plot([], [], [], 'g-', linewidth=3, label='Outline (Shadow)')

all_reconstructed_points = []

# --- 3. Animation Update Function ---

def update(frame):
    # --- a. Calculate projection direction ---
    theta_proj = frame * (2 * np.pi / num_frames)
    phi_proj = np.pi / 2.5 # Tilted view to see the dent better

    proj_normal = np.array([
        np.cos(theta_proj) * np.sin(phi_proj),
        np.sin(theta_proj) * np.sin(phi_proj),
        np.cos(phi_proj)
    ])

    # --- b. Find the rim ---
    rim_points = []
    for i in range(x_orig.shape[0]):
        for j in range(x_orig.shape[1]):
            normal = get_surface_normal_numeric(x_orig, y_orig, z_orig, i, j)
            if abs(np.dot(normal, proj_normal)) < 0.02: # Tolerance for numeric normals
                rim_points.append([x_orig[i, j], y_orig[i, j], z_orig[i, j]])
    
    rim_points = np.array(rim_points)
    if rim_points.shape[0] < 2:
        return reconstructed_scatter, outline_plot

    # --- c. Calculate the outline (shadow) ---
    outline_points = []
    for p in rim_points:
        p_proj = p - np.dot(p, proj_normal) * proj_normal
    outline_points = np.array([p - np.dot(p, proj_normal) * proj_normal for p in rim_points])

    # --- d. Update plots ---
    if rim_points.any():
        center = np.mean(rim_points, axis=0)
        angles = np.arctan2((rim_points - center)[:, 1], (rim_points - center)[:, 0])
        rim_points_sorted = rim_points[np.argsort(angles)]

    if outline_points.any():
        center_outline = np.mean(outline_points, axis=0)
        angles_outline = np.arctan2((outline_points - center_outline)[:, 1], (outline_points - center_outline)[:, 0])
        outline_points_sorted = outline_points[np.argsort(angles_outline)]
        outline_plot.set_data(outline_points_sorted[:, 0], outline_points_sorted[:, 1])
        outline_plot.set_3d_properties(outline_points_sorted[:, 2])
    
    all_reconstructed_points.extend(rim_points)
    points_array = np.array(all_reconstructed_points)
    if points_array.any():
        reconstructed_scatter._offsets3d = (points_array[:, 0], points_array[:, 1], points_array[:, 2])
    
    ax.set_title(f"Frame {frame+1}/{num_frames}: Shadow 'Fills In' the Dent")
    
    return reconstructed_scatter, outline_plot

# --- 4. Create and Run the Animation ---

num_frames = 60

ax.set_xlim([-3, 3])
ax.set_ylim([-3, 3])
ax.set_zlim([-3, 3])
ax.set_xlabel("X-axis")
ax.set_ylabel("Y-axis")
ax.set_zlabel("Z-axis")
ax.legend(loc='upper left')
ax.view_init(elev=50, azim=-20)
ani = FuncAnimation(fig, update, frames=num_frames, blit=False, interval=50)

plt.tight_layout()
plt.show()

# generate GIF
from matplotlib.animation import PillowWriter
writer = PillowWriter(fps=20)
ani.save("dented_sphere_reconstruction.gif", writer=writer, dpi=60)
