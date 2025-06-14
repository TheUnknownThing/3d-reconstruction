import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# --- 1. Define Shape Classes ---

class Shape:
    """Base class for shapes."""
    def __init__(self, name):
        self.name = name
    
    def get_boundary_points(self, num_points=300):
        """Get boundary points."""
        raise NotImplementedError
    
    def support_function(self, theta):
        """Support function."""
        raise NotImplementedError
    
    def support_function_derivative(self, theta):
        """Support function derivative."""
        raise NotImplementedError


class SuperEllipse(Shape):
    """Super-ellipse (Lamé curve)."""
    def __init__(self, a=4.0, b=3.0, n=2.5):
        super().__init__(f"Super-ellipse (n={n})")
        self.a = a
        self.b = b
        self.n = n
    
    def get_boundary_points(self, num_points=300):
        t = np.linspace(0, 2 * np.pi, num_points)
        x = self.a * np.sign(np.cos(t)) * (np.abs(np.cos(t)))**(2/self.n)
        y = self.b * np.sign(np.sin(t)) * (np.abs(np.sin(t)))**(2/self.n)
        return x, y
    
    def support_function(self, theta):
        # Numerical calculation of support function
        x, y = self.get_boundary_points(1000)
        rho_vals = []
        
        for t in theta:
            # Calculate maximum projection in direction (cos(t), sin(t))
            projections = x * np.cos(t) + y * np.sin(t)
            rho_vals.append(np.max(projections))
        
        return np.array(rho_vals)
    
    def support_function_derivative(self, theta):
        # Numerical derivative
        h = 1e-6
        rho_plus = self.support_function(theta + h)
        rho_minus = self.support_function(theta - h)
        return (rho_plus - rho_minus) / (2 * h)


class Star(Shape):
    """Star shape."""
    def __init__(self, outer_radius=4.0, inner_radius=2.0, num_points=5):
        super().__init__(f"{num_points}-pointed star")
        self.outer_radius = outer_radius
        self.inner_radius = inner_radius
        self.num_points = num_points
    
    def get_boundary_points(self, num_points=300):
        angles = np.linspace(0, 2 * np.pi, num_points)
        x, y = [], []
        
        for angle in angles:
            # Calculate radius for current angle
            star_angle = (angle % (2 * np.pi / self.num_points)) * self.num_points
            if star_angle <= np.pi:
                # From outer vertex to inner vertex
                t = star_angle / np.pi
                radius = self.outer_radius * (1 - t) + self.inner_radius * t
            else:
                # From inner vertex to outer vertex
                t = (star_angle - np.pi) / np.pi
                radius = self.inner_radius * (1 - t) + self.outer_radius * t
            
            x.append(radius * np.cos(angle))
            y.append(radius * np.sin(angle))
        
        return np.array(x), np.array(y)
    
    def support_function(self, theta):
        x, y = self.get_boundary_points(1000)
        rho_vals = []
        
        for t in theta:
            projections = x * np.cos(t) + y * np.sin(t)
            rho_vals.append(np.max(projections))
        
        return np.array(rho_vals)
    
    def support_function_derivative(self, theta):
        h = 1e-6
        rho_plus = self.support_function(theta + h)
        rho_minus = self.support_function(theta - h)
        return (rho_plus - rho_minus) / (2 * h)


class Flower(Shape):
    """Flower shape."""
    def __init__(self, radius=3.0, petals=6, petal_depth=0.3):
        super().__init__(f"{petals}-petal flower")
        self.radius = radius
        self.petals = petals
        self.petal_depth = petal_depth
    
    def get_boundary_points(self, num_points=300):
        angles = np.linspace(0, 2 * np.pi, num_points)
        r = self.radius * (1 + self.petal_depth * np.cos(self.petals * angles))
        x = r * np.cos(angles)
        y = r * np.sin(angles)
        return x, y
    
    def support_function(self, theta):
        x, y = self.get_boundary_points(1000)
        rho_vals = []
        
        for t in theta:
            projections = x * np.cos(t) + y * np.sin(t)
            rho_vals.append(np.max(projections))
        
        return np.array(rho_vals)
    
    def support_function_derivative(self, theta):
        h = 1e-6
        rho_plus = self.support_function(theta + h)
        rho_minus = self.support_function(theta - h)
        return (rho_plus - rho_minus) / (2 * h)


class Heart(Shape):
    """Heart shape."""
    def __init__(self, scale=1.0):
        super().__init__("Heart")
        self.scale = scale
    
    def get_boundary_points(self, num_points=300):
        t = np.linspace(0, 2 * np.pi, num_points)
        x = self.scale * 16 * np.sin(t)**3
        y = self.scale * (13 * np.cos(t) - 5 * np.cos(2*t) - 2 * np.cos(3*t) - np.cos(4*t))
        return x/4, y/4  # Scale to appropriate size
    
    def support_function(self, theta):
        x, y = self.get_boundary_points(1000)
        rho_vals = []
        
        for t in theta:
            projections = x * np.cos(t) + y * np.sin(t)
            rho_vals.append(np.max(projections))
        
        return np.array(rho_vals)
    
    def support_function_derivative(self, theta):
        h = 1e-6
        rho_plus = self.support_function(theta + h)
        rho_minus = self.support_function(theta - h)
        return (rho_plus - rho_minus) / (2 * h)


# --- 3. Helper Functions ---

def reconstruct_shape(rho_vals, rho_prime_vals, theta_range):
    """Olver reconstruction formula."""
    cos_t = np.cos(theta_range)
    sin_t = np.sin(theta_range)
    
    x_reconstructed = rho_vals * cos_t - rho_prime_vals * sin_t
    y_reconstructed = rho_vals * sin_t + rho_prime_vals * cos_t
    
    return x_reconstructed, y_reconstructed


def calculate_reconstruction_error(x_orig, y_orig, x_recon, y_recon):
    """Calculate reconstruction error."""
    if len(x_recon) == 0:
        return float('inf')
    
    distances = []
    for i in range(len(x_recon)):
        dist_to_orig = np.sqrt((x_orig - x_recon[i])**2 + (y_orig - y_recon[i])**2)
        distances.append(np.min(dist_to_orig))
    
    return np.mean(distances)


# --- 4. Main Program ---
if __name__ == "__main__":
    # Define multiple shapes - remove ellipse, focus on complex shapes
    shapes = [
        SuperEllipse(4.0, 3.0, 2.5),
        Star(4.0, 2.0, 5),
        Flower(3.0, 8, 0.4),
        Heart(1.2)
    ]
    
    current_shape_idx = 0
    num_angles = 60  # Reduce angle sampling points to speed up reconstruction
    
    # Initialize current shape
    shape = shapes[current_shape_idx]
    theta = np.linspace(0, 2 * np.pi, num_angles, endpoint=False)
    
    # Setup animation interface - use 2x3 layout to optimize screen display
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3, figsize=(18, 10))
    
    # Adjust subplot spacing
    plt.tight_layout(pad=2.0)
    
    colors = ['red', 'green', 'blue', 'orange', 'purple']
    
    # Configure subplots
    # ax1: Original shape
    ax1.set_title("①Original Complex Shape", fontsize=12, weight='bold')
    ax1.set_aspect('equal')
    ax1.grid(True, alpha=0.3)
    original_line, = ax1.plot([], [], 'b-', linewidth=3, label='Original Shape')
    ax1.legend(fontsize=10)
    
    # ax2: Reconstruction process
    ax2.set_title("②Reconstruction Process", fontsize=12, weight='bold')
    ax2.set_aspect('equal')
    ax2.grid(True, alpha=0.3)
    ax2.plot([], [], 'b--', alpha=0.5, linewidth=2, label='Original Shape')
    reconstruction_line, = ax2.plot([], [], 'r-', linewidth=3, label='Reconstructed Shape')
    current_point, = ax2.plot([], [], 'ro', markersize=8, label='Current Reconstruction Point')
    ax2.legend(fontsize=10)
    
    # ax3: Support function
    ax3.set_title("③Support Function ρ(θ)", fontsize=12, weight='bold')
    ax3.set_xlabel("θ (radians)", fontsize=10)
    ax3.set_ylabel("ρ(θ)", fontsize=10)
    ax3.grid(True, alpha=0.3)
    support_line, = ax3.plot([], [], 'r-', linewidth=3)
    support_points, = ax3.plot([], [], 'ro', markersize=6)
    
    # ax4: Support function derivative
    ax4.set_title("④Support Function Derivative ρ'(θ)", fontsize=12, weight='bold')
    ax4.set_xlabel("θ (radians)", fontsize=10)
    ax4.set_ylabel("ρ'(θ)", fontsize=10)
    ax4.grid(True, alpha=0.3)
    derivative_line, = ax4.plot([], [], 'g-', linewidth=3)
    derivative_points, = ax4.plot([], [], 'go', markersize=6)
    
    # ax5: Final comparison
    ax5.set_title("⑤Original vs Reconstruction Comparison", fontsize=12, weight='bold')
    ax5.set_aspect('equal')
    ax5.grid(True, alpha=0.3)
    final_original, = ax5.plot([], [], 'b--', linewidth=3, alpha=0.7, label='Original Shape')
    final_reconstruction, = ax5.plot([], [], 'r-', linewidth=3, label='Olver Reconstruction')
    ax5.legend(fontsize=10)
    
    # ax6: Multi-shape comparison
    ax6.set_title("⑥Complex Shape Reconstruction Showcase", fontsize=12, weight='bold')
    ax6.set_aspect('equal')
    ax6.grid(True, alpha=0.3)
    shape_lines = {}
    
    def animate(frame):
        global current_shape_idx, shape, theta
        
        # Switch shape every 70 frames (reduce demonstration time)
        frames_per_shape = 70
        new_shape_idx = (frame // frames_per_shape) % len(shapes)
        if new_shape_idx != current_shape_idx:
            current_shape_idx = new_shape_idx
            shape = shapes[current_shape_idx]
        
        # Current frame position within current shape
        shape_frame = frame % frames_per_shape
        current_points = min(shape_frame + 1, num_angles)
        
        if current_points == 0:
            return []
        
        # Get shape data
        x_orig, y_orig = shape.get_boundary_points()
        rho = shape.support_function(theta)
        rho_prime = shape.support_function_derivative(theta)
        
        # Current data
        theta_current = theta[:current_points]
        rho_current = rho[:current_points]
        rho_prime_current = rho_prime[:current_points]
        
        # Update original shape
        original_line.set_data(x_orig, y_orig)
        max_range = max(np.max(np.abs(x_orig)), np.max(np.abs(y_orig))) * 1.1
        ax1.set_xlim(-max_range, max_range)
        ax1.set_ylim(-max_range, max_range)
        ax1.set_title(f"Original Shape: {shape.name}", fontsize=12)
        
        # Update support function
        support_line.set_data(theta, rho)
        support_points.set_data(theta_current, rho_current)
        ax3.set_xlim(0, 2*np.pi)
        ax3.set_ylim(np.min(rho)*0.9, np.max(rho)*1.1)
        
        # Update derivative
        derivative_line.set_data(theta, rho_prime)
        derivative_points.set_data(theta_current, rho_prime_current)
        ax4.set_xlim(0, 2*np.pi)
        if np.max(rho_prime) != np.min(rho_prime):
            ax4.set_ylim(np.min(rho_prime)*1.1, np.max(rho_prime)*1.1)
        
        # Reconstruction process
        if current_points > 2:
            x_recon, y_recon = reconstruct_shape(rho_current, rho_prime_current, theta_current)
            reconstruction_line.set_data(x_recon, y_recon)
            
            if len(x_recon) > 0:
                current_point.set_data([x_recon[-1]], [y_recon[-1]])
            
            ax2.clear()
            ax2.set_title("Reconstruction Process", fontsize=12)
            ax2.set_aspect('equal')
            ax2.grid(True, alpha=0.3)
            ax2.plot(x_orig, y_orig, 'b--', alpha=0.5, linewidth=1, label='Original Shape')
            ax2.plot(x_recon, y_recon, 'r-', linewidth=2, label='Reconstructed Shape')
            if len(x_recon) > 0:
                ax2.plot(x_recon[-1], y_recon[-1], 'ro', markersize=6, label='Current Reconstruction Point')
            ax2.set_xlim(-max_range, max_range)
            ax2.set_ylim(-max_range, max_range)
            ax2.legend()
        
        # Final comparison (when shape is complete)
        if current_points >= num_angles - 5:
            x_final, y_final = reconstruct_shape(rho, rho_prime, theta)
            final_original.set_data(x_orig, y_orig)
            final_reconstruction.set_data(x_final, y_final)
            ax5.set_xlim(-max_range, max_range)
            ax5.set_ylim(-max_range, max_range)
            
            # Calculate error
            error = calculate_reconstruction_error(x_orig, y_orig, x_final, y_final)
            ax5.set_title(f"Final Comparison - Error: {error:.4f}", fontsize=12)
        
        # Multi-shape comparison
        if current_points >= num_angles - 5:  # Add to comparison plot when shape is complete
            x_final, y_final = reconstruct_shape(rho, rho_prime, theta)
            color = colors[current_shape_idx % len(colors)]
            
            if shape.name not in shape_lines:
                ax6.plot(x_orig, y_orig, '--', color=color, alpha=0.3, linewidth=1)
                line, = ax6.plot(x_final, y_final, '-', color=color, linewidth=2, 
                               label=f'{shape.name} Reconstruction')
                shape_lines[shape.name] = line
            
            ax6.set_xlim(-8, 8)
            ax6.set_ylim(-8, 8)
            ax6.legend(fontsize=10)
        
        # Progress information
        progress = (current_points / num_angles) * 100
        fig.suptitle(f"Olver Reconstruction Algorithm - Complex Shapes Demo | Current: {shape.name} | Progress: {progress:.1f}%", 
                    fontsize=16, weight='bold')
        
        return [original_line, support_line, support_points, derivative_line, 
                derivative_points, reconstruction_line, current_point, 
                final_original, final_reconstruction]

    # Create and run animation - optimize time parameters
    frames_per_shape = 70  # Slightly more than num_angles, leave completion time
    frames = len(shapes) * frames_per_shape  # Total frames
    anim = FuncAnimation(fig, animate, frames=frames, interval=60, blit=False, repeat=True)  # Faster playback speed
    
    plt.tight_layout()
    
    # Save as GIF - suitable for PPT insertion, optimize parameters
    print("Saving complex shape reconstruction animation as GIF format...")
    anim.save('complex_shapes_olver_reconstruction.gif', 
              writer='pillow', 
              fps=15,  # Higher frame rate for smoother animation
              dpi=70,   # Moderate resolution
              savefig_kwargs={'pad_inches': 0.1})
    print("✓ GIF saved as: complex_shapes_olver_reconstruction.gif")
    
    # Display animation
    plt.show()
