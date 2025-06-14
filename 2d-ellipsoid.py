import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# --- 1. Define Ellipse and Helper Functions ---

def get_ellipse_points(a, b, num_points=200):
    """Generate boundary points of an ellipse."""
    angles = np.linspace(0, 2 * np.pi, num_points)
    x = a * np.cos(angles)
    y = b * np.sin(angles)
    return x, y

def support_function_ellipse(theta, a, b):
    """Calculate the support function rho(theta) for an ellipse."""
    return np.sqrt((a * np.cos(theta))**2 + (b * np.sin(theta))**2)

def support_function_derivative_ellipse(theta, a, b):
    """Calculate the derivative of the ellipse support function rho'(theta)."""
    rho = support_function_ellipse(theta, a, b)
    rho[rho == 0] = 1e-9 
    numerator = (b**2 - a**2) * np.sin(theta) * np.cos(theta)
    return numerator / rho

def reconstruct_shape(rho_vals, rho_prime_vals, theta_range):
    """Reconstruct object boundary using Olver's formula."""
    cos_t = np.cos(theta_range)
    sin_t = np.sin(theta_range)
    
    x_reconstructed = rho_vals * cos_t - rho_prime_vals * sin_t
    y_reconstructed = rho_vals * sin_t + rho_prime_vals * cos_t
    
    return x_reconstructed, y_reconstructed

def get_tangent_line(theta_i, rho_i, rho_prime_i, length=6):
    """Get the endpoints of the tangent line."""
    normal_x = np.cos(theta_i)
    normal_y = np.sin(theta_i)
    
    contact_x = rho_i * normal_x - rho_prime_i * normal_y
    contact_y = rho_i * normal_y + rho_prime_i * normal_x
    
    tangent_x = -normal_y
    tangent_y = normal_x
    
    x1 = contact_x - length * tangent_x
    x2 = contact_x + length * tangent_x
    y1 = contact_y - length * tangent_y
    y2 = contact_y + length * tangent_y
    
    return [x1, x2], [y1, y2], contact_x, contact_y

def calculate_reconstruction_error(x_orig, y_orig, x_recon, y_recon):
    """Calculate reconstruction error."""
    if len(x_recon) == 0:
        return float('inf')
    
    # Calculate nearest point distances
    distances = []
    for i in range(len(x_recon)):
        dist_to_orig = np.sqrt((x_orig - x_recon[i])**2 + (y_orig - y_recon[i])**2)
        distances.append(np.min(dist_to_orig))
    
    return np.mean(distances)

# --- 3. Main Program ---
if __name__ == "__main__":
    # Ellipse parameters
    ellipse_a = 5.0  # Semi-major axis
    ellipse_b = 3.0  # Semi-minor axis
    num_angles = 100 # Number of scanning angles
    
    # Generate data
    x_original, y_original = get_ellipse_points(ellipse_a, ellipse_b, 300)
    theta = np.linspace(0, 2 * np.pi, num_angles, endpoint=False)
    rho = support_function_ellipse(theta, ellipse_a, ellipse_b)
    rho_prime = support_function_derivative_ellipse(theta, ellipse_a, ellipse_b)
    
    # Setup animation interface
    plt.style.use('seaborn-v0_8-whitegrid')
    # Optimize for PPT-friendly layout: more compact and clearer
    fig = plt.figure(figsize=(20, 11))
    
    # Create more compact grid layout
    gs = fig.add_gridspec(2, 4, height_ratios=[1, 1.2], width_ratios=[1, 1, 1, 1],
                          hspace=0.25, wspace=0.3)
    
    # === First row: Key steps of Olver's formula ===
    ax1 = fig.add_subplot(gs[0, 0])  # Support function ρ(θ)
    ax2 = fig.add_subplot(gs[0, 1])  # Derivative ρ'(θ)
    ax3 = fig.add_subplot(gs[0, 2])  # Polar coordinate visualization
    ax4 = fig.add_subplot(gs[0, 3])  # Current scanning state
    
    # === Second row: Reconstruction result comparison ===
    ax5 = fig.add_subplot(gs[1, :2])  # Tangent lines and envelope
    ax6 = fig.add_subplot(gs[1, 2:])  # Final reconstruction comparison
    
    max_val = max(ellipse_a, ellipse_b) * 1.2
    
    # Configure subplots - optimize font size and titles
    # ax1: Support function
    ax1.set_title("①Support Function ρ(θ)", fontsize=12, weight='bold')
    ax1.set_xlabel("θ (radians)", fontsize=10)
    ax1.set_ylabel("ρ(θ)", fontsize=10)
    ax1.plot(theta, rho, 'b-', alpha=0.3, linewidth=1)
    rho_line, = ax1.plot([], [], 'r-', linewidth=3)
    rho_point, = ax1.plot([], [], 'ro', markersize=8)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, 2*np.pi)
    ax1.set_ylim(min(rho)*0.9, max(rho)*1.1)
    
    # ax2: Derivative
    ax2.set_title("②Derivative ρ'(θ)", fontsize=12, weight='bold')
    ax2.set_xlabel("θ (radians)", fontsize=10)
    ax2.set_ylabel("ρ'(θ)", fontsize=10)
    ax2.plot(theta, rho_prime, 'g-', alpha=0.3, linewidth=1)
    rho_prime_line, = ax2.plot([], [], 'orange', linewidth=3)
    rho_prime_point, = ax2.plot([], [], 'o', color='orange', markersize=8)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, 2*np.pi)
    ax2.set_ylim(min(rho_prime)*1.1, max(rho_prime)*1.1)
    
    # ax3: Polar coordinates
    ax3.set_title("③Polar Coordinate Rays", fontsize=12, weight='bold')
    ax3.set_aspect('equal')
    ax3.plot(x_original, y_original, 'b--', alpha=0.5, linewidth=1)
    current_radius, = ax3.plot([], [], 'r-', linewidth=4, alpha=0.8)
    radius_point, = ax3.plot([], [], 'ro', markersize=10)
    ax3.set_xlim(-max_val, max_val)
    ax3.set_ylim(-max_val, max_val)
    ax3.grid(True, alpha=0.3)
    
    # ax4: Status display
    ax4.set_title("④Current State", fontsize=12, weight='bold')
    ax4.axis('off')
    status_text1 = ax4.text(0.05, 0.85, "", transform=ax4.transAxes, 
                           fontsize=11, bbox=dict(boxstyle="round,pad=0.4", facecolor="lightblue", alpha=0.8))
    status_text2 = ax4.text(0.05, 0.6, "", transform=ax4.transAxes, 
                           fontsize=11, bbox=dict(boxstyle="round,pad=0.4", facecolor="lightgreen", alpha=0.8))
    status_text3 = ax4.text(0.05, 0.35, "", transform=ax4.transAxes, 
                           fontsize=11, bbox=dict(boxstyle="round,pad=0.4", facecolor="lightyellow", alpha=0.8))
    
    # ax5: Tangent lines and envelope
    ax5.set_title("⑤Envelope Formation: Geometric Envelope of Tangent Family", fontsize=13, weight='bold')
    ax5.set_xlabel("X", fontsize=11)
    ax5.set_ylabel("Y", fontsize=11)
    ax5.plot(x_original, y_original, 'b--', linewidth=3, alpha=0.7, label='True Ellipse')
    tangent_lines_list = []
    current_tangent, = ax5.plot([], [], 'g-', linewidth=3, alpha=0.9, label='Current Tangent')
    envelope_points, = ax5.plot([], [], 'ro', markersize=4, alpha=0.8, label='Envelope Points')
    ax5.set_aspect('equal')
    ax5.grid(True, alpha=0.3)
    ax5.set_xlim(-max_val, max_val)
    ax5.set_ylim(-max_val, max_val)
    ax5.legend(fontsize=11)
    
    # Use ax6 for final result comparison
    ax6.set_title("⑥Olver Reconstruction Results & Error Analysis", fontsize=13, weight='bold')
    ax6.set_xlabel("X", fontsize=11)
    ax6.set_ylabel("Y", fontsize=11)
    ax6.plot(x_original, y_original, 'b--', linewidth=2, alpha=0.7, label='True Ellipse')
    rho_vector, = ax6.plot([], [], 'r-', linewidth=3, alpha=0.8, label='ρ Vector')
    rho_prime_vector, = ax6.plot([], [], 'orange', linewidth=3, alpha=0.8, label="ρ' Correction")
    result_point, = ax6.plot([], [], 'ko', markersize=8, label='Reconstructed Point')
    reconstruction_curve, = ax6.plot([], [], 'purple', linewidth=2, label='Reconstructed Curve')
    ax6.set_aspect('equal')
    ax6.grid(True, alpha=0.3)
    ax6.set_xlim(-max_val, max_val)
    ax6.set_ylim(-max_val, max_val)
    ax6.legend(fontsize=10)
    
    # Use ax6 for final result comparison
    ax6.set_title("Olver Reconstruction: Geometric Implementation & Result Comparison", fontsize=12)
    ax6.set_xlabel("X")
    ax6.set_ylabel("Y")
    ax6.plot(x_original, y_original, 'b--', linewidth=3, alpha=0.6, label='Original Ellipse')
    final_reconstruction, = ax6.plot([], [], 'r-', linewidth=2, label='Olver Reconstruction')
    error_text = ax6.text(0.02, 0.95, "", transform=ax6.transAxes, 
                         fontsize=10, verticalalignment='top',
                         bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    ax6.set_aspect('equal')
    ax6.grid(True, alpha=0.3)
    ax6.set_xlim(-max_val, max_val)
    ax6.set_ylim(-max_val, max_val)
    ax6.legend(fontsize=10)

    def animate(frame):
        current_points = min(frame + 1, num_angles)
        
        if current_points == 0:
            return []
        
        # Current data
        theta_current = theta[:current_points]
        rho_current = rho[:current_points]
        rho_prime_current = rho_prime[:current_points]
        current_theta = theta_current[-1] if len(theta_current) > 0 else 0
        current_rho = rho_current[-1] if len(rho_current) > 0 else 0
        current_rho_prime = rho_prime_current[-1] if len(rho_prime_current) > 0 else 0
        
        # Update first row plots
        # ax1: Support function
        rho_line.set_data(theta_current, rho_current)
        rho_point.set_data([current_theta], [current_rho])
        
        # ax2: Derivative
        rho_prime_line.set_data(theta_current, rho_prime_current)
        rho_prime_point.set_data([current_theta], [current_rho_prime])
        
        # ax3: Polar coordinate rays
        if current_points > 0:
            radius_x = [0, current_rho * np.cos(current_theta)]
            radius_y = [0, current_rho * np.sin(current_theta)]
            current_radius.set_data(radius_x, radius_y)
            radius_point.set_data([radius_x[1]], [radius_y[1]])
        
        # ax4: Status text
        progress = (current_points / num_angles) * 100
        status_text1.set_text(f"θ = {current_theta:.2f}")
        status_text2.set_text(f"ρ = {current_rho:.2f}")
        status_text3.set_text(f"ρ' = {current_rho_prime:.2f}")
        
        # Update second row plots
        # ax5: Tangent lines and envelope
        for line in tangent_lines_list:
            line.remove()
        tangent_lines_list.clear()
        
        if current_points > 2:
            # Show current tangent
            tangent_x, tangent_y, contact_x, contact_y = get_tangent_line(
                current_theta, current_rho, current_rho_prime, length=5)
            current_tangent.set_data(tangent_x, tangent_y)
            
            # Show tangent family
            step = max(1, current_points // 10)
            envelope_x, envelope_y = [], []
            for i in range(0, current_points, step):
                tx, ty, cx, cy = get_tangent_line(theta[i], rho[i], rho_prime[i], length=4)
                line_obj, = ax5.plot(tx, ty, 'g-', alpha=0.3, linewidth=0.8)
                tangent_lines_list.append(line_obj)
                envelope_x.append(cx)
                envelope_y.append(cy)
            
            envelope_points.set_data(envelope_x, envelope_y)
        
        # ax6: Olver formula geometric implementation
        if current_points > 0:
            # ρ vector
            rho_x = [0, current_rho * np.cos(current_theta)]
            rho_y = [0, current_rho * np.sin(current_theta)]
            rho_vector.set_data(rho_x, rho_y)
            
            # ρ' correction vector (perpendicular component)
            rho_prime_x = [rho_x[1], rho_x[1] + current_rho_prime * np.sin(current_theta)]
            rho_prime_y = [rho_y[1], rho_y[1] - current_rho_prime * np.cos(current_theta)]
            rho_prime_vector.set_data(rho_prime_x, rho_prime_y)
            
            # Final reconstructed point
            final_x = current_rho * np.cos(current_theta) - current_rho_prime * np.sin(current_theta)
            final_y = current_rho * np.sin(current_theta) + current_rho_prime * np.cos(current_theta)
            result_point.set_data([final_x], [final_y])
            
            # Reconstructed curve
            if current_points > 2:
                x_recon, y_recon = reconstruct_shape(rho_current, rho_prime_current, theta_current)
                reconstruction_curve.set_data(x_recon, y_recon)
        
        # Update final reconstruction result (display in ax6)
        if current_points > 5:
            x_final, y_final = reconstruct_shape(rho_current, rho_prime_current, theta_current)
            final_reconstruction.set_data(x_final, y_final)
            
            # Calculate error
            error = calculate_reconstruction_error(x_original, y_original, x_final, y_final)
            error_text.set_text(f"Reconstruction Progress: {progress:.1f}%\\n" + 
                               f"Scan Points: {current_points}/{num_angles}\\n" +
                               f"Average Error: {error:.4f}")
        
        # Set overall title - concise and clear for PPT
        fig.suptitle(f"Olver Ellipse Reconstruction Algorithm Demo | Progress: {progress:.1f}%", 
                    fontsize=18, weight='bold')
        
        return [rho_line, rho_point, rho_prime_line, rho_prime_point, current_radius, 
                radius_point, current_tangent, envelope_points, rho_vector, 
                rho_prime_vector, result_point,                reconstruction_curve, final_reconstruction] + tangent_lines_list

    # Create and run animation
    frames = num_angles + 40
    anim = FuncAnimation(fig, animate, frames=frames, interval=100, blit=False, repeat=True)
    
    plt.tight_layout()
    
    # Save as GIF - suitable for PPT insertion
    print("Saving animation as GIF format...")
    anim.save('ellipse_olver_reconstruction.gif', 
              writer='pillow', 
              fps=8,  # Moderate frame rate, file won't be too large
              dpi=80,  # Moderate resolution
              savefig_kwargs={'bbox_inches': 'tight', 'pad_inches': 0.1})
    print("✓ GIF saved as: ellipse_olver_reconstruction.gif")
    
    # Display animation
    plt.show()
