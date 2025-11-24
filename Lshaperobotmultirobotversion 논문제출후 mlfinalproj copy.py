import torch
import numpy as np
import time
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Polygon
import random
import copy
import matplotlib.animation as animation

class BPSDF_2D_Model_With_Grad:
    """
    A 2D model using Bernstein Polynomials to represent a Signed Distance Field (SDF).
    This class can also compute the gradient of the SDF.
    (This class is unchanged from your original)
    """
    def __init__(self, n_func, domain_min, domain_max):
        self.n_func = n_func
        self.domain_min = torch.tensor(domain_min, dtype=torch.float32)
        self.domain_max = torch.tensor(domain_max, dtype=torch.float32)
        self.device = torch.device("cpu")

    def binomial_coefficient(self, n, k):
        """Calculates the binomial coefficient nCk."""
        # Use torch.lgamma for numerical stability
        return torch.exp(torch.lgamma(n + 1) - torch.lgamma(k + 1) - torch.lgamma(n - k + 1))

    def build_bernstein_t(self, t, use_derivative=False):
        """
        Builds the Bernstein basis functions and their derivatives.
        """
        t = torch.clamp(t, min=1e-6, max=1 - 1e-6)
        n = self.n_func - 1
        i = torch.arange(self.n_func, device=self.device)
        
        # Binomial coefficient
        comb = self.binomial_coefficient(torch.tensor(n, device=self.device, dtype=torch.float32), i)
        
        # Bernstein basis functions
        phi = comb * (1 - t).unsqueeze(-1) ** (n - i) * t.unsqueeze(-1) ** i

        if not use_derivative:
            return phi.float(), None
        else:
            # Derivative of Bernstein basis functions
            t_pow_i = t.unsqueeze(-1) ** i
            t_pow_i_minus_1 = t.unsqueeze(-1) ** torch.clamp(i-1, min=0)
            t_pow_i_minus_1[:, 0] = 0.0 # d/dt(t^0) = 0

            one_minus_t_pow = (1 - t).unsqueeze(-1) ** (n - i)
            one_minus_t_pow_minus_1 = (1 - t).unsqueeze(-1) ** torch.clamp(n - i - 1, min=0)
            one_minus_t_pow_minus_1[:, n] = 0.0 # d/dt((1-t)^0) = 0
            
            term1_deriv = i * t_pow_i_minus_1
            term2_deriv = -(n - i) * one_minus_t_pow_minus_1
            
            dphi = comb * ( (term1_deriv * one_minus_t_pow) + (t_pow_i * term2_deriv) )

            return phi.float(), dphi.float()

    def build_basis_function_from_points_2d(self, p):
        p = torch.atleast_2d(p).float()
        p_normalized = ((p - self.domain_min) / (self.domain_max - self.domain_min))
        phi_x, _ = self.build_bernstein_t(p_normalized.T[0], use_derivative=False)
        phi_y, _ = self.build_bernstein_t(p_normalized.T[1], use_derivative=False)
        # Kronecker product using einsum
        phi_xy = torch.einsum("ij,ik->ijk", phi_x, phi_y).reshape(p.shape[0], self.n_func**2)
        return phi_xy

    def build_gradient_basis_from_points_2d(self, p):
        p = torch.atleast_2d(p).float()
        domain_span = self.domain_max - self.domain_min
        p_normalized = ((p - self.domain_min) / (domain_span))
        
        phi_x, dphi_x = self.build_bernstein_t(p_normalized.T[0], use_derivative=True)
        phi_y, dphi_y = self.build_bernstein_t(p_normalized.T[1], use_derivative=True)
        
        grad_x_basis = torch.einsum("ij,ik->ijk", dphi_x, phi_y).reshape(p.shape[0], self.n_func**2) / domain_span[0]
        grad_y_basis = torch.einsum("ij,ik->ijk", phi_x, dphi_y).reshape(p.shape[0], self.n_func**2) / domain_span[1]
        
        grad_basis_stacked = torch.cat([grad_x_basis, grad_y_basis], dim=0)
        return grad_basis_stacked
        
    def predict(self, points, weights):
        with torch.no_grad():
            psi = self.build_basis_function_from_points_2d(points)
            return (psi @ weights.flatten()).numpy()

    def gradient(self, points, weights):
        with torch.no_grad():
            grad_psi = self.build_gradient_basis_from_points_2d(points)
            num_points = points.shape[0]
            weights_flat = weights.flatten()
            
            grad_x = grad_psi[:num_points] @ weights_flat
            grad_y = grad_psi[num_points:] @ weights_flat
            
            return np.vstack([grad_x, grad_y]).T

# --- Helper functions for scene and training ---

# --- NEW FUNCTION ---
def get_square_scene_sdf(points, half_size):
    """
    Calculates the exact SDF for an axis-aligned square centered at (0,0)
    with side length 2 * half_size.
    """
    points = np.atleast_2d(points)
    
    # Calculate component-wise distances
    # d.shape = (N, 2)
    d = np.abs(points) - half_size
    
    # Distance outside the box (Euclidean distance to corners/edges)
    # np.maximum(d, 0.0) is 0 inside the box, and the positive distance vector outside
    outside_dist = np.linalg.norm(np.maximum(d, 0.0), axis=1)
    
    # Distance inside the box (max of component distances)
    # np.maximum(d[:, 0], d[:, 1]) is the Chebyshev distance
    # np.minimum(..., 0.0) ensures it's 0 outside the box and negative inside
    inside_dist = np.minimum(np.maximum(d[:, 0], d[:, 1]), 0.0)
    
    # The SDF is the sum of these two components
    # (Only one of them will be non-zero for any given point)
    return outside_dist + inside_dist

# --- NEW FUNCTION ---
def generate_training_data_square(num_surface_points, num_random_points, square_half_size, domain_min, domain_max):
    """
    Generates training data specifically for the static square.
    """
    points, sdfs, grad_points, grad_normals = [], [], [], []
    N_edges = 4
    points_per_edge = max(1, num_surface_points // N_edges)

    # 1. Generate vertices
    hs = square_half_size
    vertices = np.array([
        [ hs,  hs],
        [-hs,  hs],
        [-hs, -hs],
        [ hs, -hs]
    ])
    
    all_surface_points = []
    all_surface_normals = []

    # 2. Generate surface points and normals
    # Edge 0: Top (y = hs)
    x_vals = np.linspace(hs, -hs, points_per_edge, endpoint=False)
    edge_0_points = np.vstack([x_vals, np.full_like(x_vals, hs)]).T
    edge_0_normals = np.tile([0.0, 1.0], (points_per_edge, 1))
    all_surface_points.append(edge_0_points)
    all_surface_normals.append(edge_0_normals)

    # Edge 1: Left (x = -hs)
    y_vals = np.linspace(hs, -hs, points_per_edge, endpoint=False)
    edge_1_points = np.vstack([np.full_like(y_vals, -hs), y_vals]).T
    edge_1_normals = np.tile([-1.0, 0.0], (points_per_edge, 1))
    all_surface_points.append(edge_1_points)
    all_surface_normals.append(edge_1_normals)
    
    # Edge 2: Bottom (y = -hs)
    x_vals = np.linspace(-hs, hs, points_per_edge, endpoint=False)
    edge_2_points = np.vstack([x_vals, np.full_like(x_vals, -hs)]).T
    edge_2_normals = np.tile([0.0, -1.0], (points_per_edge, 1))
    all_surface_points.append(edge_2_points)
    all_surface_normals.append(edge_2_normals)
    
    # Edge 3: Right (x = hs)
    y_vals = np.linspace(-hs, hs, points_per_edge, endpoint=False)
    edge_3_points = np.vstack([np.full_like(y_vals, hs), y_vals]).T
    edge_3_normals = np.tile([1.0, 0.0], (points_per_edge, 1))
    all_surface_points.append(edge_3_points)
    all_surface_normals.append(edge_3_normals)

    grad_points = np.vstack(all_surface_points)
    grad_normals = np.vstack(all_surface_normals)
    
    points.append(grad_points)
    sdfs.append(np.zeros(len(grad_points)))
    
    # 3. Generate random points in the domain
    random_points = np.random.uniform(low=domain_min, high=domain_max, size=(num_random_points, 2))
    points.append(random_points)
    # Use the new square SDF function
    sdfs.append(get_square_scene_sdf(random_points, square_half_size))
    
    all_points = np.vstack(points)
    all_sdfs = np.concatenate(sdfs)
    
    # 4. Shuffle
    shuffle_indices = np.random.permutation(len(all_points))
    
    return (torch.tensor(all_points[shuffle_indices], dtype=torch.float32),
            torch.tensor(all_sdfs[shuffle_indices], dtype=torch.float32),
            torch.tensor(grad_points, dtype=torch.float32),
            torch.tensor(grad_normals, dtype=torch.float32),
            vertices) # Also return vertices for plotting


def train_batch_with_gradient_weighted_PLAYER_FOCUS(
        model, points, sdfs, grad_points, grad_normals, 
        robot_pos_tensor,
        focus_weight_factor=5.0, 
        focus_zone_threshold=0.2,
        player_focus_radius=1.0):
    """
    Trains the BPSDF model using weighted least squares to focus on high-risk areas
    that are ALSO near the player's current position.
    (This function is unchanged from your original)
    """
    # 1. Build basis matrices for SDF values and gradients
    psi_sdf = model.build_basis_function_from_points_2d(points)
    psi_grad = model.build_gradient_basis_from_points_2d(grad_points)

    # 2. Combine into a single large basis matrix and target vector
    grad_weight = 0.5  # Overall importance of gradient matching
    psi_all = torch.cat([psi_sdf, psi_grad * grad_weight], dim=0)
    target_f = torch.cat([sdfs, grad_weight * grad_normals.T.flatten()], dim=0)

    # 3. --- NEW: Create a sample-wise weight vector ---
    sdf_weights = torch.ones_like(sdfs)
    
    # Mask 1: Points near the obstacle surface
    high_risk_mask = (sdfs >= 0) & (sdfs < focus_zone_threshold)
    
    # Mask 2: Points near the robot
    dist_to_player = torch.norm(points - robot_pos_tensor, dim=1)
    player_focus_mask = (dist_to_player < player_focus_radius)
    
    # Combined Mask: Apply high weight only to points in BOTH zones
    combined_focus_mask = high_risk_mask & player_focus_mask
    
    sdf_weights[combined_focus_mask] = focus_weight_factor
    
    # Gradient points get a standard weight
    grad_weights = torch.ones(psi_grad.shape[0])
    
    W_vec = torch.cat([sdf_weights, grad_weights]).to(model.device)

    # 4. Apply weights to implement weighted least squares
    W_sqrt = torch.sqrt(W_vec).unsqueeze(1)
    
    psi_all = psi_all.to(model.device)
    target_f = target_f.to(model.device)

    psi_weighted = W_sqrt * psi_all
    target_f_weighted = W_sqrt.squeeze() * target_f
    
    # 5. Solve the weighted least squares problem
    psi_t_weighted = psi_weighted.T
    lambda_reg = 1e-4
    I = torch.eye(model.n_func**2, device=model.device)
    
    A = psi_t_weighted @ psi_weighted + lambda_reg * I
    b = psi_t_weighted @ target_f_weighted
    
    # Use pseudoinverse for stability
    weights = torch.linalg.pinv(A) @ b
    
    return weights


if __name__ == '__main__':
    np.random.seed(42)
    torch.manual_seed(42)

    # --- ROBOT A (NAVIGATING AGENT) SETUP ---
    robot_radius = 0.3 # Just for visualization

    # --- OBSTACLE (STATIC SQUARE) SETUP ---
    print("--- Setting up Static Square Environment ---")
    # --- MODIFIED ---
    square_half_size = 1.2 # Half the side length of the square
    # ---
    N_b = 24 # Number of Bernstein basis functions (N_b x N_b grid)
    domain_size_b = 4.0
    domain_min_b = np.array([-domain_size_b, -domain_size_b])
    domain_max_b = np.array([domain_size_b, domain_size_b])
    model_b = BPSDF_2D_Model_With_Grad(n_func=N_b, domain_min=domain_min_b, domain_max=domain_max_b)
    
    # --- SIMULATION SETUP ---
    num_frames = 400
    dt = 0.05
    
    # --- Robot A Orbit Controller (Parametric) ---
    orbit_radius = 2.5       # Desired orbit radius
    orbit_angular_speed = 0.7  # Radians per second
    pos_a = np.array([orbit_radius, 0.0]) # Initial position
    
    # --- Player-Facing Loss Parameters ---
    player_focus_radius = 1.5  # How far from the robot to apply high weights
    focus_weight = 200.0       # How much to weight the focused area
    focus_threshold = 0.25     # How close to the surface to apply weights

    # --- PLOTTING SETUP ---
    plt.ion()
    # Create a single plot
    fig, ax1 = plt.subplots(1, 1, figsize=(9, 8))
    fig.canvas.manager.set_window_title("Visualization of Player-Focused Weighted SDF (Square)")
    fig.tight_layout()

    vis_res = 100
    x_vis, y_vis = np.meshgrid(np.linspace(domain_min_b[0], domain_max_b[0], vis_res),
                              np.linspace(domain_min_b[1], domain_max_b[1], vis_res))
    points_to_visualize = torch.tensor(np.vstack([x_vis.ravel(), y_vis.ravel()]).T, dtype=torch.float32)

    path_a_history = []
    time_steps = []

    print("--- Starting dynamic SDF learning visualization (LIVE) ---")
    fig.show()
    plt.pause(0.001)

    for frame in range(num_frames):
        start_time_frame = time.time()

        # 1) Generate training data for the static square
        # --- MODIFIED ---
        points_b, sdfs_b, grad_points_b, grad_normals_b, square_vertices = generate_training_data_square(
            80, 1500, square_half_size, domain_min_b, domain_max_b
        )
        # ---
        
        # 2) Re-learn obstacle SDF using the new PLAYER-FOCUSED weighted method
        weights_b = train_batch_with_gradient_weighted_PLAYER_FOCUS(
            model_b, points_b, sdfs_b, grad_points_b, grad_normals_b,
            robot_pos_tensor=torch.tensor(pos_a, dtype=torch.float32),
            focus_weight_factor=focus_weight,
            focus_zone_threshold=focus_threshold,
            player_focus_radius=player_focus_radius
        )

        # 3) Calculate Robot A's position (Simple parametric circle)
        current_time = frame * dt
        current_angle = current_time * orbit_angular_speed
        pos_a[0] = orbit_radius * np.cos(current_angle)
        pos_a[1] = orbit_radius * np.sin(current_angle)

        # 4) Record history
        path_a_history.append(np.copy(pos_a))
        time_steps.append(current_time)

        # 5) LIVE PLOTTING
        ax1.clear()

        # Plot the learned SDF
        z_b = model_b.predict(points_to_visualize, weights_b)
        ax1.contourf(x_vis, y_vis, z_b.reshape(vis_res, vis_res), levels=[-100, 0], colors=['lightcoral'], alpha=0.6)
        ax1.contour(x_vis, y_vis, z_b.reshape(vis_res, vis_res), levels=[0], colors='red', linewidths=2, label='Learned SDF (0-level)')
        
        # Plot the "true" square shape
        # --- MODIFIED ---
        ax1.add_patch(Polygon(square_vertices, closed=True, color='black', fill=True, alpha=0.7, label='True Obstacle'))
        # ---

        # Plot Robot A and its focus zone
        ax1.add_patch(Circle(pos_a, robot_radius, color='royalblue', alpha=0.9))
        ax1.add_patch(Circle(pos_a, player_focus_radius, color='red', fill=False, linestyle='--', lw=2, alpha=0.7, label='Player Focus Zone'))

        # Plot Robot A's path
        path_a_np = np.array(path_a_history)
        ax1.plot(path_a_np[:, 0], path_a_np[:, 1], '-', color='blue', lw=2)
        
        # Plot desired orbit
        ax1.add_patch(Circle((0,0), orbit_radius, color='green', fill=False, linestyle=':', lw=1.5, alpha=0.7, label='Robot Orbit'))

        ax1.set_title(f'Player-Focused Weighted SDF Learning - Frame {frame+1}')
        ax1.set_xlim(domain_min_b[0], domain_max_b[0])
        ax1.set_ylim(domain_min_b[1], domain_max_b[1])
        ax1.set_aspect('equal', adjustable='box')
        ax1.grid(True, linestyle=':', alpha=0.6)

        fig.tight_layout()
        fig.canvas.draw_idle()
        plt.pause(0.001)

        elapsed_time_frame = time.time() - start_time_frame
        if frame % 20 == 0:
            print(f"Frame {frame+1}/{num_frames} | Time/frame: {elapsed_time_frame:.4f}s")

    print("\n--- Simulation finished ---")
    plt.ioff()
    # Add a legend to the final plot
    handles, labels = ax1.get_legend_handles_labels()
    # Simple way to add patches to legend
    handles.append(plt.Rectangle((0,0),1,1, color='black', alpha=0.7))
    labels.append('True Obstacle')
    handles.append(plt.Line2D([0], [0], color='red', lw=2))
    labels.append('Learned SDF (0-level)')
    ax1.legend(handles, labels, loc='upper right', fontsize='small')
    plt.show()