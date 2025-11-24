# -----------------------------------------------------------------------------
# SPDX-License-Identifier: MIT
# This file is part of the RDF project.
# Copyright (c) 2023 Idiap Research Institute <contact@idiap.ch>
# Contributor: Yimming Li <yiming.li@idiap.ch>
# -----------------------------------------------------------------------------

import torch
import os
import numpy as np
import glob
import trimesh
import mesh_to_sdf
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import skimage.measure
import time
from panda_layer.panda_layer import PandaLayer

# Disable LaTeX to avoid potential rendering lag/errors
try:
    plt.rcParams['text.usetex'] = False
except:
    pass

CUR_DIR = os.path.dirname(os.path.abspath(__file__))

class BPSDF():
    def __init__(self, n_func, domain_min, domain_max, device):
        self.n_func = n_func
        self.domain_min = domain_min
        self.domain_max = domain_max
        self.device = device

    def binomial_coefficient(self, n, k):
        return torch.exp(torch.lgamma(n + 1) - torch.lgamma(k + 1) - torch.lgamma(n - k + 1))

    def build_bernstein_t(self, t):
        t = torch.clamp(t, min=1e-4, max=1-1e-4)
        n = self.n_func - 1
        i = torch.arange(self.n_func, device=self.device)
        comb = self.binomial_coefficient(torch.tensor(n, device=self.device), i)
        phi = comb * (1 - t).unsqueeze(-1) ** (n - i) * t.unsqueeze(-1) ** i
        return phi

    def build_basis_function_from_points(self, p):
        # p shape: (N, 3)
        N = len(p)
        # Normalize to [0, 1]
        p_norm = (p - self.domain_min) / (self.domain_max - self.domain_min)
        
        # Get 1D basis for X, Y, Z
        phi_x = self.build_bernstein_t(p_norm[:, 0])
        phi_y = self.build_bernstein_t(p_norm[:, 1])
        phi_z = self.build_bernstein_t(p_norm[:, 2])
        
        # 3D Tensor Product
        phi_xy = torch.einsum("ij,ik->ijk", phi_x, phi_y).reshape(-1, self.n_func**2)
        phi_xyz = torch.einsum("ij,ik->ijk", phi_xy, phi_z).reshape(-1, self.n_func**3)
        
        return phi_xyz

    def predict_grid_chunked(self, p, weights, chunk_size=5000):
        """ Predicts SDF values in chunks to avoid OOM """
        n_points = len(p)
        results = []
        with torch.no_grad():
            for i in range(0, n_points, chunk_size):
                chunk = p[i:i+chunk_size]
                phi = self.build_basis_function_from_points(chunk)
                res = torch.matmul(phi, weights)
                results.append(res)
        return torch.cat(results, dim=0)

    def solve_weighted_batch(self, points, sdfs, player_pos, focus_radius, focus_weight):
        """
        Weighted Least Squares: Minimize sum( w_i * (y_i - f(x_i))^2 )
        """
        # 1. Calculate Distance to Player
        d_player = torch.norm(points - player_pos, dim=1)
        
        # 2. Define Weights
        W_vec = torch.ones_like(sdfs)
        
        # Mask: Points near player AND near surface
        mask_near_player = d_player < focus_radius
        mask_near_surface = torch.abs(sdfs) < 0.2
        
        # Apply focus weight
        active_mask = mask_near_player & mask_near_surface
        W_vec[active_mask] = focus_weight
        
        # 3. Prepare Matrices
        Phi = self.build_basis_function_from_points(points)
        W_sqrt = torch.sqrt(W_vec).unsqueeze(1)
        
        Phi_w = Phi * W_sqrt
        y_w   = sdfs.unsqueeze(1) * W_sqrt
        
        # 4. Solve Normal Equations
        lambda_reg = 1e-4
        I = torch.eye(Phi.shape[1], device=self.device)
        
        A = torch.matmul(Phi_w.T, Phi_w) + lambda_reg * I
        b = torch.matmul(Phi_w.T, y_w)
        
        try:
            weights = torch.linalg.solve(A, b)
        except:
            weights = torch.linalg.lstsq(A, b).solution
            
        return weights.squeeze(), active_mask.sum().item()

# -----------------------------------------------------------------------------
# LIVE 3D SIMULATION CLASS
# -----------------------------------------------------------------------------
class LiveSimulation3D:
    def __init__(self, degree=8): # Increased default degree to 8 for better shape
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.bp_sdf = BPSDF(degree, -1.0, 1.0, self.device)
        
        print(f"--- Initializing Live 3D Simulation (Degree={degree}) ---")
        print("--- Loading Link 1 Mesh... ---")
        
        # 1. Load REAL Link 1 Data
        self.points, self.sdfs = self.load_link1_data()
        
        # 2. Simulation Params
        self.orbit_radius = 0.8
        self.angle = 0.0
        self.z_angle = 0.0
        self.focus_radius = 0.5
        self.focus_weight = 50.0 
        
        # 3. Grid for Marching Cubes
        # 32 is fast. 40 is better looking. 50 is slow.
        self.grid_res = 32 
        x = np.linspace(-0.9, 0.9, self.grid_res)
        y = np.linspace(-0.9, 0.9, self.grid_res)
        z = np.linspace(-0.9, 0.9, self.grid_res)
        grid_x, grid_y, grid_z = np.meshgrid(x, y, z, indexing='ij')
        
        self.grid_points_np = np.stack([grid_x.ravel(), grid_y.ravel(), grid_z.ravel()], axis=1)
        self.grid_points = torch.from_numpy(self.grid_points_np).float().to(self.device)
        
        # 4. Matplotlib Setup
        plt.ion()
        self.fig = plt.figure(figsize=(10, 8))
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.fig.canvas.manager.set_window_title("Live 3D Player-Focused SDF (Link 1)")

    def load_link1_data(self):
        """ Loads the actual Link 1 mesh and samples points. """
        # 1. Find file
        mesh_path = os.path.join(CUR_DIR, "panda_layer/meshes/visual/link1.stl")
        if not os.path.exists(mesh_path):
            mesh_path = glob.glob(os.path.join(CUR_DIR, "panda_layer/meshes/voxel_128/*link1*.stl"))
            if mesh_path:
                mesh_path = mesh_path[0]
            else:
                print("WARNING: Link 1 mesh not found. Using Dummy Sphere.")
                return self.generate_dummy_sphere()

        print(f"Found mesh: {os.path.basename(mesh_path)}")
        
        # 2. Load and Scale
        try:
            mesh = trimesh.load(mesh_path)
            if isinstance(mesh, trimesh.Scene): mesh = list(mesh.geometry.values())[0]
            
            # Scale to unit sphere
            offset = mesh.bounding_box.centroid
            # Safe scale calculation
            scale = np.max(np.linalg.norm(mesh.vertices - offset, axis=1))
            # Manual scaling to ensure it fits in [-1, 1]
            mesh.vertices = (mesh.vertices - offset) / scale
            
            # 3. Sample Points
            # 15k surface points + 5k random points
            p_surf, s_surf = mesh_to_sdf.sample_sdf_near_surface(mesh, number_of_points=15000)
            p_rand = np.random.rand(5000, 3) * 2.0 - 1.0
            s_rand = mesh_to_sdf.mesh_to_sdf(mesh, p_rand)
            
            p_all = np.concatenate([p_surf, p_rand])
            s_all = np.concatenate([s_surf, s_rand])
            
            return torch.from_numpy(p_all).float().to(self.device), torch.from_numpy(s_all).float().to(self.device)
            
        except Exception as e:
            print(f"Error loading mesh: {e}. Using Dummy Sphere.")
            return self.generate_dummy_sphere()

    def generate_dummy_sphere(self):
        n_points = 20000
        p = np.random.rand(n_points, 3) * 2.0 - 1.0
        d = np.linalg.norm(p, axis=1) - 0.5
        # Add noise to make the optimization work harder
        d += np.random.normal(0, 0.02, size=d.shape) 
        return torch.from_numpy(p).float().to(self.device), torch.from_numpy(d).float().to(self.device)

    def run(self):
        print("Starting 3D Loop... (Keep focus on the Plot window)")
        frame = 0
        
        while True:
            t_start = time.time()
            
            # 1. Update Player Position (3D Orbit)
            self.angle += 0.08
            self.z_angle += 0.03
            px = np.cos(self.angle) * self.orbit_radius
            py = np.sin(self.angle) * self.orbit_radius
            pz = np.sin(self.z_angle) * 0.4
            player_pos = torch.tensor([px, py, pz]).float().to(self.device)
            
            # 2. Solve Weighted SDF
            weights, active_pts = self.bp_sdf.solve_weighted_batch(
                self.points, self.sdfs, 
                player_pos, 
                self.focus_radius, 
                self.focus_weight
            )
            
            # 3. Generate 3D Mesh (Marching Cubes)
            sdf_grid = self.bp_sdf.predict_grid_chunked(self.grid_points, weights)
            sdf_grid_np = sdf_grid.cpu().numpy().reshape(self.grid_res, self.grid_res, self.grid_res)
            
            try:
                # Use spacing to match the domain size (approx 1.8 from -0.9 to 0.9)
                verts, faces, _, _ = skimage.measure.marching_cubes(
                    sdf_grid_np, level=0.0, 
                    spacing=(1.8/(self.grid_res-1), 1.8/(self.grid_res-1), 1.8/(self.grid_res-1))
                )
                # Centering correction
                verts -= [0.9, 0.9, 0.9] 
            except RuntimeError:
                verts, faces = [], []

            # 4. Visualize
            self.ax.clear()
            
            # A. Plot The Mesh
            if len(verts) > 0:
                self.ax.plot_trisurf(
                    verts[:, 0], verts[:, 1], verts[:, 2], 
                    triangles=faces, 
                    cmap='viridis', alpha=0.9, linewidth=0.1, edgecolor='k'
                )

            # B. Plot Player (Red Sphere)
            self.ax.scatter([px], [py], [pz], color='red', s=200, label='Player')
            
            # C. Plot Focus Zone (Wireframe approximation)
            u, v = np.mgrid[0:2*np.pi:10j, 0:np.pi:10j]
            sx = px + self.focus_radius * np.cos(u) * np.sin(v)
            sy = py + self.focus_radius * np.sin(u) * np.sin(v)
            sz = pz + self.focus_radius * np.cos(v)
            self.ax.plot_wireframe(sx, sy, sz, color='red', alpha=0.1)

            # D. Formatting
            self.ax.set_xlim(-1, 1)
            self.ax.set_ylim(-1, 1)
            self.ax.set_zlim(-1, 1)
            self.ax.set_title(f"Live 3D SDF (Link 1, Degree={self.bp_sdf.n_func})\nActive Points: {active_pts}")
            
            plt.draw()
            plt.pause(0.001)
            
            t_end = time.time()
            # print(f"Frame {frame} | FPS: {1.0/(t_end-t_start):.2f}")
            frame += 1

if __name__ == '__main__':
    # Increased degree to 8 for better Link 1 representation
    sim = LiveSimulation3D(degree=8)
    sim.run()

