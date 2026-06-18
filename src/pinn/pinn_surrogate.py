import torch
import torch.nn as nn
import numpy as np
import os
import yaml
import matplotlib.pyplot as plt

class FourierFeatureNetwork(nn.Module):
    def __init__(self, in_features=2, out_features=3, n_layers=5, n_neurons=256, sigma=2.0):
        super(FourierFeatureNetwork, self).__init__()
        
        # Fourier Feature Mapping Matrix B
        self.B = nn.Parameter(torch.randn(in_features, n_neurons // 2) * sigma, requires_grad=False)
        
        layers = []
        # First layer after Fourier mapping expects n_neurons
        layers.append(nn.Linear(n_neurons, n_neurons))
        layers.append(nn.Tanh())
        
        for _ in range(n_layers - 2):
            layers.append(nn.Linear(n_neurons, n_neurons))
            layers.append(nn.Tanh())
            
        layers.append(nn.Linear(n_neurons, out_features))
        self.net = nn.Sequential(*layers)
        
    def forward(self, x):
        # Apply Fourier mapping: [cos(2pi B^T x), sin(2pi B^T x)]
        x_proj = 2.0 * np.pi * x @ self.B
        x_ff = torch.cat([torch.cos(x_proj), torch.sin(x_proj)], dim=-1)
        return self.net(x_ff)

def gradients(outputs, inputs):
    return torch.autograd.grad(outputs, inputs, grad_outputs=torch.ones_like(outputs), create_graph=True)[0]

def train_pinn():
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
    config_path = os.path.join(PROJECT_ROOT, 'config.yaml')
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        
    pinn_cfg = config['pinn']
    phys_cfg = config['physics']['fluid']
    
    print("Initializing Journal-Grade Fourier Feature PINN for 2D Stokes Flow...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = FourierFeatureNetwork(
        n_layers=pinn_cfg['architecture']['n_layers'],
        n_neurons=pinn_cfg['architecture']['n_neurons'],
        sigma=pinn_cfg['architecture']['fourier_sigma']
    ).to(device)
    
    adam_epochs = pinn_cfg['training']['adam_epochs']
    lbfgs_epochs = pinn_cfg['training']['lbfgs_epochs']
    lr = pinn_cfg['training']['lr']
    
    optimizer_adam = torch.optim.Adam(model.parameters(), lr=lr)
    optimizer_lbfgs = torch.optim.LBFGS(
        model.parameters(), 
        lr=1.0, 
        max_iter=20, 
        max_eval=25, 
        history_size=50,
        tolerance_grad=1e-5, 
        tolerance_change=1e-9, 
        line_search_fn="strong_wolfe"
    )
    
    mask_path = os.path.join(PROJECT_ROOT, 'assets/mask.npy')
    mask = np.load(mask_path) if os.path.exists(mask_path) else np.zeros((100, 100))
        
    ny, nx = mask.shape
    x = np.linspace(0, 2, nx)
    y = np.linspace(0, 1, ny)
    X, Y = np.meshgrid(x, y)
    
    # Collocation
    fluid_mask = (mask == 0)
    X_f = X[fluid_mask].flatten()
    Y_f = Y[fluid_mask].flatten()
    n_colloc = min(pinn_cfg['training']['n_colloc_points'], len(X_f))
    idx_colloc = np.random.choice(len(X_f), n_colloc, replace=False)
    pts_colloc = np.vstack([X_f[idx_colloc], Y_f[idx_colloc]]).T
    colloc_tensor = torch.tensor(pts_colloc, dtype=torch.float32, requires_grad=True).to(device)
    
    # Boundary
    solid_mask = (mask == 1)
    solid_mask[0, :] = True
    solid_mask[-1, :] = True
    X_b = X[solid_mask].flatten()
    Y_b = Y[solid_mask].flatten()
    n_bound = min(pinn_cfg['training']['n_boundary_points'], len(X_b))
    idx_bound = np.random.choice(len(X_b), n_bound, replace=False)
    pts_bound = np.vstack([X_b[idx_bound], Y_b[idx_bound]]).T
    bound_tensor = torch.tensor(pts_bound, dtype=torch.float32).to(device)
    
    rho = phys_cfg['rho']
    nu = phys_cfg['nu']
    
    def closure():
        optimizer_lbfgs.zero_grad()
        if optimizer_adam is not None:
             optimizer_adam.zero_grad()
             
        preds_colloc = model(colloc_tensor)
        u_c, v_c, p_c = preds_colloc[:, 0:1], preds_colloc[:, 1:2], preds_colloc[:, 2:3]
        
        du = gradients(u_c, colloc_tensor)
        dv = gradients(v_c, colloc_tensor)
        dp = gradients(p_c, colloc_tensor)
        
        d2u_dx2 = gradients(du[:, 0:1], colloc_tensor)[:, 0:1]
        d2u_dy2 = gradients(du[:, 1:2], colloc_tensor)[:, 1:2]
        d2v_dx2 = gradients(dv[:, 0:1], colloc_tensor)[:, 0:1]
        d2v_dy2 = gradients(dv[:, 1:2], colloc_tensor)[:, 1:2]
        
        f_u = nu * (d2u_dx2 + d2u_dy2) - (1/rho) * dp[:, 0:1]
        f_v = nu * (d2v_dx2 + d2v_dy2) - (1/rho) * dp[:, 1:2]
        f_c = du[:, 0:1] + dv[:, 1:2]
        
        loss_pde = torch.mean(f_u**2) + torch.mean(f_v**2) + torch.mean(f_c**2)
        
        preds_b = model(bound_tensor)
        loss_bc = torch.mean(preds_b[:, 0:1]**2) + torch.mean(preds_b[:, 1:2]**2)
        
        loss = loss_pde + 10.0 * loss_bc
        loss.backward()
        return loss

    print(f"Stage 1: Adam Optimizer ({adam_epochs} epochs)")
    for epoch in range(adam_epochs):
        loss = closure()
        optimizer_adam.step()
        if epoch % 100 == 0:
            print(f"Adam Epoch {epoch}: Loss = {loss.item():.6f}")
            
    print(f"Stage 2: L-BFGS Optimizer ({lbfgs_epochs} epochs)")
    optimizer_adam = None # disable Adam in closure
    for epoch in range(lbfgs_epochs):
        loss = optimizer_lbfgs.step(closure)
        if epoch % 20 == 0:
            print(f"L-BFGS Epoch {epoch}: Loss = {loss.item():.6f}")

    print("PINN Training Complete. Cross-Validating against CFD Ground Truth...")
    
    # Validation against CFD
    results_dir = os.path.join(PROJECT_ROOT, 'results')
    u_cfd_path = os.path.join(results_dir, 'u.npy')
    if os.path.exists(u_cfd_path):
        u_cfd = np.load(u_cfd_path)
        v_cfd = np.load(os.path.join(results_dir, 'v.npy'))
        
        # Eval PINN on full grid
        pts_all = np.vstack([X.flatten(), Y.flatten()]).T
        tensor_all = torch.tensor(pts_all, dtype=torch.float32).to(device)
        with torch.no_grad():
            preds_all = model(tensor_all).cpu().numpy()
            
        u_pinn = preds_all[:, 0].reshape((ny, nx))
        v_pinn = preds_all[:, 1].reshape((ny, nx))
        
        # Apply mask zeroing for fair comparison
        u_pinn[mask == 1] = 0
        v_pinn[mask == 1] = 0
        
        # Relative L2 Error
        u_error = np.linalg.norm(u_cfd - u_pinn) / (np.linalg.norm(u_cfd) + 1e-8)
        v_error = np.linalg.norm(v_cfd - v_pinn) / (np.linalg.norm(v_cfd) + 1e-8)
        
        print(f"Validation Results - U Velocity L2 Error: {u_error*100:.2f}%")
        print(f"Validation Results - V Velocity L2 Error: {v_error*100:.2f}%")
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        im1 = axes[0].contourf(X, Y, u_cfd, levels=50, cmap='jet')
        axes[0].set_title("CFD Ground Truth (U Velocity)")
        fig.colorbar(im1, ax=axes[0])
        
        im2 = axes[1].contourf(X, Y, u_pinn, levels=50, cmap='jet')
        axes[1].set_title("PINN Prediction (U Velocity)")
        fig.colorbar(im2, ax=axes[1])
        
        im3 = axes[2].contourf(X, Y, np.abs(u_cfd - u_pinn), levels=50, cmap='hot')
        axes[2].set_title(f"Absolute Error (Rel L2: {u_error*100:.2f}%)")
        fig.colorbar(im3, ax=axes[2])
        
        val_path = os.path.join(results_dir, 'pinn_vs_cfd_validation.png')
        plt.tight_layout()
        plt.savefig(val_path, dpi=300)
        print(f"Validation plot saved to {val_path}")

if __name__ == "__main__":
    train_pinn()
