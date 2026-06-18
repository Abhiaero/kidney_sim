import torch
import torch.nn as nn
import numpy as np
import os

class PINN_Stokes(nn.Module):
    def __init__(self):
        super(PINN_Stokes, self).__init__()
        # Input: x, y coordinates
        # Output: u, v (velocities), p (pressure)
        self.net = nn.Sequential(
            nn.Linear(2, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, 3)
        )
        
    def forward(self, x):
        return self.net(x)

def gradients(outputs, inputs):
    return torch.autograd.grad(outputs, inputs, grad_outputs=torch.ones_like(outputs), create_graph=True)[0]

def train_pinn():
    print("Initializing Physics-Informed Neural Network (PINN) for 2D Stokes Flow...")
    
    # Check for CUDA
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on device: {device}")
    
    model = PINN_Stokes().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
    mask_path = os.path.join(PROJECT_ROOT, 'assets/mask.npy')
    
    if os.path.exists(mask_path):
        mask = np.load(mask_path)
    else:
        print(f"Mask not found at {mask_path}, using default blank grid.")
        mask = np.zeros((100, 100))
        
    ny, nx = mask.shape
    x = np.linspace(0, 2, nx)
    y = np.linspace(0, 1, ny)
    X, Y = np.meshgrid(x, y)
    
    # Prepare Collocation Points (Fluid Domain)
    fluid_mask = (mask == 0)
    X_fluid = X[fluid_mask].flatten()
    Y_fluid = Y[fluid_mask].flatten()
    
    # We will randomly sample 2000 points from the fluid domain to keep training fast
    n_colloc = min(2000, len(X_fluid))
    idx_colloc = np.random.choice(len(X_fluid), n_colloc, replace=False)
    
    pts_colloc = np.vstack([X_fluid[idx_colloc], Y_fluid[idx_colloc]]).T
    colloc_tensor = torch.tensor(pts_colloc, dtype=torch.float32, requires_grad=True).to(device)
    
    # Prepare Boundary Points (Solid Tissue & Walls)
    solid_mask = (mask == 1)
    # Add external boundary walls
    solid_mask[0, :] = True
    solid_mask[-1, :] = True
    
    X_solid = X[solid_mask].flatten()
    Y_solid = Y[solid_mask].flatten()
    
    n_boundary = min(500, len(X_solid))
    idx_boundary = np.random.choice(len(X_solid), n_boundary, replace=False)
    
    pts_boundary = np.vstack([X_solid[idx_boundary], Y_solid[idx_boundary]]).T
    boundary_tensor = torch.tensor(pts_boundary, dtype=torch.float32).to(device)
    
    # Fluid Properties
    rho = 1.0
    nu = 0.16 # Constant kinematic viscosity for PINN baseline
    
    epochs = 1000
    print(f"Training PINN with Physics Loss (Stokes Equations) for {epochs} epochs...")
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        # 1. Physics Loss (PDE Residuals) at Collocation Points
        # Forward pass requires grad
        preds_colloc = model(colloc_tensor)
        u_colloc = preds_colloc[:, 0:1]
        v_colloc = preds_colloc[:, 1:2]
        p_colloc = preds_colloc[:, 2:3]
        
        # First derivatives
        du = gradients(u_colloc, colloc_tensor)
        du_dx = du[:, 0:1]
        du_dy = du[:, 1:2]
        
        dv = gradients(v_colloc, colloc_tensor)
        dv_dx = dv[:, 0:1]
        dv_dy = dv[:, 1:2]
        
        dp = gradients(p_colloc, colloc_tensor)
        dp_dx = dp[:, 0:1]
        dp_dy = dp[:, 1:2]
        
        # Second derivatives
        d2u_dx2 = gradients(du_dx, colloc_tensor)[:, 0:1]
        d2u_dy2 = gradients(du_dy, colloc_tensor)[:, 1:2]
        
        d2v_dx2 = gradients(dv_dx, colloc_tensor)[:, 0:1]
        d2v_dy2 = gradients(dv_dy, colloc_tensor)[:, 1:2]
        
        # Stokes Equations Residuals (No convective term)
        f_u = nu * (d2u_dx2 + d2u_dy2) - (1/rho) * dp_dx
        f_v = nu * (d2v_dx2 + d2v_dy2) - (1/rho) * dp_dy
        
        # Continuity Equation Residual
        f_c = du_dx + dv_dy
        
        loss_pde = torch.mean(f_u**2) + torch.mean(f_v**2) + torch.mean(f_c**2)
        
        # 2. Boundary Condition Loss (No-slip at tissue walls)
        preds_boundary = model(boundary_tensor)
        u_bound = preds_boundary[:, 0:1]
        v_bound = preds_boundary[:, 1:2]
        
        loss_bc = torch.mean(u_bound**2) + torch.mean(v_bound**2)
        
        # 3. Total Loss
        loss = loss_pde + (10.0 * loss_bc) # Weight BC higher
        
        loss.backward()
        optimizer.step()
        
        if epoch % 100 == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch}/{epochs} - Total Loss: {loss.item():.6f} (PDE: {loss_pde.item():.6f}, BC: {loss_bc.item():.6f})")
            
    print("PINN surrogate training completed successfully. Real-time inference now possible.")
    
    assets_dir = os.path.join(PROJECT_ROOT, 'assets')
    os.makedirs(assets_dir, exist_ok=True)
    out_path = os.path.join(assets_dir, 'pinn_surrogate.pth')
    torch.save(model.state_dict(), out_path)
    print(f"Saved PyTorch PINN model to {out_path}")

if __name__ == "__main__":
    train_pinn()
