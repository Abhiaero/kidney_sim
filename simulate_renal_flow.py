import numpy as np
import os

def simulate_masked_flow():
    print("Starting 2D CFD simulation with Non-Newtonian Carreau-Yasuda Rheology...")
    
    mask_path = 'assets/mask.npy'
    if os.path.exists(mask_path):
        mask = np.load(mask_path)
        ny, nx = mask.shape
    else:
        print("Mask not found, using default 60x20 channel.")
        nx, ny = 60, 20
        mask = np.zeros((ny, nx))
        
    nt = 800
    nit = 50
    dx = 2.0 / (nx - 1)
    dy = 1.0 / (ny - 1)
    
    rho = 1.0
    dt = 0.001
    
    # Carreau-Yasuda parameters for Blood (normalized for this specific scale)
    mu_inf = 0.035   # Infinite shear viscosity
    mu_0 = 0.16      # Zero shear viscosity
    lam = 3.313      # Relaxation time
    a = 2.0          # Yasuda parameter
    n = 0.3568       # Power law index
    
    u = np.zeros((ny, nx))
    v = np.zeros((ny, nx))
    p = np.zeros((ny, nx))
    b = np.zeros((ny, nx))
    nu_field = np.ones((ny, nx)) * mu_0 / rho # Initial kinematic viscosity field
    
    def build_up_b(b, rho, dt, u, v, dx, dy):
        b[1:-1, 1:-1] = (rho * (1 / dt * 
                        ((u[1:-1, 2:] - u[1:-1, 0:-2]) / (2 * dx) + 
                         (v[2:, 1:-1] - v[0:-2, 1:-1]) / (2 * dy)) -
                        ((u[1:-1, 2:] - u[1:-1, 0:-2]) / (2 * dx))**2 -
                        2 * ((u[2:, 1:-1] - u[0:-2, 1:-1]) / (2 * dy) *
                             (v[1:-1, 2:] - v[1:-1, 0:-2]) / (2 * dx)) -
                        ((v[2:, 1:-1] - v[0:-2, 1:-1]) / (2 * dy))**2))
        return b

    def pressure_poisson(p, dx, dy, b, mask):
        pn = np.empty_like(p)
        for q in range(nit):
            pn = p.copy()
            p[1:-1, 1:-1] = (((pn[1:-1, 2:] + pn[1:-1, 0:-2]) * dy**2 + 
                              (pn[2:, 1:-1] + pn[0:-2, 1:-1]) * dx**2) /
                             (2 * (dx**2 + dy**2)) -
                             dx**2 * dy**2 / (2 * (dx**2 + dy**2)) * 
                             b[1:-1, 1:-1])
            p[:, -1] = 0
            p[:, 0] = p[:, 1] + 1.0
            p[0, :] = p[1, :]
            p[-1, :] = p[-2, :]
            p[mask == 1] = 0
        return p

    for n_step in range(nt):
        un = u.copy()
        vn = v.copy()
        
        # Calculate local shear rate (gamma_dot)
        du_dy = (un[2:, 1:-1] - un[0:-2, 1:-1]) / (2 * dy)
        dv_dx = (vn[1:-1, 2:] - vn[1:-1, 0:-2]) / (2 * dx)
        du_dx = (un[1:-1, 2:] - un[1:-1, 0:-2]) / (2 * dx)
        dv_dy = (vn[2:, 1:-1] - vn[0:-2, 1:-1]) / (2 * dy)
        
        # Strain rate magnitude (simplified 2D)
        gamma_dot = np.zeros_like(u)
        gamma_dot[1:-1, 1:-1] = np.sqrt(2*(du_dx**2 + dv_dy**2) + (du_dy + dv_dx)**2)
        
        # Apply Carreau-Yasuda model for dynamic viscosity
        mu_field = mu_inf + (mu_0 - mu_inf) * (1 + (lam * gamma_dot)**a)**((n - 1) / a)
        nu_field = mu_field / rho
        
        b = build_up_b(b, rho, dt, u, v, dx, dy)
        p = pressure_poisson(p, dx, dy, b, mask)
        
        # Update velocities using spatially variable nu_field
        nu_local = nu_field[1:-1, 1:-1]
        
        u[1:-1, 1:-1] = (un[1:-1, 1:-1] -
                         un[1:-1, 1:-1] * dt / dx * (un[1:-1, 1:-1] - un[1:-1, 0:-2]) -
                         vn[1:-1, 1:-1] * dt / dy * (un[1:-1, 1:-1] - un[0:-2, 1:-1]) -
                         dt / (2 * rho * dx) * (p[1:-1, 2:] - p[1:-1, 0:-2]) +
                         nu_local * (dt / dx**2 * (un[1:-1, 2:] - 2 * un[1:-1, 1:-1] + un[1:-1, 0:-2]) +
                                     dt / dy**2 * (un[2:, 1:-1] - 2 * un[1:-1, 1:-1] + un[0:-2, 1:-1])))
        
        v[1:-1, 1:-1] = (vn[1:-1, 1:-1] -
                         un[1:-1, 1:-1] * dt / dx * (vn[1:-1, 1:-1] - vn[1:-1, 0:-2]) -
                         vn[1:-1, 1:-1] * dt / dy * (vn[1:-1, 1:-1] - vn[0:-2, 1:-1]) -
                         dt / (2 * rho * dy) * (p[2:, 1:-1] - p[0:-2, 1:-1]) +
                         nu_local * (dt / dx**2 * (vn[1:-1, 2:] - 2 * vn[1:-1, 1:-1] + vn[1:-1, 0:-2]) +
                                     dt / dy**2 * (vn[2:, 1:-1] - 2 * vn[1:-1, 1:-1] + vn[0:-2, 1:-1])))
        
        u[0, :] = 0; u[-1, :] = 0
        v[0, :] = 0; v[-1, :] = 0
        u[:, -1] = u[:, -2]; v[:, -1] = 0
        u[:, 0] = u[:, 1]; v[:, 0] = 0
        
        # Apply mask no-slip condition
        u[mask == 1] = 0
        v[mask == 1] = 0

    # Calculate Wall Shear Stress dynamically based on local dynamic viscosity
    mu_local_top = mu_field[-2, :]
    mu_local_bottom = mu_field[1, :]
    wss_bottom = mu_local_bottom * np.abs((u[1, :] - u[0, :]) / dy)
    wss_top = mu_local_top * np.abs((u[-2, :] - u[-1, :]) / dy)
    
    os.makedirs('results', exist_ok=True)
    np.save('results/u.npy', u)
    np.save('results/v.npy', v)
    np.save('results/p.npy', p)
    np.save('results/nu.npy', nu_field) # Export viscosity field
    np.save('results/wss_top.npy', wss_top)
    np.save('results/wss_bottom.npy', wss_bottom)
    np.save('results/mask.npy', mask) 
    
    print("Non-Newtonian Simulation completed. Results saved to 'results' directory.")

if __name__ == '__main__':
    simulate_masked_flow()
