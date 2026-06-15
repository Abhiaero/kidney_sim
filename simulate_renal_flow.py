import numpy as np
import os

def simulate_masked_flow():
    print("Starting 2D CFD simulation with real geometry masking...")
    
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
    nu = 0.1
    dt = 0.001
    
    u = np.zeros((ny, nx))
    v = np.zeros((ny, nx))
    p = np.zeros((ny, nx))
    b = np.zeros((ny, nx))
    
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
            
            # Apply solid mask to pressure
            p[mask == 1] = 0
            
        return p

    for n in range(nt):
        un = u.copy()
        vn = v.copy()
        
        b = build_up_b(b, rho, dt, u, v, dx, dy)
        p = pressure_poisson(p, dx, dy, b, mask)
        
        u[1:-1, 1:-1] = (un[1:-1, 1:-1] -
                         un[1:-1, 1:-1] * dt / dx * (un[1:-1, 1:-1] - un[1:-1, 0:-2]) -
                         vn[1:-1, 1:-1] * dt / dy * (un[1:-1, 1:-1] - un[0:-2, 1:-1]) -
                         dt / (2 * rho * dx) * (p[1:-1, 2:] - p[1:-1, 0:-2]) +
                         nu * (dt / dx**2 * (un[1:-1, 2:] - 2 * un[1:-1, 1:-1] + un[1:-1, 0:-2]) +
                               dt / dy**2 * (un[2:, 1:-1] - 2 * un[1:-1, 1:-1] + un[0:-2, 1:-1])))
        
        v[1:-1, 1:-1] = (vn[1:-1, 1:-1] -
                         un[1:-1, 1:-1] * dt / dx * (vn[1:-1, 1:-1] - vn[1:-1, 0:-2]) -
                         vn[1:-1, 1:-1] * dt / dy * (vn[1:-1, 1:-1] - vn[0:-2, 1:-1]) -
                         dt / (2 * rho * dy) * (p[2:, 1:-1] - p[0:-2, 1:-1]) +
                         nu * (dt / dx**2 * (vn[1:-1, 2:] - 2 * vn[1:-1, 1:-1] + vn[1:-1, 0:-2]) +
                               dt / dy**2 * (vn[2:, 1:-1] - 2 * vn[1:-1, 1:-1] + vn[0:-2, 1:-1])))
        
        u[0, :] = 0
        u[-1, :] = 0
        v[0, :] = 0
        v[-1, :] = 0
        
        u[:, -1] = u[:, -2]
        v[:, -1] = 0
        u[:, 0] = u[:, 1]
        v[:, 0] = 0
        
        # Apply mask no-slip condition
        u[mask == 1] = 0
        v[mask == 1] = 0

    mu = rho * nu
    wss_bottom = mu * np.abs((u[1, :] - u[0, :]) / dy)
    wss_top = mu * np.abs((u[-2, :] - u[-1, :]) / dy)
    
    os.makedirs('results', exist_ok=True)
    np.save('results/u.npy', u)
    np.save('results/v.npy', v)
    np.save('results/p.npy', p)
    np.save('results/wss_top.npy', wss_top)
    np.save('results/wss_bottom.npy', wss_bottom)
    np.save('results/mask.npy', mask) # Save mask for visualization
    
    print("Simulation completed. Results saved to 'results' directory.")

if __name__ == '__main__':
    simulate_masked_flow()
