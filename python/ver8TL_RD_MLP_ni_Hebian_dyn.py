import torch
import math
import matplotlib.pyplot as plt
import numpy as np

# Set up device and data type
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float64

# -----------------------------
# Plant Model: Two-Link Manipulator
# -----------------------------
class TwoLinkManipulator:
    def __init__(self, params, dt, device, dtype):
        """
        params: dictionary including m1, m2, a1, a2, g, c1, c2.
        dt: time step.
        """
        self.params = params
        self.dt = dt
        self.device = device
        self.dtype = dtype

    def build_system_matrices(self, q, q_dot):
        p = self.params
        m1 = p.get('m1', 0.0)
        m2 = p.get('m2', 0.0)
        a1 = p['a1']
        a2 = p['a2']
        g  = p['g']
        c1 = p['c1']
        c2 = p['c2']
        
        theta1 = q[0]
        theta2 = q[1]
        dtheta1 = q_dot[0]
        dtheta2 = q_dot[1]
        
        M11 = (m1 + m2)*a1**2 + m2*a2**2 + 2*m2*a1*a2*torch.cos(theta2)
        M12 = m2*a2**2 + m2*a1*a2*torch.cos(theta2)
        M21 = M12
        M22 = m2*a2**2
        M_q = torch.tensor([[M11, M12],
                             [M21, M22]], dtype=q.dtype, device=q.device)
        
        V11 = c1 - m2*a1*a2*torch.sin(theta2)*(2*dtheta2)
        V12 = -m2*a1*a2*torch.sin(theta2)*dtheta2
        V21 = m2*a1*a2*torch.sin(theta2)*dtheta1
        V22 = c2
        V_q_qdot = torch.tensor([[V11, V12],
                                  [V21, V22]], dtype=q.dtype, device=q.device)
        
        G1 = (m1 + m2)*g*a1*torch.cos(theta1) + m2*g*a2*torch.cos(theta1 + theta2)
        G2 = m2*g*a2*torch.cos(theta1 + theta2)
        G_q = torch.tensor([G1, G2], dtype=q.dtype, device=q.device)
        
        return M_q, V_q_qdot, G_q

    def step(self, x, u):
        q = x[0:2]
        q_dot = x[2:4]
        M_q, V_q_qdot, G_q = self.build_system_matrices(q, q_dot)
        q_ddot = torch.linalg.solve(M_q, (u - V_q_qdot @ q_dot - G_q))
        q_dot_new = q_dot + self.dt * q_ddot
        q_new = q + self.dt * q_dot_new
        return torch.cat((q_new, q_dot_new))

class MLPControllerNiIdealHebianDyn:
    def __init__(self, dt, device, dtype, H=50, K=2, I=14, initial_w=None, initial_v=None):
        """
        dt: time step for controller weight update.
        device, dtype: torch device and data type.
        H: number of hidden neurons (set to 5 to match MATLAB).
        K: output dimension.
        I: input dimension for the MLP (here, 14).
        initial_w: initial outer weight matrix of shape (H+1, K).
        initial_v: initial inner weight matrix of shape (I, H+1).
        """
        self.dt = dt
        self.device = device
        self.dtype = dtype
        # Controller gains as in MATLAB code:
        self.Kv = 5 * torch.eye(K, dtype=dtype, device=device)    # Kv = 5*I₂
        self.lam = 1 * torch.eye(2, dtype=dtype, device=device)     # lam = 1*I₂
        self.F_rate = 100.0   # F = 10
        self.G_rate = 100.0   # G = 10
        self.kappa = 0.01    # kappa = 0.01
        self.H = H           # number of hidden neurons
        self.K = K           # output dimension
        self.I = I           # input dimension (14)
        # Outer weight matrix w: shape (H+1, K)
        if initial_w is None:
            self.w = torch.zeros((H+1, K), dtype=dtype, device=device)
        else:
            self.w = initial_w.clone()
        # Inner weight matrix v: shape (I, H+1) 
        # (MATLAB code uses v(:,h) for h=1...H and then later loops for h=1 to H+1)
        if initial_v is None:
            self.v = torch.zeros((I, H+1), dtype=dtype, device=device)
        else:
            self.v = initial_v.clone()
        self.w_history = [self.w.clone()]
        self.v_history = [self.v.clone()]

    def step(self, x_des, x):
        """
        Compute control input using the nonideal Hebbian FLNN controller (slow, loop‐based version).
        
        Inputs:
            x_des: desired state vector (6 elements) [qd; qdp; qdpp]
            x: current state vector (4 elements) [q; q_dot]
        
        Returns:
            u: control input vector (2 elements)
            L: Lyapunov candidate (scalar)
            weight_dot: outer weight update (matrix of shape (H+1, K))
            veight_dot: inner weight update (matrix of shape (I, H+1))
            tau_out: network output (tau) before adding feedback.
        Also updates the internal weight matrices.
        """
        # Arm parameters and controller gains
        g = 9.81
        Kv = 5 * torch.eye(2, dtype=self.dtype, device=self.device)
        lam = 1 * torch.eye(2, dtype=self.dtype, device=self.device)
        
        # Tracking errors
        qd = x_des[0:2]
        qdp = x_des[2:4]
        qdpp = x_des[4:6]
        q = x[0:2]
        q_dot = x[2:4]
        e = qd - q
        ep = qdp - q_dot
        r = ep + torch.matmul(lam, e)  # r is a 2-element tensor
        
        # Construct x_new: [x, x_des, e, ep] → 14 elements
        x_new = torch.cat([x, x_des, e, ep])
        
        # Use local copies for clarity
        w = self.w  # shape (H+1, K)
        v = self.v  # shape (I, H+1)
        
        # Compute combined Frobenius norm: Z = [w, zeros(H+1, H+1); zeros(I, K), v]
        Z_top = torch.cat([w, torch.zeros((self.H+1, self.H+1), dtype=self.dtype, device=self.device)], dim=1)
        Z_bottom = torch.cat([torch.zeros((self.I, self.K), dtype=self.dtype, device=self.device), v], dim=1)
        Z = torch.cat([Z_top, Z_bottom], dim=0)
        Z_f_norm = torch.norm(Z, p='fro')
        Zb = 1.0
        Kz = torch.eye(2, dtype=self.dtype, device=self.device)
        v_t = - torch.matmul(Kz, (Z_f_norm + Zb) * r)  # shape (2,)
        
        # Hidden layer computation using explicit loops
        H = self.H
        K = self.K
        I = self.I
        sigma_x = [0.0] * H
        sigma_x_dot = [0.0] * H
        for h in range(H):
            in_out = 0.0
            for i in range(len(x_new)):
                in_out += v[i, h].item() * x_new[i].item()
            sigma_x[h] = 1.0 / (1.0 + math.exp(-in_out))
            sigma_x_dot[h] = sigma_x[h] * (1.0 - sigma_x[h])
        sigma_x = torch.tensor(sigma_x, dtype=self.dtype, device=self.device)
        sigma_x_dot = torch.tensor(sigma_x_dot, dtype=self.dtype, device=self.device)
        
        # Augmented hidden output: sigma_x_new = [sigma_x; 1]
        sigma_x_new = torch.cat([sigma_x, torch.tensor([1.0], dtype=self.dtype, device=self.device)])
        # Similarly, sigma_x_dot_new = [sigma_x_dot; 0]
        sigma_x_dot_new = torch.cat([sigma_x_dot, torch.tensor([0.0], dtype=self.dtype, device=self.device)])
        
        # Output function: compute tau
        tau = torch.zeros(K, dtype=self.dtype, device=self.device)
        for k in range(K):
            cont_in = 0.0
            for h in range(H+1):
                cont_in += w[h, k].item() * sigma_x_new[h].item()
            tau[k] = cont_in
        
        # Outer weight update (w_dot)
        F = 10.0
        w_dot = torch.zeros_like(w)
        norm_r = torch.norm(r).item()
        for k in range(K):
            for h in range(H+1):
                w_dot[h, k] = F * sigma_x_new[h] * r[k] - F * self.kappa * norm_r * w[h, k]
        
        # Inner weight update (v_dot)
        G = 10.0
        v_dot = torch.zeros_like(v)
        # Loop for h = 0 to H+1 (MATLAB: 1 to H+1)
        for h in range(H+1):
            for i in range(I):
                # For each h (including bias row)
                v_dot[i, h] = G * norm_r * x_new[i] * sigma_x_new[h] - G * self.kappa * norm_r * v[i, h]
        
        weight_dot = w_dot
        veight_dot = v_dot
        
        # Compute control input: u = tau + Kv*r - v_t, saturate between -10 and 10
        u = tau + torch.matmul(Kv, r) - v_t
        u = torch.clamp(u, -10, 10)
        
        tau_out = tau.clone()
        L = 0.5 * torch.dot(r, r)
        
        # Update weights using Euler integration
        self.w = self.w + self.dt * w_dot
        self.v = self.v + self.dt * v_dot
        self.w_history.append(self.w.clone())
        self.v_history.append(self.v.clone())
        
        return u, L, weight_dot, veight_dot, tau_out

    def stepFast(self, x_des, x):
        """
        Compute control input using the nonideal Hebbian FLNN controller (vectorized fast version).
        
        Inputs:
            x_des: desired state vector (6 elements) [qd; qdp; qdpp]
            x: current state vector (4 elements) [q; q_dot]
            
        Returns:
            u: control input vector (2 elements)
            L: Lyapunov candidate (scalar)
            w_dot: outer weight update (matrix of shape (H+1, K))
            v_dot: inner weight update (matrix of shape (I, H+1))
            tau_out: network output (tau) before adding feedback.
        Also updates the internal weight matrices.
        """
        # Extract states
        qd = x_des[0:2]
        qdp = x_des[2:4]
        qdpp = x_des[4:6]
        q = x[0:2]
        q_dot = x[2:4]
        
        # Tracking errors and filtered error
        e = qd - q
        ep = qdp - q_dot
        r = ep + torch.matmul(self.lam, e)  # shape (2,)
        
        # Construct x_new: [x, x_des, e, ep] -> shape (14,)
        x_new = torch.cat([x, x_des, e, ep])
        
        # Local weight matrices
        w = self.w  # shape (H+1, K)
        v = self.v  # shape (I, H+1)
        
        # Compute combined Frobenius norm:
        zeros_top = torch.zeros((self.H+1, self.H+1), dtype=self.dtype, device=self.device)
        zeros_bottom = torch.zeros((self.I, self.K), dtype=self.dtype, device=self.device)
        Z_top = torch.cat([w, zeros_top], dim=1)
        Z_bottom = torch.cat([zeros_bottom, v], dim=1)
        Z = torch.cat([Z_top, Z_bottom], dim=0)
        Z_f_norm = torch.norm(Z, p='fro')
        Zb = 1.0
        v_t = -(Z_f_norm + Zb) * r  # shape (2,)
        
        # Hidden layer: vectorized computation of in_out for each hidden neuron
        # Use only the first H+1 columns? We want to compute for h = 0,...,H-1.
        in_out = torch.matmul(x_new, self.v[:, :self.H])  # shape (H,)
        sigma_x = 1.0 / (1.0 + torch.exp(-in_out))
        sigma_x_dot = sigma_x * (1 - sigma_x)
        
        # Augment hidden output and derivative
        sigma_x_new = torch.cat([sigma_x, torch.tensor([1.0], dtype=self.dtype, device=self.device)])  # shape (H+1,)
        sigma_x_dot_new = torch.cat([sigma_x_dot, torch.tensor([0.0], dtype=self.dtype, device=self.device)])  # shape (H+1,)
        
        # Compute network output: tau = sigma_x_new^T * w, shape (K,)
        tau = torch.matmul(sigma_x_new, w)
        
        # Outer weight update: vectorized outer product minus decay
        norm_r = torch.norm(r)
        w_dot = self.F_rate * (sigma_x_new.unsqueeze(1) * r.unsqueeze(0)) - self.kappa * self.F_rate * norm_r * w
        
        # Inner weight update: update for hidden neurons (all columns) vectorized
        v_dot = self.G_rate * norm_r * (x_new.unsqueeze(1) * sigma_x_new[:self.H].unsqueeze(0)) - self.kappa * self.G_rate * norm_r * v[:, :self.H]
        # For the bias column in v (last column), you can update similarly if desired.
        # Here we leave it as is (or you can set it to zero).
        
        # Compute control input: u = tau + Kv*r - v_t, saturated between -10 and 10.
        u = tau + torch.matmul(self.Kv, r) - v_t
        u = torch.clamp(u, -10, 10)
        
        tau_out = tau.clone()
        L = 0.5 * torch.dot(r, r)
        
        # Update weights using Euler integration
        self.w = self.w + self.dt * w_dot
        # For inner weights, update all columns (we update the first H columns as computed)
        self.v[:, :self.H] = self.v[:, :self.H] + self.dt * v_dot
        
        self.w_history.append(self.w.clone())
        self.v_history.append(self.v.clone())
        
        return u, L, w_dot, v_dot, tau_out
# -----------------------------
# Complete Simulation Code
# -----------------------------
def main():
    dt = 0.0001  # simulation time step
    tFinal = 10.0
    timeVec = torch.arange(0, tFinal + dt, dt, dtype=dtype, device=device)
    numSteps = len(timeVec)

    known_params = {'a1': 0.2, 'a2': 0.2, 'g': 9.81, 'c1': 0.0, 'c2': 0.0}
    true_params = {'m1': 0.5, 'm2': 0.5}
    plant_params = dict(known_params)
    plant_params['m1'] = true_params['m1']
    plant_params['m2'] = true_params['m2']

    plant = TwoLinkManipulator(plant_params, dt, device, dtype)
    
    # Create controller instance using the nonideal Hebian method
    controller = MLPControllerNiIdealHebianDyn(dt, device, dtype)
    
    # Precompute reference trajectories (vectorized)
    theta1_ref = 0.3 + 0.2 * torch.sin(2 * math.pi * 0.5 * timeVec)
    theta1_ref_dot = 0.2 * (2 * math.pi * 0.5) * torch.cos(2 * math.pi * 0.5 * timeVec)
    theta1_ref_ddot = -0.2 * ((2 * math.pi * 0.5)**2) * torch.sin(2 * math.pi * 0.5 * timeVec)
    theta2_ref = 0.5 + 0.1 * torch.sin(2 * math.pi * 0.2 * timeVec)
    theta2_ref_dot = 0.1 * (2 * math.pi * 0.2) * torch.cos(2 * math.pi * 0.2 * timeVec)
    theta2_ref_ddot = -0.1 * ((2 * math.pi * 0.2)**2) * torch.sin(2 * math.pi * 0.2 * timeVec)

    X_true = torch.zeros((numSteps, 4), dtype=dtype, device=device)
    u_store = torch.zeros((numSteps, 2), dtype=dtype, device=device)
    L_store = torch.zeros(numSteps, dtype=dtype, device=device)
    
    X0 = torch.tensor([0.0, 0.2, 0.0, 0.0], dtype=dtype, device=device)
    X_true[0, :] = X0

    for k in range(numSteps - 1):
        x = X_true[k, :]
        print(k)
        qd = torch.tensor([theta1_ref[k], theta2_ref[k]], dtype=dtype, device=device)
        qdp = torch.tensor([theta1_ref_dot[k], theta2_ref_dot[k]], dtype=dtype, device=device)
        qdpp = torch.tensor([theta1_ref_ddot[k], theta2_ref_ddot[k]], dtype=dtype, device=device)
        x_des = torch.cat([qd, qdp, qdpp])
        
        u, L_val, w_dot, v_dot, tau_out = controller.stepFast(x_des, x)
        u_store[k, :] = u
        L_store[k] = L_val
        X_true[k+1, :] = plant.step(x, u)
    
    timeVec_np = timeVec.cpu().numpy()
    X_true_np = X_true.cpu().numpy()
    u_np = u_store.cpu().numpy()
    L_np = L_store.cpu().numpy()
    w_history_np = torch.stack(controller.w_history).cpu().numpy()
    v_history_np = torch.stack(controller.v_history).cpu().numpy()
    theta1_ref_np = theta1_ref.cpu().numpy()
    theta2_ref_np = theta2_ref.cpu().numpy()

    # Plot outer weight evolution
    plt.figure()
    numW = w_history_np.shape[0]
    time_control = timeVec_np[:numW]
    for i in range(w_history_np.shape[1]):
        for j in range(w_history_np.shape[2]):
            plt.plot(time_control, w_history_np[:, i, j])
    plt.xlabel('Time (s)')
    plt.ylabel('Outer Weight Value')
    plt.title('Evolution of Outer Weight Coefficients')
    plt.grid(True)
    plt.show()

    # Plot inner weight evolution
    plt.figure()
    numV = v_history_np.shape[0]
    time_control_v = timeVec_np[:numV]
    for i in range(v_history_np.shape[1]):
        for j in range(v_history_np.shape[2]):
            plt.plot(time_control_v, v_history_np[:, i, j])
    plt.xlabel('Time (s)')
    plt.ylabel('Inner Weight Value')
    plt.title('Evolution of Inner Weight Coefficients')
    plt.grid(True)
    plt.show()

    # Plot joint angles (actual vs. reference)
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 1, 1)
    plt.plot(timeVec_np, X_true_np[:, 0], 'b', label='Actual theta1')
    plt.plot(timeVec_np, theta1_ref_np, 'm-.', label='Ref theta1')
    plt.xlabel('Time (s)')
    plt.ylabel('Theta1 (rad)')
    plt.legend()
    plt.grid(True)
    plt.subplot(2, 1, 2)
    plt.plot(timeVec_np, X_true_np[:, 1], 'b', label='Actual theta2')
    plt.plot(timeVec_np, theta2_ref_np, 'm-.', label='Ref theta2')
    plt.xlabel('Time (s)')
    plt.ylabel('Theta2 (rad)')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Plot control inputs
    plt.figure(figsize=(12, 6))
    plt.subplot(2, 1, 1)
    plt.plot(timeVec_np, u_np[:, 0], 'k', label='u1')
    plt.xlabel('Time (s)')
    plt.ylabel('Torque 1')
    plt.legend()
    plt.grid(True)
    plt.subplot(2, 1, 2)
    plt.plot(timeVec_np, u_np[:, 1], 'k', label='u2')
    plt.xlabel('Time (s)')
    plt.ylabel('Torque 2')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Plot Lyapunov candidate over time
    plt.figure()
    plt.plot(timeVec_np, L_np, 'r', label='Lyapunov Candidate L')
    plt.xlabel('Time (s)')
    plt.ylabel('L')
    plt.title('Lyapunov Candidate over Time')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == '__main__':
    main()
