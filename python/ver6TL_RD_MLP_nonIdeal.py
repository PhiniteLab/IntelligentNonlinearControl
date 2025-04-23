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
        
        # Inertia matrix
        M11 = (m1 + m2)*a1**2 + m2*a2**2 + 2*m2*a1*a2*torch.cos(theta2)
        M12 = m2*a2**2 + m2*a1*a2*torch.cos(theta2)
        M21 = M12
        M22 = m2*a2**2
        M_q = torch.tensor([[M11, M12],
                             [M21, M22]], dtype=q.dtype, device=q.device)
        
        # Coriolis/centrifugal matrix (with damping)
        V11 = c1 - m2*a1*a2*torch.sin(theta2)*(2*dtheta2)
        V12 = -m2*a1*a2*torch.sin(theta2)*dtheta2
        V21 = m2*a1*a2*torch.sin(theta2)*dtheta1
        V22 = c2
        V_q_qdot = torch.tensor([[V11, V12],
                                  [V21, V22]], dtype=q.dtype, device=q.device)
        
        # Gravity vector
        G1 = (m1 + m2)*g*a1*torch.cos(theta1) + m2*g*a2*torch.cos(theta1 + theta2)
        G2 = m2*g*a2*torch.cos(theta1 + theta2)
        G_q = torch.tensor([G1, G2], dtype=q.dtype, device=q.device)
        
        return M_q, V_q_qdot, G_q

    def step(self, x, u):
        """
        Propagate the state one time step using forward Euler integration.
        x: state vector [theta1, theta2, dtheta1, dtheta2]
        u: control torque vector [tau1, tau2]
        """
        q = x[0:2]
        q_dot = x[2:4]
        M_q, V_q_qdot, G_q = self.build_system_matrices(q, q_dot)
        q_ddot = torch.linalg.solve(M_q, (u - V_q_qdot @ q_dot - G_q))
        q_dot_new = q_dot + self.dt * q_ddot
        q_new = q + self.dt * q_dot_new
        x_next = torch.cat((q_new, q_dot_new))
        return x_next

# -----------------------------
#  (MLP Non Ideal Case, MATLAB Version)
# -----------------------------
class MLPControllerNonIdealCase:
    def __init__(self, dt, device, dtype, H=10, K=2, I=14, initial_w=None, initial_v=None):
        """
        dt: time step for controller weight update.
        device, dtype: torch device and data type.
        H: number of hidden neurons.
        K: output dimension.
        I: input dimension for the MLP (here, 14).
        initial_w: initial outer weight matrix of shape (H+1, K).
        initial_v: initial inner weight matrix of shape (I, H).
        """
        self.dt = dt
        self.device = device
        self.dtype = dtype
        # Controller gains
        self.Kv = 20 * torch.eye(K, dtype=dtype, device=device)  # output feedback gain (2x2)
        self.lam = 5 * torch.eye(2, dtype=dtype, device=device)   # filter gain (2x2)
        self.F_rate = 5.0  # learning rate for outer weights
        self.G_rate = 5.0  # learning rate for inner weights
        self.kappa = 0.01   # weight decay gain
        self.H = H
        self.K = K
        self.I = I
        # Outer weight matrix w: shape (H+1, K) including bias row
        if initial_w is None:
            self.w = torch.zeros((H+1, K), dtype=dtype, device=device)
        else:
            self.w = initial_w.clone()
        # Inner weight matrix v: shape (I, H)
        if initial_v is None:
            self.v = torch.zeros((I, H), dtype=dtype, device=device)
        else:
            self.v = initial_v.clone()
        self.w_history = [self.w.clone()]
        self.v_history = [self.v.clone()]

    def step(self, x_des, x):
        """
        Compute control input using the MLP nonideal-case controller.
        
        x_des: desired state vector (6 elements)
               [qd (pos,2); qdp (vel,2); qdpp (acc,2)]
        x: current state vector (4 elements)
               [q (pos,2); q_dot (vel,2)]
               
        Returns:
            u: control input vector (2 elements)
            L_val: Lyapunov candidate value (scalar)
            w_dot: outer weight update (matrix of shape (H+1, K))
            v_dot: inner weight update (matrix of shape (I, H))
            tau_out: network output before adding feedback (tau)
        Also updates the internal weight matrices.
        """
        # Extract states
        qd = x_des[0:2]
        qdp = x_des[2:4]
        qdpp = x_des[4:6]
        q = x[0:2]
        q_dot = x[2:4]
        
        # Tracking errors
        e = qd - q
        ep = qdp - q_dot
        r = ep + torch.matmul(self.lam, e)  # r ∈ ℝ²
        
        # Construct x_new: concatenation of x (4), x_des (6), e (2), ep (2) => 14 elements
        x_new = torch.cat([x, x_des, e, ep])  # shape: (14,)
        
        # Outer weight matrix and inner weight matrix from previous iteration
        w = self.w  # shape (H+1, K)
        v = self.v  # shape (I, H)
        
        # Compute Frobenius norm of combined weight matrices:
        # Here we approximate Z_f_norm = sqrt(norm(w,'fro')^2 + norm(v,'fro')^2)
        Z_f_norm = torch.sqrt(torch.norm(w, p='fro')**2 + torch.norm(v, p='fro')**2)
        Zb = 1.0
        # Kz is identity 2x2
        # Compute v_t = - (Z_f_norm + Zb) * r
        v_t = -(Z_f_norm + Zb) * r  # shape (2,)
        
        # Hidden layer computation using inner weights v:
        sigma_x = torch.zeros(self.H, dtype=self.dtype, device=self.device)
        sigma_x_dot = torch.zeros(self.H, dtype=self.dtype, device=self.device)
        # Loop over hidden neurons h = 0,...,H-1
        for h in range(self.H):
            # in_out = dot(v[:,h], x_new)
            in_out = torch.dot(v[:, h], x_new)
            sigma_x[h] = 1.0 / (1.0 + torch.exp(-in_out))
            sigma_x_dot[h] = sigma_x[h] * (1 - sigma_x[h])
        
        # Augmented hidden layer output: add bias 1
        sigma_x_new = torch.cat([sigma_x, torch.tensor([1.0], dtype=self.dtype, device=self.device)])  # shape (H+1,)
        sigma_x_dot_new = torch.cat([sigma_x_dot, torch.tensor([0.0], dtype=self.dtype, device=self.device)])  # shape (H+1,)
        
        # Compute network output: tau = sigma_x_new^T * w, for each output dimension
        tau = torch.matmul(sigma_x_new, w)  # shape (K,)
        
        # Outer weight update (w_dot): loop over each output dimension k and each row h in 0,...,H
        norm_r = torch.norm(r)
        w_dot = torch.zeros_like(w)
        for k in range(self.K):
            for h in range(self.H + 1):
                # For h from 0 to H-1, include term from inner weights; for bias row (h == H), use 0
                term = torch.dot(x_new, v[:, h]) if h < self.H else 0.0
                w_dot[h, k] = (self.F_rate * sigma_x_new[h] * r[k] -
                               self.F_rate * sigma_x_dot_new[h] * term * r[k] -
                               self.kappa * self.F_rate * norm_r * w[h, k])
        
        # Inner weight update (v_dot): loop for h = 0,...,H-1 and for each input dimension i
        v_dot = torch.zeros_like(v)
        for h in range(self.H):
            # Compute dot product of row h of w with r
            s = torch.dot(w[h, :], r)
            for i in range(self.I):
                v_dot[i, h] = (self.G_rate * x_new[i] * (sigma_x_dot_new[h] * s) -
                               self.kappa * self.G_rate * norm_r * v[i, h])
        
        # Compute control input: u = tau + Kv*r - v_t
        u = tau + torch.matmul(self.Kv, r) - v_t
        u = torch.clamp(u, -10, 10)
        
        # Save tau as tau_out
        tau_out = tau.clone()
        
        # Lyapunov candidate: L = 0.5 * ||r||^2
        L_val = 0.5 * torch.dot(r, r)
        
        # Update weights with Euler integration
        self.w = self.w + self.dt * w_dot
        self.v = self.v + self.dt * v_dot
        self.w_history.append(self.w.clone())
        self.v_history.append(self.v.clone())
        
        return u, L_val, w_dot, v_dot, tau_out

    def stepFast(self, x_des, x):
        """
        Compute control input using the MLP ideal-case FLNN controller.
        
        x_des: desired state vector (6 elements)
               [qd (pos,2); qdp (vel,2); qdpp (acc,2)]
        x: current state vector (4 elements) [q (pos,2); q_dot (vel,2)]
               
        Returns:
            u: control input vector (2 elements)
            L_val: Lyapunov candidate value (scalar)
            w_dot: outer weight update (matrix of shape (H+1, K))
            v_dot: inner weight update (matrix of shape (I, H))
            tau_out: network output before adding feedback (tau)
        Also updates the internal weight matrices.
        """
        # Extract states
        qd   = x_des[0:2]
        qdp  = x_des[2:4]
        qdpp = x_des[4:6]
        q    = x[0:2]
        q_dot = x[2:4]
        
        # Compute tracking errors and filtered error
        e = qd - q
        ep = qdp - q_dot
        r = ep + torch.matmul(self.lam, e)  # shape (2,)
        
        # Construct x_new: concatenate [x, x_des, e, ep] -> shape (14,)
        x_new = torch.cat([x, x_des, e, ep])
        
        # Compute Frobenius norm of combined weight matrices (outer and inner)
        Z_f_norm = torch.sqrt(torch.norm(self.w, p='fro')**2 + torch.norm(self.v, p='fro')**2)
        Zb = 1.0
        # Compute v_t = - (Z_f_norm + Zb)*r
        v_t = -(Z_f_norm + Zb) * r  # shape (2,)
        
        # --- Hidden layer computation (vectorized) ---
        # Compute in_out for each hidden neuron: in_out = v^T * x_new, where v is (I, H)
        in_out = torch.matmul(x_new.unsqueeze(0), self.v).squeeze(0)  # shape (H,)
        sigma_x = 1.0 / (1.0 + torch.exp(-in_out))  # shape (H,)
        sigma_x_dot = sigma_x * (1 - sigma_x)  # shape (H,)
        
        # Augmented hidden output (append bias 1) and its derivative (append 0)
        sigma_x_new = torch.cat([sigma_x, torch.tensor([1.0], dtype=self.dtype, device=self.device)])
        sigma_x_dot_new = torch.cat([sigma_x_dot, torch.tensor([0.0], dtype=self.dtype, device=self.device)])
        
        # --- Outer weight update ---
        # Compute T for hidden neurons: T[h] = dot(x_new, v[:, h]) for h in 0...H-1; bias row gets 0.
        T = torch.matmul(x_new.unsqueeze(0), self.v).squeeze(0)  # shape (H,)
        T = torch.cat([T, torch.tensor([0.0], dtype=self.dtype, device=self.device)])  # shape (H+1,)
        
        norm_r = torch.norm(r)
        # Vectorized outer weight update: 
        # w_dot = F_rate*(sigma_x_new outer r) - F_rate*(sigma_x_dot_new outer r)*T - kappa*F_rate*norm_r*w
        w_dot = self.F_rate * (sigma_x_new.unsqueeze(1) * r.unsqueeze(0)) \
                - self.F_rate * (sigma_x_dot_new.unsqueeze(1) * T.unsqueeze(1) * r.unsqueeze(0)) \
                - self.kappa * self.F_rate * norm_r * self.w
        
        # --- Inner weight update ---
        # For each hidden neuron h=0,...,H-1, compute s[h] = dot(w[h,:], r)
        s = torch.matmul(self.w[:-1], r)  # shape (H,)
        # Compute inner update vectorized: 
        # v_dot = G_rate * (x_new[:, None] * (sigma_x_dot_new[:H] * s)[None, :]) - kappa*G_rate*norm_r*v
        v_dot = self.G_rate * (x_new.unsqueeze(1) * ( (sigma_x_dot_new[:self.H] * s).unsqueeze(0) )) \
                - self.kappa * self.G_rate * norm_r * self.v
        
        # Compute network output: tau = sigma_x_new^T * w (shape: (K,))
        tau = torch.matmul(sigma_x_new, self.w)
        
        # Compute control input: u = tau + Kv*r - v_t
        u = tau + torch.matmul(self.Kv, r) - v_t
        u = torch.clamp(u, -10, 10)
        
        tau_out = tau.clone()
        L_val = 0.5 * torch.dot(r, r)
        
        # Update weights using Euler integration
        self.w = self.w + self.dt * w_dot
        self.v = self.v + self.dt * v_dot
        
        self.w_history.append(self.w.clone())
        self.v_history.append(self.v.clone())
        
        return u, L_val, w_dot, v_dot, tau_out

# -----------------------------
# Complete Simulation Code
# -----------------------------
def main():
    # Simulation time settings
    dt = 0.0001  # simulation time step
    tFinal = 10.0
    timeVec = torch.arange(0, tFinal + dt, dt, dtype=dtype, device=device)
    numSteps = len(timeVec)

    # Plant parameters
    known_params = {'a1': 0.2, 'a2': 0.2, 'g': 9.81, 'c1': 0.0, 'c2': 0.0}
    true_params = {'m1': 0.5, 'm2': 0.5}
    plant_params = dict(known_params)
    plant_params['m1'] = true_params['m1']
    plant_params['m2'] = true_params['m2']

    # Create plant
    plant = TwoLinkManipulator(plant_params, dt, device, dtype)
    
    # Create MLP non ideal-case controller (MATLAB version)
    controller = MLPControllerNonIdealCase(dt, device, dtype)
    
    # Precompute reference trajectories for both joints (vectorized)
    theta1_ref = 0.3 + 0.2 * torch.sin(2 * math.pi * 0.5 * timeVec)
    theta1_ref_dot = 0.2 * (2 * math.pi * 0.5) * torch.cos(2 * math.pi * 0.5 * timeVec)
    theta1_ref_ddot = -0.2 * ((2 * math.pi * 0.5)**2) * torch.sin(2 * math.pi * 0.5 * timeVec)
    theta2_ref = 0.5 + 0.1 * torch.sin(2 * math.pi * 0.2 * timeVec)
    theta2_ref_dot = 0.1 * (2 * math.pi * 0.2) * torch.cos(2 * math.pi * 0.2 * timeVec)
    theta2_ref_ddot = -0.1 * ((2 * math.pi * 0.2)**2) * torch.sin(2 * math.pi * 0.2 * timeVec)

    # Storage for simulation data
    X_true = torch.zeros((numSteps, 4), dtype=dtype, device=device)
    u_store = torch.zeros((numSteps, 2), dtype=dtype, device=device)
    L_store = torch.zeros(numSteps, dtype=dtype, device=device)
    
    # Initial plant state: [theta1, theta2, dtheta1, dtheta2]
    X0 = torch.tensor([0.0, 0.2, 0.0, 0.0], dtype=dtype, device=device)
    X_true[0, :] = X0

    # Main simulation loop
    for k in range(numSteps - 1):
        x = X_true[k, :]
        print(k)
        # Form desired state vector x_des = [qd; qdp; qdpp]
        qd = torch.tensor([theta1_ref[k], theta2_ref[k]], dtype=dtype, device=device)
        qdp = torch.tensor([theta1_ref_dot[k], theta2_ref_dot[k]], dtype=dtype, device=device)
        qdpp = torch.tensor([theta1_ref_ddot[k], theta2_ref_ddot[k]], dtype=dtype, device=device)
        x_des = torch.cat([qd, qdp, qdpp])
        
        # Compute control input and Lyapunov candidate using the MLP ideal-case controller
        u, L_val, w_dot, v_dot, tau_out = controller.stepFast(x_des, x)
        u_store[k, :] = u
        L_store[k] = L_val
        
        # Update plant state
        X_true[k+1, :] = plant.step(x, u)
    
    # Convert data for plotting
    timeVec_np = timeVec.cpu().numpy()
    X_true_np = X_true.cpu().numpy()
    u_np = u_store.cpu().numpy()
    L_np = L_store.cpu().numpy()
    w_history_np = torch.stack(controller.w_history).cpu().numpy()
    v_history_np = torch.stack(controller.v_history).cpu().numpy()
    theta1_ref_np = theta1_ref.cpu().numpy()
    theta2_ref_np = theta2_ref.cpu().numpy()

    # Plot full outer weight evolution
    plt.figure()
    numW = w_history_np.shape[0]
    time_control = timeVec_np[:numW]
    for i in range(w_history_np.shape[1]):
        for j in range(w_history_np.shape[2]):
            plt.plot(time_control, w_history_np[:, i, j])
    plt.xlabel('Time (s)')
    plt.ylabel('Outer Weight Value')
    plt.title('Evolution of All Outer Weight Coefficients')
    plt.grid(True)
    plt.show()

    # Plot full inner weight evolution
    plt.figure()
    numV = v_history_np.shape[0]
    time_control_v = timeVec_np[:numV]
    for i in range(v_history_np.shape[1]):
        for j in range(v_history_np.shape[2]):
            plt.plot(time_control_v, v_history_np[:, i, j])
    plt.xlabel('Time (s)')
    plt.ylabel('Inner Weight Value')
    plt.title('Evolution of All Inner Weight Coefficients')
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
