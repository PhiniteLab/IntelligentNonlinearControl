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
# FLNN Controller Class
# -----------------------------
class FLNNController:
    def __init__(self, dt, device, dtype, initial_weight=None):
        """
        dt: time step for controller weight update.
        device, dtype: torch device and data type.
        initial_weight: optional initial weight matrix of shape (4,2).
        """
        self.dt = dt
        self.device = device
        self.dtype = dtype
        # Controller gains
        self.Kv = 20 * torch.eye(2, dtype=dtype, device=device)
        self.lam = 5 * torch.eye(2, dtype=dtype, device=device)
        self.F_rate = 50.0  # learning rate
        
        # Initialize weight matrix (H=4 features, K=2 outputs)
        if initial_weight is None:
            self.weight = torch.zeros((4, 2), dtype=dtype, device=device)
        else:
            self.weight = initial_weight.clone()
        self.weight_history = [self.weight.clone()]

    def step(self, x_des, x):
        """
        Compute control input using FLNN method.
        
        x_des: desired state vector (6 elements) 
               [qd (pos,2); qdp (vel,2); qdpp (acc,2)]
        x: current state vector (4 elements) [q (pos,2); q_dot (vel,2)]
        
        Returns:
            u: control input vector (2 elements)
            L_val: Lyapunov candidate value (scalar)
        Also updates the internal weight matrix.
        """
        # Extract states
        qd   = x_des[0:2]
        qdp  = x_des[2:4]
        qdpp = x_des[4:6]
        q     = x[0:2]
        q_dot = x[2:4]
        
        # Tracking errors and filtered error
        e  = qd - q
        ep = qdp - q_dot
        r = ep + torch.matmul(self.lam, e)
        
        # Auxiliary signals
        zeta1 = qdpp + torch.matmul(self.lam, ep)
        zeta2 = qdp  + torch.matmul(self.lam, e)
        
        # FLNN feature extraction:
        # x_m_q: features based on zeta1 and q: [zeta1, zeta1*sin(q), zeta1*cos(q)]
        x_m_q = torch.cat([zeta1, 
                           zeta1 * torch.sin(q), 
                           zeta1 * torch.cos(q)])
        # x_v_m: features based on zeta2, q, and q_dot: [zeta2*sin(q)*q_dot, zeta2*cos(q)*q_dot]
        x_v_m = torch.cat([zeta2 * torch.sin(q) * q_dot, 
                           zeta2 * torch.cos(q) * q_dot])
        # x_g_q: features based on q: [cos(q), sin(q), sin(q)*cos(q), cos(q)*cos(q)]
        x_g_q = torch.cat([torch.cos(q), 
                           torch.sin(q), 
                           torch.sin(q) * torch.cos(q), 
                           torch.cos(q) * torch.cos(q)])
        # x_f_qd: features based on q_dot: [q_dot, sign(q_dot)]
        x_f_qd = torch.cat([q_dot, torch.sign(q_dot)])
        
        # Build the feature vector phi_x (4x1) by summing each group:
        phi_x = torch.zeros(4, dtype=self.dtype, device=self.device)
        phi_x[0] = torch.sum(x_m_q)
        phi_x[1] = torch.sum(x_v_m)
        phi_x[2] = torch.sum(x_g_q)
        phi_x[3] = torch.sum(x_f_qd)
        
        # Compute NN output: tau = weight^T * phi_x
        tau = torch.matmul(phi_x, self.weight)  # shape (2,)
        
        # Compute weight update: weight_dot = F_rate * (phi_x * r^T)
        weight_dot = self.F_rate * (phi_x.unsqueeze(1) * r.unsqueeze(0))
        
        # Compute control input: u = tau + Kv * r
        u = tau + torch.matmul(self.Kv, r)
        u = torch.clamp(u, -10, 10)
        
        # Compute Lyapunov candidate: L = 0.5 * r^T * r
        L_val = 0.5 * torch.dot(r, r)
        
        # Update weight with Euler integration using dt
        self.weight = self.weight + self.dt * weight_dot
        self.weight_history.append(self.weight.clone())
        
        return u, L_val

# -----------------------------
# Complete Simulation Code
# -----------------------------
def main():
    # Simulation time settings
    dt = 0.0001  # simulation time step
    tFinal = 5.0
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
    
    # Create FLNN controller with its own dt for weight update
    controller = FLNNController(dt, device, dtype)
    
    # Precompute reference trajectories for both joints
    theta1_ref = torch.zeros(numSteps, dtype=dtype, device=device)
    theta2_ref = torch.zeros(numSteps, dtype=dtype, device=device)
    theta1_ref_dot = torch.zeros(numSteps, dtype=dtype, device=device)
    theta2_ref_dot = torch.zeros(numSteps, dtype=dtype, device=device)
    theta1_ref_ddot = torch.zeros(numSteps, dtype=dtype, device=device)
    theta2_ref_ddot = torch.zeros(numSteps, dtype=dtype, device=device)
    
    # Assuming timeVec is a torch tensor of shape (numSteps,)
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
        # Form desired state vector x_des = [qd; qdp; qdpp]
        qd = torch.tensor([theta1_ref[k], theta2_ref[k]], dtype=dtype, device=device)
        qdp = torch.tensor([theta1_ref_dot[k], theta2_ref_dot[k]], dtype=dtype, device=device)
        qdpp = torch.tensor([theta1_ref_ddot[k], theta2_ref_ddot[k]], dtype=dtype, device=device)
        x_des = torch.cat([qd, qdp, qdpp])
        
        # Compute control input and Lyapunov candidate using FLNN controller
        u, L_val = controller.step(x_des, x)
        u_store[k, :] = u
        L_store[k] = L_val
        
        # Update plant state
        X_true[k+1, :] = plant.step(x, u)
    
    # Convert data for plotting
    timeVec_np = timeVec.cpu().numpy()
    X_true_np = X_true.cpu().numpy()
    u_np = u_store.cpu().numpy()
    L_np = L_store.cpu().numpy()
    weight_history_np = torch.stack(controller.weight_history).cpu().numpy()
    theta1_ref_np = theta1_ref.cpu().numpy()
    theta2_ref_np = theta2_ref.cpu().numpy()

    # Plot weight evolution (example: weight[0,0])
    plt.figure()
    numWeights = weight_history_np.shape[0]
    time_control = timeVec_np[:numWeights]  # time vector for controller updates

    for i in range(weight_history_np.shape[1]):  # over rows (4 features)
        for j in range(weight_history_np.shape[2]):  # over columns (2 outputs)
            plt.plot(time_control, weight_history_np[:, i, j])

    plt.xlabel('Time (s)')
    plt.ylabel('Weight Value')
    plt.title('Evolution of All Weight Coefficients')
    plt.legend()
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
