import torch
import math
import matplotlib.pyplot as plt
import numpy as np

# Select device: use CUDA if available, otherwise CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float64

class TwoLinkManipulator:
    def __init__(self, params, dt, device, dtype):
        self.params = params
        self.dt = dt
        self.device = device
        self.dtype = dtype

    def build_system_matrices(self, q, q_dot):
        """
        Compute the inertia matrix, Coriolis/centrifugal matrix, and gravity vector.
        """
        m1 = self.params['m1']
        m2 = self.params['m2']
        a1 = self.params['a1']
        a2 = self.params['a2']
        g  = self.params['g']
        c1 = self.params['c1']
        c2 = self.params['c2']

        theta1 = q[0]
        theta2 = q[1]
        dtheta1 = q_dot[0]
        dtheta2 = q_dot[1]

        # Inertia matrix M_q
        M11 = (m1 + m2)*a1**2 + m2*a2**2 + 2*m2*a1*a2*torch.cos(theta2)
        M12 = m2*a2**2 + m2*a1*a2*torch.cos(theta2)
        M21 = M12
        M22 = m2*a2**2
        M_q = torch.tensor([[M11, M12],
                            [M21, M22]], dtype=q.dtype, device=q.device)

        # Coriolis/centrifugal matrix V_q_qdot
        V11 = c1 - m2*a1*a2*torch.sin(theta2)*(2*dtheta2)
        V12 = -m2*a1*a2*torch.sin(theta2)*dtheta2
        V21 = m2*a1*a2*torch.sin(theta2)*dtheta1
        V22 = c2
        V_q_qdot = torch.tensor([[V11, V12],
                                 [V21, V22]], dtype=q.dtype, device=q.device)

        # Gravity vector G_q
        G1 = (m1 + m2)*g*a1*torch.cos(theta1) + m2*g*a2*torch.cos(theta1 + theta2)
        G2 = m2*g*a2*torch.cos(theta1 + theta2)
        G_q = torch.tensor([G1, G2], dtype=q.dtype, device=q.device)
        
        return M_q, V_q_qdot, G_q

    def step(self, x, Tau):
        """
        Propagate the state one time step using forward Euler integration.
        x: state vector [theta1, theta2, dtheta1, dtheta2]
        Tau: control torque vector [tau1, tau2]
        """
        q = x[0:2]
        q_dot = x[2:4]
        M_q, V_q_qdot, G_q = self.build_system_matrices(q, q_dot)
        q_ddot = torch.linalg.solve(M_q, (Tau - V_q_qdot @ q_dot - G_q))
        q_dot_new = q_dot + self.dt * q_ddot
        q_new = q + self.dt * q_dot_new
        x_next = torch.cat((q_new, q_dot_new))
        return x_next

class DFLController:
    def __init__(self, Kp, Kd, Ki, max_torque, dt, params, device, dtype):
        self.Kp = Kp
        self.Kd = Kd
        self.Ki = Ki
        self.max_torque = max_torque
        self.dt = dt
        self.params = params
        self.device = device
        self.dtype = dtype
        # Integrated error (for the I-term)
        self.error_int = torch.zeros(2, dtype=dtype, device=device)

    def step(self, q, q_dot, q_r, q_r_dot, q_r_ddot):
        """
        Compute the control torque using dynamic feedback linearization (DFL).
        """
        error = q_r - q
        error_dot = q_r_dot - q_dot
        self.error_int = self.error_int + self.dt * error
        v = self.Kp @ error + self.Kd @ error_dot + self.Ki @ self.error_int

        # Recompute system matrices using the current state
        m1 = self.params['m1']
        m2 = self.params['m2']
        a1 = self.params['a1']
        a2 = self.params['a2']
        g  = self.params['g']
        c1 = self.params['c1']
        c2 = self.params['c2']

        theta1 = q[0]
        theta2 = q[1]
        dtheta1 = q_dot[0]
        dtheta2 = q_dot[1]

        M11 = (m1 + m2)*a1**2 + m2*a2**2 + 2*m2*a1*a2*torch.cos(theta2)
        M12 = m2*a2**2 + m2*a1*a2*torch.cos(theta2)
        M21 = M12
        M22 = m2*a2**2
        M_current = torch.tensor([[M11, M12],
                                  [M21, M22]], dtype=q.dtype, device=q.device)

        V11 = c1 - m2*a1*a2*torch.sin(theta2)*(2*dtheta2)
        V12 = -m2*a1*a2*torch.sin(theta2)*dtheta2
        V21 = m2*a1*a2*torch.sin(theta2)*dtheta1
        V22 = c2
        V_current = torch.tensor([[V11, V12],
                                  [V21, V22]], dtype=q.dtype, device=q.device)

        G1 = (m1 + m2)*g*a1*torch.cos(theta1) + m2*g*a2*torch.cos(theta1 + theta2)
        G2 = m2*g*a2*torch.cos(theta1 + theta2)
        G_current = torch.tensor([G1, G2], dtype=q.dtype, device=q.device)
        
        Tau = V_current @ q_dot + G_current + M_current @ (q_r_ddot + v)
        Tau = torch.clamp(Tau, -self.max_torque, self.max_torque)
        return Tau

def main():
    # -------------------------------
    # Simulation and system parameters
    # -------------------------------
    dt = 0.01
    tFinal = 10.0
    timeVec = torch.arange(0, tFinal + dt, dt, dtype=dtype, device=device)
    numSteps = len(timeVec)

    # Two-link manipulator parameters
    params = {
        'm1': 0.5,    # link 1 mass
        'm2': 0.5,    # link 2 mass
        'a1': 0.2,    # link 1 length (or CoM distance)
        'a2': 0.2,    # link 2 length (or CoM distance)
        'g': 9.81,    # gravity
        'c1': 0.5,
        'c2': 0.5
    }

    # Controller gains (as in your original code)
    s1 = 2.0
    s2 = 1.0
    s3 = 4.0
    Kp = (s3*(s1 + s2) + s1*s2) * torch.eye(2, dtype=dtype, device=device)
    Kd = (s1 + s2 + s3) * torch.eye(2, dtype=dtype, device=device)
    Ki = (s1 * s2 * s3) * torch.eye(2, dtype=dtype, device=device)
    max_torque = 10.0

    # Initial "true" state: [theta1, theta2, dtheta1, dtheta2]
    X0_true = torch.tensor([0.0, 0.2, 0.0, 0.0], dtype=dtype, device=device)

    # Create plant (system) object
    plant = TwoLinkManipulator(params, dt, device, dtype)

    # -------------------------------
    # Precompute reference trajectories
    # -------------------------------
    theta1_ref_store = torch.zeros(numSteps, dtype=dtype, device=device)
    theta2_ref_store = torch.zeros(numSteps, dtype=dtype, device=device)
    theta1_ref_dot_store = torch.zeros(numSteps, dtype=dtype, device=device)
    theta1_ref_ddot_store = torch.zeros(numSteps, dtype=dtype, device=device)
    theta2_ref_dot_store = torch.zeros(numSteps, dtype=dtype, device=device)
    theta2_ref_ddot_store = torch.zeros(numSteps, dtype=dtype, device=device)

    with torch.inference_mode():
        for k in range(numSteps):
            t_k = timeVec[k].item()
            # Joint 1 reference signals
            theta1_r      = 0.3 + 0.2 * math.sin(2 * math.pi * 0.5 * t_k)
            theta1_r_dot  = 0.2 * (2 * math.pi * 0.5) * math.cos(2 * math.pi * 0.5 * t_k)
            theta1_r_ddot = -0.2 * ((2 * math.pi * 0.5)**2) * math.sin(2 * math.pi * 0.5 * t_k)
            # Joint 2 reference signals
            theta2_r      = 0.5 + 0.1 * math.sin(2 * math.pi * 0.2 * t_k)
            theta2_r_dot  = 0.1 * (2 * math.pi * 0.2) * math.cos(2 * math.pi * 0.2 * t_k)
            theta2_r_ddot = -0.1 * ((2 * math.pi * 0.2)**2) * math.sin(2 * math.pi * 0.2 * t_k)
            
            theta1_ref_store[k] = theta1_r
            theta1_ref_dot_store[k] = theta1_r_dot
            theta1_ref_ddot_store[k] = theta1_r_ddot
            theta2_ref_store[k] = theta2_r
            theta2_ref_dot_store[k] = theta2_r_dot
            theta2_ref_ddot_store[k] = theta2_r_ddot

    # -------------------------------
    # Main simulation loop over measurement cases
    # -------------------------------
    torch.manual_seed(0)

    # Preallocate arrays for storing true states, measurements, and torques.
    X_true = torch.zeros((numSteps, 4), dtype=dtype, device=device)
    tau_store = torch.zeros((numSteps, 2), dtype=dtype, device=device)
        
    X_true[0, :] = X0_true
        
    controller = DFLController(Kp, Kd, Ki, max_torque, dt, params, device, dtype)
        
    # Simulation loop (time stepping)
    with torch.inference_mode():
        for k in range(numSteps - 1):
            # Get current reference signals
            theta1_r      = theta1_ref_store[k]
            theta1_r_dot  = theta1_ref_dot_store[k]
            theta1_r_ddot = theta1_ref_ddot_store[k]
            theta2_r      = theta2_ref_store[k]
            theta2_r_dot  = theta2_ref_dot_store[k]
            theta2_r_ddot = theta2_ref_ddot_store[k]
            q_r      = torch.tensor([theta1_r, theta2_r], dtype=dtype, device=device)
            q_r_dot  = torch.tensor([theta1_r_dot, theta2_r_dot], dtype=dtype, device=device)
            q_r_ddot = torch.tensor([theta1_r_ddot, theta2_r_ddot], dtype=dtype, device=device)
                
            # Use the true state directly for control
            q = X_true[k, 0:2]
            q_dot = X_true[k, 2:4]
            Tau = controller.step(q, q_dot, q_r, q_r_dot, q_r_ddot)
            tau_store[k, :] = Tau
                
            # Simulate the true system using plant dynamics
            x_current = X_true[k, :]
            x_next = plant.step(x_current, Tau)
            X_true[k+1, :] = x_next
                
        # -------------------------------
        # Plotting the results for current measurement case
        # -------------------------------
        timeVec_np = timeVec.cpu().numpy()
        X_true_np = X_true.cpu().numpy()
        tau_store_np = tau_store.cpu().numpy()
        
        th1_true   = X_true_np[:, 0]
        th2_true   = X_true_np[:, 1]
        dth1_true  = X_true_np[:, 2]
        dth2_true  = X_true_np[:, 3]
        
        th1_ref = theta1_ref_store.cpu().numpy()
        th2_ref = theta2_ref_store.cpu().numpy()
        
        plt.figure(figsize=(12, 8))
        plt.suptitle(f'Two link Computed Torque Controller')
        
        # Joint 1 Angle
        plt.subplot(3, 2, 1)
        plt.plot(timeVec_np, th1_true, label='True')
        plt.plot(timeVec_np, th1_ref, '-.', label='Ref')
        plt.xlabel('Time (s)')
        plt.ylabel('theta1')
        plt.title('Joint 1 Angle')
        plt.grid(True)
        plt.legend()
        
        # Joint 2 Angle
        plt.subplot(3, 2, 2)
        plt.plot(timeVec_np, th2_true, label='True')
        plt.plot(timeVec_np, th2_ref, '-.', label='Ref')
        plt.xlabel('Time (s)')
        plt.ylabel('theta2')
        plt.title('Joint 2 Angle')
        plt.grid(True)
        plt.legend()
        
        # Joint 1 Velocity
        plt.subplot(3, 2, 3)
        plt.plot(timeVec_np, dth1_true, label='True')
        plt.xlabel('Time (s)')
        plt.ylabel('dtheta1')
        plt.title('Joint 1 Velocity')
        plt.grid(True)
        plt.legend()
        
        # Joint 2 Velocity
        plt.subplot(3, 2, 4)
        plt.plot(timeVec_np, dth2_true, label='True')
        plt.xlabel('Time (s)')
        plt.ylabel('dtheta2')
        plt.title('Joint 2 Velocity')
        plt.grid(True)
        plt.legend()
        
        # Joint 1 Torque
        plt.subplot(3, 2, 5)
        plt.plot(timeVec_np, tau_store_np[:, 0])
        plt.xlabel('Time (s)')
        plt.ylabel('tau1')
        plt.title('Joint 1 Torque')
        plt.grid(True)
        
        # Joint 2 Torque
        plt.subplot(3, 2, 6)
        plt.plot(timeVec_np, tau_store_np[:, 1])
        plt.xlabel('Time (s)')
        plt.ylabel('tau2')
        plt.title('Joint 2 Torque')
        plt.grid(True)
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()

if __name__ == '__main__':
    main()
