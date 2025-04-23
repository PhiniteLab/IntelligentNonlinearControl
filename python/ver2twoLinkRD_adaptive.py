import torch
import math
import matplotlib.pyplot as plt
import numpy as np

# Set up device and data type
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float64

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
        """
        Compute the inertia matrix M_q, Coriolis/centrifugal matrix V_q_qdot, 
        and gravity vector G_q for the two-link manipulator.
        """
        p = self.params
        m1 = p['m1'] if 'm1' in p else 0.0
        m2 = p['m2'] if 'm2' in p else 0.0
        a1 = p['a1']
        a2 = p['a2']
        g  = p['g']
        c1 = p['c1']
        c2 = p['c2']
        
        theta1 = q[0]
        theta2 = q[1]
        dtheta1 = q_dot[0]
        dtheta2 = q_dot[1]
        
        # Inertia matrix (2x2)
        M11 = (m1 + m2)*a1**2 + m2*a2**2 + 2*m2*a1*a2*torch.cos(theta2)
        M12 = m2*a2**2 + m2*a1*a2*torch.cos(theta2)
        M21 = M12
        M22 = m2*a2**2
        M_q = torch.tensor([[M11, M12],
                            [M21, M22]], dtype=q.dtype, device=q.device)
        
        # Coriolis/centrifugal matrix (including damping terms c1, c2)
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

class AdaptiveController:
    def __init__(self, dt, params, Gamma, Kr, Lambda, max_torque, device, dtype, initial_param_hat):
        """
        dt: time step.
        params: dictionary with known parameters (a1, a2, g, c1, c2).
        Gamma, Kr, Lambda: adaptive controller gains (2x2 tensors).
        max_torque: saturation limit for torque.
        initial_param_hat: initial estimate [alpha1, alpha2] = [m1+m2, m2].
        """
        self.dt = dt
        self.params = params
        self.Gamma = Gamma
        self.Kr = Kr
        self.Lambda = Lambda  # This is a 2x2 matrix; we will extract a scalar lambda for the regressor.
        self.max_torque = max_torque
        self.device = device
        self.dtype = dtype
        self.param_hat = initial_param_hat.clone()
        self.param_hat_history = [self.param_hat.clone()]

    def build_regressor(self, q, q_dot, e, q_ddot_des):
        """
        Build the regressor matrix W(q, q_dot, e, q_ddot_des) from the provided formula:
        
          W_11 = a1^2 (q_ddot1 + lambda*e1) + a1*g*cos(q1)
          W_12 = [ (a2^2 + 2*a1*a2*cos(q2) + a1^2)*(q_ddot1 + lambda*e1)
                   + (a2^2 + a1*a2*cos(q2))*(q_ddot2 + lambda*e2)
                   + a1*a2*(dot{q1} + dot{q2})*sin(q2)
                   + a2*g*cos(q1) ]
          W_21 = 0
          W_22 = [ (a2^2 + a1*a2*cos(q2))*(q_ddot2 + lambda*e2)
                   + a1*a2*(dot{q1} + dot{q2})*sin(q2)
                   + a2*g*cos(q1 + q2) ]
        
        where:
          q        = [q1, q2]^T
          q_dot    = [dot{q1}, dot{q2}]^T
          e        = [e1, e2]^T   (tracking error)
          q_ddot_des = [q_ddot1, q_ddot2]^T (desired acceleration)
          lambda   = scalar, extracted from self.Lambda (assume self.Lambda[0,0])
          a1, a2, g are from params dict.
        
        Returns:
          W (2x2 PyTorch tensor)
        """
        a1 = self.params['a1']
        a2 = self.params['a2']
        g  = self.params['g']
        
        # Unpack state and error (assumed to be scalar 0-d tensors)
        q1, q2 = q[0], q[1]
        dq1, dq2 = q_dot[0], q_dot[1]
        e1, e2 = e[0], e[1]
        ddq1_des, ddq2_des = q_ddot_des[0], q_ddot_des[1]
        
        # Extract scalar lambda from the diagonal of self.Lambda
        lam = self.Lambda[0, 0]
        # Combine desired accelerations with lambda*error
        dqd1 = ddq1_des + lam * e1
        dqd2 = ddq2_des + lam * e2
        
        # Build each entry of the regressor
        W11 = a1**2 * dqd1 + a1 * g * torch.cos(q1)
        W12 = ((a2**2 + 2*a1*a2*torch.cos(q2) + a1**2) * dqd1 +
               (a2**2 + a1*a2*torch.cos(q2)) * dqd2 +
               a1*a2*(dq1 + dq2)*torch.sin(q2) +
               a2*g*torch.cos(q1))
        W21 = torch.tensor(0.0, dtype=self.dtype, device=self.device)
        W22 = ((a2**2 + a1*a2*torch.cos(q2)) * dqd2 +
               a1*a2*(dq1 + dq2)*torch.sin(q2) +
               a2*g*torch.cos(q1 + q2))
        
        # Use torch.stack to create a 2x2 tensor without conversion errors.
        W = torch.stack([torch.stack([W11, W12]), torch.stack([W21, W22])])
        return W

    def compute_control(self, q, q_dot, q_r, q_r_dot, q_r_ddot):
        """
        Implements the adaptive control law:
          Tau = W(q,q_dot,q_ddot_des)*param_hat + Kr*r,
        where r = (q_r_dot - q_dot) + Lambda*(q_r - q),
        and the parameter update is:
          dot{param_hat} = Gamma * W^T * r.
        Updates the internal parameter estimate.
        """
        # Tracking errors
        e = q_r - q
        e_dot = q_r_dot - q_dot
        r = e_dot + self.Lambda @ e

        # Build regressor matrix
        W = self.build_regressor(q, q_dot, e, q_r_ddot)
        
        # Compute control torque
        Tau = W @ self.param_hat + self.Kr @ r
        
        # Update parameter estimate (forward Euler)
        param_dot = self.Gamma @ (W.T @ r)
        self.param_hat = self.param_hat + self.dt * param_dot
        self.param_hat_history.append(self.param_hat.clone())
        
        # Saturate torque
        Tau = torch.clamp(Tau, -self.max_torque, self.max_torque)
        return Tau, r

def main():
    # Simulation time settings
    dt = 0.001
    tFinal = 5.0
    timeVec = torch.arange(0, tFinal + dt, dt, dtype=dtype, device=device)
    numSteps = len(timeVec)

    # Known parameters (except for the masses, which are adapted)
    known_params = {
        'a1': 0.2,
        'a2': 0.2,
        'g': 9.81,
        'c1': 0.0,
        'c2': 0.0
    }
    # True system masses
    true_params = {
        'm1': 0.5,
        'm2': 0.5
    }
    # For the plant simulation, combine known parameters with true masses.
    plant_params = dict(known_params)
    plant_params['m1'] = true_params['m1']
    plant_params['m2'] = true_params['m2']

    # Adaptive controller gains
    # Note: Here, Lambda is used both as a gain in the control law (matrix multiplication)
    # and to compute the regressor. For the regressor, we use its (0,0) element as a scalar.
    Lambda = 1.0 * torch.eye(2, dtype=dtype, device=device)
    Kr = 5.0 * torch.eye(2, dtype=dtype, device=device)
    Gamma = 10.0 * torch.eye(2, dtype=dtype, device=device)
    max_torque = 10.0
    initial_param_hat = torch.tensor([0.3, 0.3], dtype=dtype, device=device)

    # Create plant and controller objects
    plant = TwoLinkManipulator(plant_params, dt, device, dtype)
    controller = AdaptiveController(dt, known_params, Gamma, Kr, Lambda, max_torque, device, dtype, initial_param_hat)

    # Precompute reference trajectories
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

    # Storage arrays for simulation data
    X_true = torch.zeros((numSteps, 4), dtype=dtype, device=device)
    tau_store = torch.zeros((numSteps, 2), dtype=dtype, device=device)
    
    # Set initial true state: [theta1, theta2, dtheta1, dtheta2]
    X0 = torch.tensor([0.0, 0.2, 0.0, 0.0], dtype=dtype, device=device)
    X_true[0, :] = X0

    # Main simulation loop
    for k in range(numSteps - 1):
        # Reference signals at step k
        q_r = torch.tensor([theta1_ref[k], theta2_ref[k]], dtype=dtype, device=device)
        q_r_dot = torch.tensor([theta1_ref_dot[k], theta2_ref_dot[k]], dtype=dtype, device=device)
        q_r_ddot = torch.tensor([theta1_ref_ddot[k], theta2_ref_ddot[k]], dtype=dtype, device=device)
        
        # True state at step k
        q = X_true[k, 0:2]
        q_dot = X_true[k, 2:4]
        
        # Compute adaptive control using the true state
        Tau, _ = controller.compute_control(q, q_dot, q_r, q_r_dot, q_r_ddot)
        tau_store[k, :] = Tau
        
        # For simulation, update the plant with the true masses.
        X_true[k+1, :] = plant.step(X_true[k, :], Tau)

    # Convert data to NumPy arrays for plotting
    timeVec_np = timeVec.cpu().numpy()
    X_true_np = X_true.cpu().numpy()
    tau_np = tau_store.cpu().numpy()
    param_np = torch.stack(controller.param_hat_history).cpu().numpy()
    theta1_ref_np = theta1_ref.cpu().numpy()
    theta2_ref_np = theta2_ref.cpu().numpy()

    # Plot the adaptive parameter estimates
    plt.figure()
    plt.plot(timeVec_np[:len(param_np)], param_np[:, 0], label='hat (m1+m2)')
    plt.plot(timeVec_np[:len(param_np)], param_np[:, 1], label='hat m2')
    plt.xlabel('Time (s)')
    plt.ylabel('Parameter Estimates')
    plt.title('Adaptive Parameter Estimates')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Plot joint angles (true state vs. reference)
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 1, 1)
    plt.plot(timeVec_np, X_true_np[:, 0], 'b', label='True theta1')
    plt.plot(timeVec_np, theta1_ref_np, 'm-.', label='Ref theta1')
    plt.xlabel('Time (s)')
    plt.ylabel('Theta1')
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 1, 2)
    plt.plot(timeVec_np, X_true_np[:, 1], 'b', label='True theta2')
    plt.plot(timeVec_np, theta2_ref_np, 'm-.', label='Ref theta2')
    plt.xlabel('Time (s)')
    plt.ylabel('Theta2')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Plot control torques
    plt.figure(figsize=(12, 6))
    plt.subplot(2, 1, 1)
    plt.plot(timeVec_np, tau_np[:, 0], 'k', label='tau1')
    plt.xlabel('Time (s)')
    plt.ylabel('Tau1')
    plt.legend()
    plt.grid(True)
    plt.subplot(2, 1, 2)
    plt.plot(timeVec_np, tau_np[:, 1], 'k', label='tau2')
    plt.xlabel('Time (s)')
    plt.ylabel('Tau2')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == '__main__':
    main()
