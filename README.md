# IntelligentNonlinearControl

Intelligent Nonlinear Control

Reference Book : Neural Netowrk Control of Robot Manipulators and Nonlinear Systems, (F.L. Lewis, S. Jagannathan, A. Yeşildirek)

Chapter 1 : Background on Neural Network

Example 1.1.1 : Output Surface for one-layer Neural Network

This example generates a 3D mesh of a single-layer perceptron’s output for inputs x₁,x₂ ∈ [−2,2] sampled at 0.1 intervals, using MATLAB’s simuff routine with sigmoid activation. The resulting surface plot clearly shows how the network mapping varies across the input grid. 

  <p>
    <strong>Activation output:</strong><br>
    $$y = \sigma\bigl(v^{\mathsf{T}} x + b\bigr)$$
  </p>                   

Example 1.2.1 : Optimal NN Weights and Biases for Pattern Association

This example trains a single-layer perceptron to map the exemplar pairs  =0.4 and =0.8, then plots the error surface E(v,b) to identify the weight–bias combination that minimizes the output error. The network output uses a sigmoid activation, and performance is assessed by the least-squares criterion.

  <p>
    <strong>Activation output:</strong><br>
    $$y = \sigma\bigl(v^{\mathsf{T}} x + b\bigr)$$
  </p>

  <p>
    <strong>Error function:</strong><br>
    $$E = \tfrac{1}{2}\Bigl[(Y^{1} - y^{1})^{2} \;+\; (Y^{2} - y^{2})^{2}\Bigr]$$
  </p>                                  

Chapter 2 : Background on Dynamic Systems

Example 2.4.1 : Simulation of Feedbcak Linearization Controller

This script first defines the nonlinear plant dynamics are ẋ₁ = x₁·x₂ + x₃, ẋ₂ = −2·x₂ + x₁·u ẋ₃ = sin(x₁) + 2·x₁·x₂ + u and the desired sinusoidal trajectory y<sub>d</sub> = sin(2π·t / T). At each time step it computes the feedback-linearization terms f(x) = sin(x₁) + x₂·x₃ + x₁·x₂²,  g(x) = 1 + x₁² then forms the tracking error e = y<sub>d</sub> − x₁, (and its derivatives) and applies MATLAB simulate the closed-loop response. Finally, it plots the plant output 
x₁, the reference y<sub>d</sub> and the error.

  <p>
    <strong>Control Input:</strong><br>
    $$u = \tfrac{-f(x) + \ddot y_{d} + K_{d}\,\dot e + K_{p}\,e}{g(x)}$$
  </p>

Result: Referance Trajectory Graphic

![git1](https://github.com/user-attachments/assets/a00e304b-eb49-4c93-b143-882ae17fb7a8)
 
Chapter 3 : Robot Dynamics and Control

Example 3.3.1 : Performance of PD-CT Controller for Two-Link Manipulator

In this approach, the manipulator’s nonlinear dynamics are exactly canceled by inverting the model and applying PD feedback around the desired trajectory, thereby reducing the closed‐loop behavior to a set of decoupled, linear second‐order systems. Its principal benefit lies in the high tracking precision and rapid transient response afforded by model‐based linearization. However, owing to the absence of an integral action, steady‐state errors cannot be eliminated, and robustness against parametric uncertainties is limited.

  <p>
    <strong>Control Torque:</strong><br>
    $$\tau \;=\; M(q)\bigl(\ddot q_d + K_v\,\dot e + K_p\,e\bigr)
       \;+\; V_m(q,\dot q)\,\dot q \;+\; F(\dot q) \;+\; G(q)$$
  </p>

Result : Referance Trajectory Graphic (Sample)

![git3_CTPD](https://github.com/user-attachments/assets/d2941676-cf9a-4c4f-9311-3cb433163b0e)

Example 3.3.1 : Performance of PID-CT Controller for Two-Link Manipulator

By augmenting the PD CT law with an integral term, this controller compensates for constant disturbances and modeling inaccuracies, thus eliminating residual steady‐state errors and enhancing long‐term tracking performance. Nevertheless, the integral component may introduce overshoot, degrade transient dynamics, and risk integrator wind-up under saturating inputs.

  <p>
    <strong>Derivation of Error:</strong><br>
    $$\dot \epsilon = e$$
  </p>
  <p>
    <strong>Control Torque:</strong><br>
    $$\tau \;=\; M(q)\bigl(\ddot q_d + K_v\,\dot e + K_p\,e + K_i\,\epsilon\bigr)
       \;+\; V_m(q,\dot q)\,\dot q \;+\; F(\dot q) \;+\; G(q)$$
  </p>

Result : Referance Trajectory Graphic (Sample)

![git3_CTPID](https://github.com/user-attachments/assets/abfab594-ce8f-4709-ab2c-cd2baaefde3e)

Example 3.3.2 : Performance of Classical Joint Controller (Sample)

A conventional PID controller is applied independently at each joint, without employing any model knowledge. This method is favored in industry due to its straightforward design, tuning ease, and minimal hardware requirements. Its drawback is that inter‐joint coupling is ignored, often necessitating high gains that can excite vibrations and provoke instability.

  <p>
    <strong>Derivative of Error:</strong><br>
    $$\dot \epsilon = e$$
  </p>
  <p>
    <strong>Control Torque:</strong><br>
    $$\tau = K_v\,\dot e + K_p\,e + K_i\,\epsilon$$
  </p>

Result: 

![class_git](https://github.com/user-attachments/assets/e7a27868-0251-4a6b-a65f-5f3f43713977)

Example 3.3.2 : Performance of PD-Gravity Controller (Sample)

This scheme combines a PD feedback term with a gravity‐cancellation element, while neglecting inertial and Coriolis effects. Its simplicity and direct compensation of gravitational loads render it effective for low-speed, precision tasks. However, omission of velocity‐dependent dynamics significantly degrades performance during fast motions or under heavy payload variations.

  <p>
    <strong>Control Torque:</strong><br>
    $$\tau = K_v\,\dot e \;+\; K_p\,e \;+\; G(q)$$
  </p>

Result : Referance Trajectory Graphic (Sample)

![git_gravity](https://github.com/user-attachments/assets/0b916054-c0b4-464e-9dc9-3975a94180ed)

Example 3.4.1 : Performance of Adapitve Controller (Sample)

This strategy employs online parameter estimation and adjustment to accommodate modeling uncertainties and load changes, thereby providing strong robustness against parametric variations. The principal challenge lies in its computational complexity, the requirement for persistent excitation to guarantee parameter convergence, and potentially slow adaptation rates.

  <p>
    <strong>Error definitions:</strong><br>
    $$e = q_d - q,\quad \dot e = \dot q_d - \dot q,\quad r = \dot e + \Lambda\,e$$
  </p>

  <p>
    <strong>Control Torque:</strong><br>
    $$\tau = K_v\,r + W(q,\dot q,q_d,\dot q_d,\ddot q_d,e,\dot e)\,\hat\phi$$
  </p>

  <p>
    <strong>Adaptation law:</strong><br>
    $$\dot{\hat\phi} = \Gamma\,W(q,\dot q,q_d,\dot q_d,\ddot q_d,e,\dot e)^{\!\top}\,r$$
  </p>

Result : Referance Trajectory Graphic (Sample)

![adaptive_referance](https://github.com/user-attachments/assets/e97fcc2e-a52f-4a05-b14e-b6b365a91b19)

Example 3.4.2 : Performance of Robust Controller (Sample)

  <p>
    <strong>Error definitions:</strong><br>
    $$e = q_d - q,\quad \dot e = \dot q_d - \dot q,\quad r = \dot e + \Lambda\,e$$
  </p>

  <p>
    <strong>Bounding function:</strong><br>
    $$F = \mu_2\,\|\ddot q_d + \Lambda\,\dot e\| + v_B\,\|\dot q\|\,\|\dot q_d + \Lambda\,e\|$$
  </p>

  <p>
    <strong>Robust term:</strong><br>
    $$v = -\,\frac{r\,F}{\max(\|r\|,\epsilon)}$$
  </p>

  <p>
    <strong>Control Torque:</strong><br>
    $$\tau = K_v\,r + G(q) - v$$
  </p>

Result : Referance Trajectory Graphic (Sample)

![robust_ref](https://github.com/user-attachments/assets/ed684475-68b2-4984-b933-7c2e9ca10cd9)

Summary of Controller for Robot Manipulator :

<table>
  <caption><strong>Robot Manipulator Control Algorithms</strong></caption>
  <thead>
    <tr>
      <th style="text-align:left;">Controller</th>
      <th style="text-align:left;">Equation</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><em>Robot Dynamics</em></td>
      <td>
        $$M(q)\,\ddot q + V_m(q,\dot q)\,\dot q + F(\dot q) + G(q) + \tau_d = \tau$$
      </td>
    </tr>
    <tr>
      <td><em>PD Computed Torque (CT) Control</em></td>
      <td>
        $$\tau = M(q)\bigl(\ddot q_d + K_v\,\dot e + K_p\,e\bigr)
         + V_m(q,\dot q)\,\dot q + F(\dot q) + G(q)$$
      </td>
    </tr>
    <tr>
      <td><em>PID Computed Torque (CT) Control</em></td>
      <td>
        $$\dot \epsilon = e$$<br>
        $$\tau = M(q)\bigl(\ddot q_d + K_v\,\dot e + K_p\,e + K_i\,\epsilon\bigr)
         + V_m(q,\dot q)\,\dot q + F(\dot q) + G(q)$$
      </td>
    </tr>
    <tr>
      <td><em>PD-Gravity Controller</em></td>
      <td>
        $$\tau = K_v\,\dot e + K_p\,e + G(q)$$
      </td>
    </tr>
    <tr>
      <td><em>Classical Joint Controller</em></td>
      <td>
        $$\dot \epsilon = e$$<br>
        $$\tau = K_v\,\dot e + K_p\,e + K_i\,\epsilon$$
      </td>
    </tr>
     <tr>
      <td><em>Adptive Controller</em></td>
      <td>
        $$\dot{\hat\phi} = \Gamma\,W(x)^{\!\top}\,r$$<br>
        $$\tau = K_v\,r + W(x)\,\hat\phi$$
      </td>
    </tr>
     <tr>
      <td><em>Robust Controller</em></td>
      <td>
        $$v = -\,\frac{r\,F}{\max(\|r\|,\epsilon)}$$<br>
        $$e = q_d - q,\quad \dot e = \dot q_d - \dot q,\quad r = \dot e + \Lambda\,e$$<br>
        $$\tau = K_v\,r + G(q) - v$$
      </td>
    </tr>
  </tbody>
</table>

Chapter 4 : Neural Network Robot Control

These controllers are designed to cover adaptive and robust control approaches starting from the classical PD based Computed Torque method and augmented with artificial neural networks. PD Computed Torque (plant1_FLNN_without_nn) provides low computational load with the simplest model inversion strategy but has limited flexibility against parameter uncertainties. FLNN based controllers offer better adaptation by modeling nonlinear dynamic terms with basis functions; especially damped adaptation (κ term) in plant1_FLNN_aug_no_pe aims at more stable learning by controlling excessive weight oscillations. plant1_FLNN_with_nn maintains the adaptation speed by updating the weights only with Hebbian-like adaptation, but the lack of damping may sometimes result in overfitting. Ideal MLP controller captures complex input-output relationships using 10 neurons in the hidden layer; it offers a richer learning mechanism with both weight and input weight updates. The nonideal MLP version provides tolerance to model uncertainties and noise with the addition of a robust term, thus reducing performance fluctuations. The Hebbian-based MLP controller provides a more biodynamic adaptation with a biologically inspired learning rule; this method allows adaptation at low complexity with simple learning equations. The dynamic Hebbian variant improves the response time in fast-changing dynamic environments by making the learning rates sensitive to the signal rate, but may be slightly more sensitive to noise. FLNN approaches can be scaled according to the number of input functions; faster computation is achieved at low H values ​​and increased model capacity at high H. In MLP-based controllers, the hidden layer size (H) and the input size (I) affect both the overall accuracy and computational load. The adaptation constants F and G control the convergence speed, while the κ term provides stability and damping. The robust term provides robustness against parameter uncertainties, thus improving reliability in real-time applications. PD-based methods provide fast response but limited flexibility; FLNN and MLP approaches provide learning mechanisms that restore this flexibility. Hebbian learning can reduce parameter tuning dependency, but due to its biological inspiration, it is difficult to tune. Dynamic Hebbian offers both fast adaptation and structured damping. Controllers should be selected according to the application's accuracy requirement, computational resources, and dynamic change rate; the balance between complexity, performance, and stability plays a critical role in this selection.

Example 4.1 : Table 4.2.1 - FLNN control for Ideal Case or Nonideal Case

In two-link manipulators, FLNN controllers are employed because they can approximate the arm’s strong nonlinear dynamics and adapt online to parameter uncertainties, yielding precise trajectory tracking without requiring an exact model inversion. Their main advantage is the universal‐approximation capability combined with a rigorous adaptation law, but they demand persistent excitation for weight convergence and incur additional computational overhead that may challenge strict real-time constraints.

  <p>
    <strong>Control Torque:</strong><br>
    $$\tau = \hat{W}^{T}\,\phi(x) + K_{v}\,r$$
  </p>
  
  <p>
    <strong>NN weight/threshold tuning:</strong><br>
    $$\dot{\hat{W}} = F\,\phi(x)\,r^{T}$$
  </p>

Result : Referance Trajectory Graphic (Sample)

  <p>
    <strong> FNN without NN </strong><br>
  </p>

![git_fnn_wo_nn](https://github.com/user-attachments/assets/049840be-9428-4715-a222-fb36da92f8f2)

  <p>
    <strong> FNN with NN </strong><br>
  </p>
  
![git_fnn_w_nn](https://github.com/user-attachments/assets/1767d594-d45f-421e-9a8d-c2c8412209d0)


Example 4.2 : Table 4.2.2 - FLNN Controller with Augmented Tuning to Avoid PE

In two-link manipulators, FLNN controllers are employed because they can approximate the arm’s strong nonlinear dynamics and adapt online to parameter uncertainties.The augmented tuning term preserves the model fit by suppressing parameter deviations even when the permanent excitation (PE) condition is not met. While these approaches offer advantages in robustness to uncertainties and parameter convergence, the design complexity and adaptation speed may remain low due to the additional 𝜅 κ parameter and computational load.

  <p>
    <strong>Control input:</strong><br>
    $$\tau = \hat{W}^{T}\,\phi(x) + K_{v}\,r$$
  </p>

  <p>
    <strong>NN weight/threshold tuning (augmented):</strong><br>
    $$\dot{\hat{W}} \;=\; F\,\phi(x)\,r^{T}\;-\;\kappa\,F\,\|r\|\;\hat{W}$$<br>
    <em>Design parameters:</em> $F$ positive-definite matrix, $\kappa>0$.
  </p>

Result : Referance Trajectory Graphic (Sample)

![gti_augm_4 2 2](https://github.com/user-attachments/assets/50637951-0898-43b7-9a58-f188032b0a38)

Example 4.3 : Table 4.3.1 - Two-Layer NN Controller for Ideal Case and Nonideal Case

Two-layer neural‐network controllers are used on two‐link manipulators because they can capture both the primary nonlinear dynamics and the residual modeling errors through a hidden layer structure, yielding superior trajectory tracking compared to single‐layer networks. Their online weight‐tuning laws ensure continual adaptation to parameter uncertainties, improving robustness under changing payloads and friction. However, the added network depth introduces more tuning parameters and computational overhead, which can complicate real‐time implementation and stability guarantees when excitation is limited.

  <p>
    <strong>Control input:</strong><br>
    $$\tau = \hat{W}^{T}\,\sigma\bigl(\hat{V}^{T}x\bigr) + K_{v}\,r - \nu$$
  </p>

  <p>
    <strong>NN weight/threshold tuning:</strong><br>
    $$\dot{\hat{W}} = F\,\hat{r}^{T},\quad
      \dot{\hat{V}} = G\,x\,\bigl(\sigma^{T}\hat{W}\,r\bigr)^{T}$$<br>
    <em>Design parameters:</em> $F,\,G$ positive-definite matrices.
  </p>

Result : Referance Trajectory Graphic (Sample)

Ideal Case

![git_mlp_ideal](https://github.com/user-attachments/assets/7f6416de-71cc-4dcf-a7bd-6c7659ec93e5)

Nonideal Case

![git_mlp_nonideal](https://github.com/user-attachments/assets/b7660933-746d-4fe1-bd0a-f8ff0ea03895)

Example 4.4 : Table 4.3.3 - Two-Layer NN Controller with Augmented Hebbian Tuning

</div>
Two-layer NN controllers with augmented Hebbian tuning are used on two-link manipulators because they combine local correlation learning with error-weighted suppression to ensure parameter adaptation even under weak excitation. This yields improved robustness to model uncertainty and persistent tracking accuracy without requiring full Lyapunov-based excitation conditions. However, the extra Hebbian term and κ tuning add design complexity, can introduce bias under sustained error, and increase computational overhead.

  <p>
    <strong>Control input:</strong><br>
    $$\tau = \hat{W}^{T}\,\sigma\bigl(\hat{V}^{T}x\bigr)\;+\;K_{v}\,r\;-\;v$$
  </p>

  <p>
    <strong>Robustifying signal:</strong><br>
    $$v(t) = -\,K_{z}\,\bigl(\|\hat{Z}\|_{F} + Z_{B}\bigr)\,r$$
  </p>
  
   <p>
    <strong>NN weight/threshold tuning (augmented Hebbian):</strong><br>
    $$\dot{\hat{W}} = F\,\hat{\sigma}\,r^{T}\;-\;\kappa\,F\,\|r\|\,\hat{W},$$  
    $$\dot{\hat{V}} = G\,\|r\|\,x\,\hat{\sigma}^{T}r^{T}\;-\;\kappa\,G\,\|r\|\,\hat{V}$$<br>
    <em>Design parameters:</em> $F,\,G$ positive-definite matrices, $\kappa>0$.
  </p>

Result : Referance Trajectory Graphic (Sample)

![git_mlp_hıb-dyn](https://github.com/user-attachments/assets/bd3f7e07-65a9-44ee-b8b6-bed1bc2edd2e)







