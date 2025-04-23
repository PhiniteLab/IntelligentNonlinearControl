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

This code suite, implemented in MATLAB, defines and simulates the dynamic model of a two-link robotic manipulator and its PD computed-torque controller design.

  <p>
    <strong>Control Torque:</strong><br>
    $$\tau \;=\; M(q)\bigl(\ddot q_d + K_v\,\dot e + K_p\,e\bigr)
       \;+\; V_m(q,\dot q)\,\dot q \;+\; F(\dot q) \;+\; G(q)$$
  </p>

Result : Referance Trajectory Graphic (Sample)

![git3_CTPD](https://github.com/user-attachments/assets/d2941676-cf9a-4c4f-9311-3cb433163b0e)

Example 3.3.1 : Performance of PID-CT Controller for Two-Link Manipulator

This code suite, implemented in MATLAB, defines and simulates the dynamic model of a two-link robotic manipulator and its PID computed-torque controller design.

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

This code suite, implemented in MATLAB, defines and simulates the dynamic model of a two-link robotic manipulator and its classical joint controller design.

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

  <p>
    <strong>Control Torque:</strong><br>
    $$\tau = K_v\,\dot e \;+\; K_p\,e \;+\; G(q)$$
  </p>

Result : Referance Trajectory Graphic (Sample)

![git_gravity](https://github.com/user-attachments/assets/0b916054-c0b4-464e-9dc9-3975a94180ed)

Example 3.4.1 : Performance of Adapitve Controller (Sample)

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
    <strong>Gravity compensation:</strong><br>
    $$G(q) = 
    \begin{bmatrix}
      (m_1 + m_2)\,g\,a_1\cos(q_1) + m_2\,g\,a_2\cos(q_1 + q_2)\\[4pt]
      m_2\,g\,a_2\cos(q_1 + q_2)
    \end{bmatrix}$$
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

