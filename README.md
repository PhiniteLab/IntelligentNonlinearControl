# IntelligentNonlinearControl
Intelligent Nonlinear Control

Reference Book : Neural Netowrk Control of Robot Manipulators and Nonlinear Systems, (F.L. Lewis, S. Jagannathan, A. Yeşildirek)

Chapter 1 : Background on Neural Network

Example 1.1.1 : Output Surface for one-layer Neural Network

This example generates a 3D mesh of a single-layer perceptron’s output for inputs x₁,x₂ ∈ [−2,2] sampled at 0.1 intervals, using MATLAB’s simuff routine with sigmoid activation. The resulting surface plot clearly shows how the network mapping varies across the input grid.

**Formulas:**  

y = σ(v x + b)                    (1.1)

Example 1.2.1 : Optimal NN Weights and Biases for Pattern Association

This example trains a single-layer perceptron to map the exemplar pairs  =0.4 and =0.8, then plots the error surface E(v,b) to identify the weight–bias combination that minimizes the output error. The network output uses a sigmoid activation, and performance is assessed by the least-squares criterion.

**Formulas:**  

y = σ(vᵀ x + b) 

E = ½[(Y¹ − y¹)² + (Y² − y²)²]                                   (1.2)

Chapter 2 : Background on Dynamic Systems

Example 2.4.1 : Simulation of Feedbcak Linearization Controller

This script first defines the nonlinear plant dynamics are ẋ₁ = x₁·x₂ + x₃, ẋ₂ = −2·x₂ + x₁·u ẋ₃ = sin(x₁) + 2·x₁·x₂ + u and the desired sinusoidal trajectory y<sub>d</sub> = sin(2π·t / T). At each time step it computes the feedback-linearization terms f(x) = sin(x₁) + x₂·x₃ + x₁·x₂²,  g(x) = 1 + x₁² then forms the tracking error e = y<sub>d</sub> − x₁, (and its derivatives) and applies MATLAB simulate the closed-loop response. Finally, it plots the plant output 
x₁, the reference y<sub>d</sub> and the error.

Control Laws:

u = (−f(x) + ÿ<sub>d</sub> + K<sub>d</sub>·ė + K<sub>p</sub>·e) / g(x) (2.1)

Result: Referance Trajectory Graphic

![git1](https://github.com/user-attachments/assets/a00e304b-eb49-4c93-b143-882ae17fb7a8)
 

Chapter 3 : Robot Dynamics and Control

Example 3.3.1 : Performance of PD-CT Controller for Two-Link Manipulator

This code suite, implemented in MATLAB, defines and simulates the dynamic model of a two-link robotic manipulator and its PD computed-torque controller design.

Control Laws:

τ = M(q)·(q̈<sub>d</sub> + K<sub>d</sub>·ė + K<sub>p</sub>·e) + N(q, q̇) (3.1)

Result : Referance Trajectory Graphic (Sample)

![git3_CTPD](https://github.com/user-attachments/assets/d2941676-cf9a-4c4f-9311-3cb433163b0e)

Result : Control Torque (Sample)

![git3_CTPD_torque](https://github.com/user-attachments/assets/63771717-444e-4938-b0b7-eb89de222871)

Example 3.3.1 : Performance of PID-CT Controller for Two-Link Manipulator

This code suite, implemented in MATLAB, defines and simulates the dynamic model of a two-link robotic manipulator and its PID computed-torque controller design.

Control Laws:

ε̇ = e

τ = M(q)·(q̈<sub>d</sub> + K<sub>d</sub>·ė + K<sub>p</sub>·e + K<sub>i</sub>·ε̇) + N(q, q̇) (3.2)

Result : Referance Trajectory Graphic (Sample)

![git3_CTPID](https://github.com/user-attachments/assets/abfab594-ce8f-4709-ab2c-cd2baaefde3e)

Result : Control Torque (Sample)

![git3_CTPID_torque](https://github.com/user-attachments/assets/89f9e865-558b-42e3-b4aa-219e1b61b991)

Example 3.3.2 : Performance of Classical Joint Controller (Sample)

This code suite, implemented in MATLAB, defines and simulates the dynamic model of a two-link robotic manipulator and its classical joint controller design.

Control Laws:

τ = K<sub>d</sub>·ė + K<sub>p</sub>·e + K<sub>i</sub>·ε̇  (3.3)

Result: 

Example 3.3.2 : Performance of PD-Gravity Controller (Sample)

Control Laws:

Result :

Example 3.4.1 : Performance of Adapitve Controller (Sample)

Control Laws:

Result :

Example 3.4.2 : Performance of Robust Controller (Sample)

Control Laws:

Result :

Summary of Controller:

![image](https://github.com/user-attachments/assets/8a85075c-833f-46fb-81af-c7be483d6198)

Chapter 4 : Neural Network Robot Control

