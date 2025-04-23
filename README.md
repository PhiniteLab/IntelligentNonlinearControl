# IntelligentNonlinearControl
Intelligent Nonlinear Control

Reference Book : Neural Netowrk Control of Robot Manipulators and Nonlinear Systems, (F.L. Lewis, S. Jagannathan, A. Yeşildirek)

Chapter 1 : Background on Neural Network

Example 1.1.1 : Output Surface for one-layer Neural Network

This example generates a 3D mesh of a single-layer perceptron’s output for inputs x₁,x₂ ∈ [−2,2] sampled at 0.1 intervals, using MATLAB’s simuff routine with sigmoid activation. The resulting surface plot clearly shows how the network mapping varies across the input grid.

**Formulas:**  

y = σ(v x + b) = σ(-4.79 x₁ + 5.90 x₂ - 0.93)  (1.1)

Example 1.2.1 : Optimal NN Weights and Biases for Pattern Association

This example trains a single-layer perceptron to map the exemplar pairs  =0.4 and =0.8, then plots the error surface E(v,b) to identify the weight–bias combination that minimizes the output error. The network output uses a sigmoid activation, and performance is assessed by the least-squares criterion.

**Formulas:**  

y = σ(vᵀ x + b)
E = ½[(Y¹ − y¹)² + (Y² − y²)²]

Chapter 2 : Background on Dynamic Systems

Chapter 3 : Robot Dynamics and Control

![image](https://github.com/user-attachments/assets/8a85075c-833f-46fb-81af-c7be483d6198)

Chapter 4 : Neural Network Robot Control

