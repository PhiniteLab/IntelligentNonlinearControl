%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% obtain lyapunov
clear all;
close all;
clc;

syms e q_des2 q_dot_des2 A B C D q1 q2 u;

e_dot = D*e + q_dot_des2 - D*q_des2 - C*q1;

q1_dot = A*q1 + B*q2 + u;


%lyapunov
L = e*e_dot;

pretty(collect(L,u))












%% obtain lyapunov
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%