%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% example 1.1.1 - output surface for one layer neural network

%y = ?(?4.79x1 + 5.90x2 ? 0.93) = ?(vx + b),

% set up NN weights
v= [-4.79 5.9];
b= [-0.93];
% set up plotting grid for sampling x:
[x1,x2]= meshgrid(-2 : 0.1 : 2);
% compute NN input vectors p and simulate NN using sigmoid:
p1= x1(:);
p2= x2(:);
p= [p1'; p2'];
a= simuff(p,v,b,'sigmoid');
% format results for using ’mesh’ or ’surfl’ plot routines:
a1= eye(41);
a1(:)= a';
mesh(x1,x2,a1);



%% example 1.1.1 - output surface for one layer neural network
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%