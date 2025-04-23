%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% example 1.1.1 - output surface for one layer neural network

%y = ?(vx + b).

% set up input patterns, output targets, and weight/bias ranges:
p= [-3 2];
t= [0.4 0.8];
wv= -4 : 0.1 : 4;
bv= -4 : 0.1 : 4;
% compute output error surface:
es= errsurf(p,t,wv,bv,'logsig');
% plot and label error surface:
mesh(wv,bv,es)
view(60,30)
set(gca,'xlabel',text(0,0,'weight'))
set(gca,'ylabel',text(0,0,'bias'))
title('Error surface plot using sigmoid')


%% example 1.1.1 - output surface for one layer neural network
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%