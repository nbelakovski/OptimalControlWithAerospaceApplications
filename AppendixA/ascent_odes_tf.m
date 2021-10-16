function dX_dtau = ascent_odes_tf(tau,X,tf)
%
% State and Costate Differential Equation Function for the Flat-Moon
% Optimal Ascent Problem
%
%
% The independent variable here is the nondimensional time, tau, the state
% vector is X, and the final time, tf, is an unknown parameter that must
% also be passed to the DE function.
% Note that the state vector X has components
% X(1) = x, horizontal component of position
% X(2) = y, vertical component of position
% X(3) = Vx, horizontal component of velocity
% X(4) = Vy, vertical component of velocity
% X(5) = lambda2_bar
% X(6) = lambda4_bar
global g_accel Thrust2Weight
% Acceleration (F/m) of the Ascent Vehicle, m/s^2
Acc = Thrust2Weight*g_accel;
% State and Costate differential equations in terms of d/dt:
xdot = X(3);
ydot = X(4);
Vxdot = Acc*(1/sqrt(1+X(6)^2));
Vydot = Acc*(X(6)/sqrt(1+X(6)^2)) - g_accel;
lambda2_bar_dot = 0;
lambda4_bar_dot = -X(5);
% Nondimensionalize time (with tau = t/tf and d/dtau = tf*d/dt). We must
% multiply each differential equation by tf to convert our derivatives from
% d/dt to d/dtau.
dX_dtau = tf*[xdot; ydot; Vxdot; Vydot; lambda2_bar_dot; lambda4_bar_dot];
return