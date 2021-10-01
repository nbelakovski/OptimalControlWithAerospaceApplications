% Optimal Ascent Problem with MATLAB's bvp4c
%
% by Jose J. Guzman and George E. Pollock
%
% This script uses MATLAB's bvp4c to solve the problem of finding the
% optimal ascent trajectory for launch from a flat Moon to a 100 nautical
% mile circular orbit. In addition to this script, we use two functions:
% one to provide the differential equations and another that gives the
% boundary conditions:
%
% This file: OptimalAscent.m
% State and Costate Equations: ascent_odes_tf.m
% Boundary Conditions: ascent_bcs_tf.m
%
close all; clear all; clc;
% Define parameters of the problem
global g_accel Thrust2Weight % pass these parameters to the DE function
h = 185.2e3; % meters, final altitude (100 nmi circular orbit)
Vc = 1.627e3; % m/s, Circular speed at 100 nmi
g_accel = 1.62; % m/sˆ2, gravitational acceleration of Moon
Thrust2Weight = 3; % Thrust to Weight ratio for Ascent Vehicle, in lunar G's
rad2deg = 180/pi;
%----------------------------------------------------------------------------
%% Boundary Conditions
%----------------------------------------------------------------------------
global x0 y0 Vx0 Vy0 yf Vxf Vyf % pass these BCs to the BC function
% Initial conditions
% Launch from zero altitude with zero initial velocity
x0 = 0; % meters, initial x-position
y0 = 0; % meters, initial y-position
Vx0 = 0; % m/s, initial downrange velocity
Vy0 = 0; % m/s, initial vertical velocity
% Final conditions
yf = h; % meters, final altitude
Vxf = Vc; % m/s, final downrange velocity
Vyf = 0; % m/s, final vertical velocity
%----------------------------------------------------------------------------
%% Initial Guesses
%----------------------------------------------------------------------------
% initial time
t0 = 0;
% list initial conditions in yinit, use zero if unknown
yinit = [x0 y0 Vx0 Vy0 0 0]; % guess for initial state and costate variables
tf_guess = 700; % sec, initial guess for final time

%% Page 218

% Because the final time is free, we must parameterize the problem by
% the unknown final time, tf. Create a nondimensional time vector,
% tau, with Nt linearly spaced elements. (tau = time/tf) We will pass the
% unknown final time to bvp4c as an unknown parameter and the code will
% attempt to solve for the actual final time as it solves our TPBVP.
Nt = 41;
tau = linspace(0,1,Nt)'; % nondimensional time vector
% Create an initial guess of the solution using the MATLAB function
% bvpinit, which requires as inputs the (nondimensional) time vector,
% initial states (or guesses, as applicable), and the guess for the final
% time. The initial guess of the solution is stored in the structure
% solinit.
solinit = bvpinit(tau,yinit',tf_guess);
%----------------------------------------------------------------------------
%% Solution
%----------------------------------------------------------------------------
% Call bvp4c to solve the TPBVP. Point the solver to the functions
% containing the differential equations and the boundary conditions and
% provide it with the initial guess of the solution.
sol = bvp4cfe(@ascent_odes_tf, @ascent_bcs_tf, solinit);
% Extract the final time from the solution:
tf = sol.parameters(1);
% Evaluate the solution at all times in the nondimensional time vector tau
% and store the state variables in the matrix Z.
Z = deval(sol,tau);
% Convert back to dimensional time for plotting
time = t0 + tau.*(tf-t0);
% Extract the solution for each state variable from the matrix Z:
x_sol = Z(1,:);
y_sol = Z(2,:);
vx_sol = Z(3,:);
vy_sol = Z(4,:);
lambda2_bar_sol = Z(5,:);
lambda4_bar_sol = Z(6,:);
%% Plots
figure;
subplot(3,2,1);plot(time,x_sol/1000);
ylabel('x, km','fontsize',14);
xlabel('Time, sec','fontsize',14);
hold on; grid on;
title('Optimal Ascent from Flat Moon')
subplot(3,2,2);plot(time,y_sol/1000);
ylabel('y, km','fontsize',14);
xlabel('Time, sec','fontsize',14); hold on; grid on;

%% Page 219

subplot(3,2,3);plot(time,vx_sol/1000);
ylabel('V_x, km/s','fontsize',14);
xlabel('Time, sec','fontsize',14);
hold on; grid on;
subplot(3,2,4);plot(time,vy_sol/1000);
ylabel('V_y, km/s','fontsize',14);
xlabel('Time, sec','fontsize',14);
hold on; grid on;
subplot(3,2,5);plot(time,rad2deg*atan(lambda4_bar_sol));
ylabel('\alpha, deg','fontsize',14);
xlabel('Time, sec','fontsize',14);
grid on; 
hold on;
subplot(3,2,6);plot(time,lambda4_bar_sol);
ylabel('\lambda_4','fontsize',14);
xlabel('Time, sec','fontsize',14);
grid on; hold on;
% Plot of ascent trajectory in y versus x coordinates [km]
figure;
plot(x_sol/1000, y_sol/1000); grid on; axis equal
xlabel('Downrange Position, x, km','fontsize',14)
ylabel('Altitude, y, km','fontsize',14)