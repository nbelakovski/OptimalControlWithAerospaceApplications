# Optimal Ascent Problem with MATLAB’s bvp4c
#
# by Jose J. Guzman, George E. Pollock and Peter J. Edelman
#
# This script uses MATLAB’s bvp4c to solve the problem of finding the
# optimal ascent trajectory for launch from a flat Earth to a 180 kilometer
# circular orbit with the inclusion of the effects from atmospheric drag
# and linearly time-varying mass.
# In addition to this script, we use two functions:
# one to provide the differential equations and another that gives the
# boundary conditions:
#
# This file: OptimalAscent.m
# State and Costate Equations: ascent_odes_tf.m
# Boundary Conditions: ascent_bcs_tf.m
#

import numpy as np
from scipy.integrate import solve_bvp
import matplotlib.pyplot as plt

# Define parameters of the problem
# pass these parameters to the DE and BC functions
h = 180000 # meters, final altitude (180 km circular orbit)
Vc = np.sqrt(3.9860044e5/(6378.14+h/1000))*1000 # m/s, Circular speed at 180 km
g_accel = 9.80665 # m/sˆ2, gravitational acceleration of Earth
f = 2.1e6 # N, thrust of the first stage of Titan II Rocket
h_scale = 8440 # m, atmospheric scale-height
beta = h/h_scale #[Nondim], constant used to reduce EOM equations
rhoref = 1.225 # kg/mˆ3, reference density
A = 7.069 # mˆ2, aerodynamic reference area (cross-sectional area)
#---------------------------------------------------------------------------
## Boundary Conditions
#---------------------------------------------------------------------------
# pass these BCs to the BC function
# Initial conditions
# Launch from zero altitude with zero initial velocity
# All Boundary Conditions are nondimensional
xbar0 = 0 # initial x-position
ybar0 = 0 # initial y-position
Vxbar0 = 0 # initial downrange velocity
Vybar0 = 0 # initial vertical velocity
# Final conditions
ybarf = h/h # final altitude
Vxbarf = Vc/Vc # final downrange velocity
Vybarf = 0 # final vertical velocity
## Parameters for the NO DRAG and CONSTANT MASS case
# Solve TPBVP without drag (C_D = 0) and constant mass (mdot = 0)
m0 = 60880 # kg, average mass of a Titan II Rocket
CD = 0 # Drag coefficient set to zero for no drag cases
mdot = 0 # kg/s, mass flow rate for the constant mass case
# eta - a constant composed of the reference density, coefficient
# of drag and the aerodynamic reference area. It is only used to simplify
# the drag expressions in the EOMs.
eta = rhoref*CD*A/2
#---------------------------------------------------------------------------
## Initial Guesses
#---------------------------------------------------------------------------
# initial time
t0 = 0
# list initial conditions in yinit, use zero if unknown
# guess for initial state and costate variables
yinit = np.array([xbar0, ybar0, Vxbar0, Vybar0, 0, -1, 0])
tf_guess = 248.897; # sec, initial guess for final time
# Because the final time is free, we must parameterize the problem by
# the unknown final time, tf. Create a nondimensional time vector,
# tau, with Nt linearly spaced elements. (tau = time/tf) We will pass the
# unknown final time to bvp4c as an unknown parameter and the code will
# attempt to solve for the actual final time as it solves our TPBVP.
# Nt is the number of points that the TPBVP will be discretized into. The
# larger Nt is, the more accurate your solution. However be aware that
# if Nt is too large, the solver may not be able to compute a solution
# using its algorithm.
Nt = 80
tau = np.linspace(0,1,Nt) # nondimensional time vector
# Create an initial guess of the solution using the MATLAB function
# bvpinit, which requires as inputs the (nondimensional) time vector,
# initial states (or guesses, as applicable), and the guess for the final
# time. The initial guess of the solution is stored in the structure
# solinit.
solinit = np.array([yinit for _ in range(Nt)]).T
#---------------------------------------------------------------------------
## Solution for the NO DRAG and CONSTANT MASS case
#---------------------------------------------------------------------------
def ascent_odes_tf(tau, X, p):
    xbar = X[0]
    ybar = X[1]
    Vxbar = X[2]
    Vybar = X[3]
    lambda_2_bar = X[4]
    lambda_3_bar = X[5]
    lambda_4_bar = X[6]
    tf = p[0]
    m = m0 - abs(mdot)*tau*tf
    xbardot = Vxbar*Vc/h
    ybardot = Vybar*Vc/h
    Vxbardot = f/Vc*(-lambda_3_bar/np.sqrt(lambda_3_bar**2 + lambda_4_bar**2)) \
        - eta*np.exp(-ybar*beta)*Vxbar*np.sqrt(Vxbar**2 + Vybar**2)*Vc/m
    Vybardot = f/Vc*(-lambda_4_bar/np.sqrt(lambda_3_bar**2 + lambda_4_bar**2)) \
        - eta*np.exp(-ybar*beta)*Vybar*np.sqrt(Vxbar**2 + Vybar**2)*Vc/m - g_accel/Vc
    if all(np.sqrt(Vxbar**2 + Vybar**2)) == 0:
        lambda_2_bar = np.zeros(xbar.shape[0])
        lambda_3_bar = np.zeros(xbar.shape[0])
        lambda_4_bar = -lambda_2_bar*Vc/h
    else:
        lambda_2_bar = \
            -(lambda_3_bar*Vxbar+lambda_4_bar*Vybar)*eta*beta*np.exp(-ybar*beta)*np.sqrt(Vxbar**2+Vybar**2)*Vc/m
        lambda_3_bar = eta*np.exp(-ybar*beta)*Vc*(lambda_3_bar*(2*Vxbar**2+Vybar**2) \
            + lambda_4_bar*Vxbar*Vybar)/np.sqrt(Vxbar**2+Vybar**2)/m
        lambda_4_bar = -lambda_2_bar*Vc/h+eta*np.exp(-ybar*beta)*Vc*(lambda_4_bar*(Vxbar**2
            + 2*Vybar**2) \
            + lambda_3_bar*Vxbar*Vybar)/np.sqrt(Vxbar**2+Vybar**2)/m

    dX_dtau = tf * np.array([xbardot, ybardot, Vxbardot, Vybardot, lambda_2_bar, lambda_3_bar, lambda_4_bar])
    return dX_dtau

def ascent_bcs_tf(ya, yb, p):
    xbari = ya[0]
    ybari = ya[1]
    Vxbari = ya[2]
    Vybari = ya[3]
    lambda_2_bari = ya[4]
    lambda_3_bari = ya[5]
    lambda_4_bari = ya[6]
    xbarff = yb[0]  # use two f's to distinguish from the desired final
    ybarff = yb[1]
    Vxbarff = yb[2]
    Vybarff = yb[3]
    lambda_2_barff = yb[4]
    lambda_3_barff = yb[5]
    lambda_4_barff = yb[6]
    tf = p[0]
    mf = m0 - abs(mdot)*tf
    Hf = (-np.sqrt(lambda_3_barff**2+lambda_4_barff**2)*f/mf/Vc 
            -(lambda_3_barff*Vxbarff)*eta*np.exp(-beta)*np.sqrt(Vxbarff**2)*Vc/mf-lambda_4_barff*g_accel/Vc)*tf + 1
    return np.array([
        xbari - xbar0,
        ybari - ybar0,
        Vxbari - Vxbari,
        Vybari - Vybari,
        ybarff - ybarf,
        Vxbarff - Vxbarf,
        Vybarff - Vybarf,
        Hf
    ])
    


# Call bvp4c to solve the TPBVP. Point the solver to the functions
# containing the differential equations and the boundary conditions and
# provide it with the initial guess of the solution.
# sol = bvp4c(@ascent_odes_tf, @ascent_bcs_tf, solinit);
sol = solve_bvp(ascent_odes_tf, ascent_bcs_tf, tau, solinit, np.array([tf_guess]), verbose=2)
import sys
sys.exit(0)
# Extract the final time from the solution:
tf = sol.p[0]
# Evaluate the solution at all times in the nondimensional time vector tau
# and store the state variables in the matrix Z.
Z = sol.sol(tau)
# Convert back to dimensional time for plotting
time = t0 + tau * (tf-t0)
# Extract the solution for each state variable from the matrix Z and
# convert them back into dimensional units by multiplying each by their
# respective scaling constants.
x_sol = Z[0,:]*h/1000
y_sol = Z[1,:]*h/1000
vx_sol = Z[2,:]*Vc/1000
vy_sol = Z[3,:]*Vc/1000
lambda2_bar_sol = Z[4,:]
lambda3_bar_sol = Z[5,:]
lambda4_bar_sol = Z[6,:]
## Parameters for VARYING MASS and NO DRAG case
m0 = 117020; # Initial mass of Titan II rocket (kg)
mdot = (117020-4760)/139; # Mass flow rate (kg/s)
delta_tf = 115; # Amount subtracted from final time of constant mass
# case
#---------------------------------------------------------------------------
## Solution for the VARYING MASS and NO DRAG case
#---------------------------------------------------------------------------
# Copy initial guess for the drag solution into a new structure named
# solinit_mass
solinit_mass = Z
# Save the time histories of the 7 state and costate variables from the NO
# DRAG, CONSTANT MASS solution in the structure of the initial guess for
# the VARYING MASS, NO DRAG case
# solinit_mass.y = Z
# Save the final time of the NO DRAG, CONSTANT MASS solution and use it as
# the guess for the final time for the VARYING MASS, NO DRAG case. Also
# subtract delta_tf from this guess as described before
# solinit_mass.parameters(1) = tf-delta_tf
# Run bvp4c for the VARYING MASS, NO DRAG
sol_mass = solve_bvp(ascent_odes_tf,ascent_bcs_tf,tau, solinit_mass, np.array([tf-delta_tf]))
# Evaluate the solution at all times in the nondimensional time vector tau
# and store the state variables in the matrix Z_mass.
Z_mass = sol_mass.sol(tau)
# Extract the final time from the solution with VARYING MASS, NO DRAG:
tf_mass = sol_mass.p[0]
# Convert back to dimensional time for plotting
time_mass = t0+tau*(tf_mass-t0)
# Extract the solution for each state variable from the matrix Z_mass and
# convert them back into dimensional units by multiplying each by their
# respective scaling constants.
x_sol_mass = Z_mass[0,:]*h/1000
y_sol_mass = Z_mass[1,:]*h/1000
vx_sol_mass = Z_mass[2,:]*Vc/1000
vy_sol_mass = Z_mass[3,:]*Vc/1000
lambda2_bar_sol_mass = Z_mass[4,:]
lambda3_bar_sol_mass = Z_mass[5,:]
lambda4_bar_sol_mass = Z_mass[6,:]
## Parameters for the VARYING MASS AND DRAG case
CD = .5; # Drag coefficient
eta = rhoref*CD*A/2; # Update eta, since CD is now nonzero
#---------------------------------------------------------------------------
## Solution for the VARYING MASS AND DRAG case
#---------------------------------------------------------------------------
# Copy initial guess for the VARYING MASS AND DRAG solution into a new
# structure named solinit_mass_drag
solinit_mass_drag = Z_mass
# Save the time histories of the 7 state and costate variables from the
# VARYING MASS, NO DRAG solution in the structure of the initial guess for
# the VARYING MASS AND DRAG case
# solinit_mass_drag.y = Z_mass
# Save the final time of the VARYING MASS, NO DRAG solution and use it as
# the guess for the final time for the VARYING MASS AND DRAG case
# solinit_mass_drag.parameters(1) = tf_mass
# Run bvp4c for the drag case
sol_mass_drag = solve_bvp(ascent_odes_tf,ascent_bcs_tf, tau, solinit_mass_drag, np.array([tf_mass]))
# Evaluate the solution at all times in the nondimensional time vector tau
# and store the state variables in the matrix Z_mass_drag.
Z_mass_drag = sol_mass_drag.sol(tau)
# Extract the final time from the solution:
tf_mass_drag = sol_mass_drag.p[0]
# Convert back to dimensional time for plotting
time_mass_drag = t0+tau*(tf_mass_drag-t0)
# Extract the solution for each state variable from the matrix Z_mass_drag
# and convert them back into dimensional units by multiplying each by their
# respective scaling constants.
x_sol_mass_drag = Z_mass_drag[0,:]*h/1000
y_sol_mass_drag = Z_mass_drag[1,:]*h/1000
vx_sol_mass_drag = Z_mass_drag[2,:]*Vc/1000
vy_sol_mass_drag = Z_mass_drag[3,:]*Vc/1000
lambda2_bar_sol_mass_drag = Z_mass_drag[4,:]
lambda3_bar_sol_mass_drag = Z_mass_drag[5,:]
lambda4_bar_sol_mass_drag = Z_mass_drag[6,:]
## Plot the solutions
fig = plt.figure()
ax = fig.add_subplot(221)
ax.plot(time_mass_drag, x_sol_mass_drag,'k')
ax.set_xlabel('Time [s]')
ax.set_ylabel('x [km]')
ax.set_xlim(t0, tf_mass_drag)
ax = fig.add_subplot(222)
ax.plot(time_mass_drag,y_sol_mass_drag,'k')
ax.set_xlabel('Time [s]')
ax.set_ylabel('y [km]')
ax.set_xlim(t0, tf_mass_drag)
ax = fig.add_subplot(223)
ax.plot(time_mass_drag,vx_sol_mass_drag,'k')
ax.set_xlabel('Time [s]')
ax.set_ylabel('V_x [km/s]')
ax.set_xlim(t0, tf_mass_drag)
ax = fig.add_subplot(224)
ax.plot(time_mass_drag,vy_sol_mass_drag,'k')
ax.set_xlabel('Time [s]')
ax.set_ylabel('V_y [km/s]')
ax.set_xlim(t0, tf_mass_drag)
fig = plt.figure()
plt.plot(time_mass_drag, np.arctan2(lambda4_bar_sol_mass_drag, lambda3_bar_sol_mass_drag),'k')
ax.set_xlabel('Time [s]')
ax.set_ylabel('\theta[deg]')
ax.set_xlim(t0, tf_mass_drag)
fig = plt.figure()
plt.plot(x_sol_mass_drag,y_sol_mass_drag,'k')
ax.set_xlabel('Downrange Position, x [km]')
ax.set_ylabel('Altitude, y [km]')
ax.set_xlim(x_sol_mass_drag[0], x_sol_mass_drag[-1])
ax.set_ylim(0, 200)
