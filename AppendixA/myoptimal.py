# Optimal Ascent Problem with Python's solve_bvp
#
# Original script for MATLAB's bvp4c by by Jose J. Guzman and George E. Pollock
#
# Adapted for Python by Nickolai Belakovski
#
# This script uses scipy's solve_bvp to solve the problem of finding the
# optimal ascent trajectory for launch from a flat Moon to a 100 nautical
# mile circular orbit.


import numpy as np
from scipy.integrate import solve_bvp
import matplotlib.pyplot as plt


# Define parameters of the problem
h = 185.2e3  # meters, final altitude (100 nmi circular orbit)
Vc = 1.627e3  # m/s, Circular speed at 100 nmi
g_accel = 1.62  # m/sˆ2, gravitational acceleration of Moon
Thrust2Weight = 3  # Thrust to Weight ratio for Ascent Vehicle, in lunar G's
# ----------------------------------------------------------------------------
# # Boundary Conditions
# ----------------------------------------------------------------------------
# Initial conditions
# Launch from zero altitude with zero initial velocity
xi_d = 0  # meters, initial x-position (_d means 'desired')
yi_d = 0  # meters, initial y-position
Vxi_d = 0  # m/s, initial downrange velocity
Vyi_d = 0  # m/s, initial vertical velocity
# Final conditions
yf_d = h  # meters, final altitude
Vxf_d = Vc  # m/s, final downrange velocity
Vyf_d = 0  # m/s, final vertical velocity
Hf_d = -1


# Note: this function uses g_accel and Thrust2Weight and so must be defined
# after those two variables
def ascent_odes_tf(tau, X, tf):
    #
    # State and Costate Differential Equation Function for the Flat-Moon
    # Optimal Ascent Problem
    #
    #
    # The independent variable here is the nondimensional time, tau, the state
    # vector is X, and the final time, tf, is an unknown parameter that must
    # also be passed to the DE function.
    # Note that the state vector X has components
    # X[0] = x, horizontal component of position
    # X[1] = y, vertical component of position
    # X[2] = Vx, horizontal component of velocity
    # X[3] = Vy, vertical component of velocity
    # X[4] = lambda2
    # X[5] = lambda3
    # X[6] = lambda4
    # Acceleration (F/m) of the Ascent Vehicle, m/s^2
    Acc = Thrust2Weight*g_accel
    # State and Costate differential equations in terms of d/dt:
    x = X[0]  # noqa
    y = X[1]  # noqa
    Vx = X[2]
    Vy = X[3]
    lambda2 = X[4]
    lambda3 = X[5]
    lambda4 = X[6]
    xdot = Vx
    ydot = Vy
    Vxdot = Acc*(-lambda3/np.sqrt(lambda3**2+lambda4**2))
    Vydot = Acc*(-lambda4/np.sqrt(lambda3**2+lambda4**2)) - g_accel
    lambda2_dot = np.zeros(X.shape[1])
    lambda3_dot = np.zeros(X.shape[1])
    lambda4_dot = -lambda2
    # Nondimensionalize time (with tau = t/tf and d/dtau = tf*d/dt). We must
    # multiply each differential equation by tf to convert our derivatives from
    # d/dt to d/dtau.
    dX_dtau = tf*np.array([xdot, ydot, Vxdot, Vydot, lambda2_dot, lambda3_dot, lambda4_dot])
    return dX_dtau


# Note: this function uses the initial and final boundary conditions and so
# must be defined after those variables are defined
def ascent_bcs_tf(Y0, Yf, tf):
    # Boundary Condition Function for the Flat-Moon Optimal Ascent Problem
    # Hamiltonian final needs to be -1
    xi = Y0[0]
    yi = Y0[1]
    Vxi = Y0[2]
    Vyi = Y0[3]
    lambda2i = Y0[4]  # noqa
    lambda3i = Y0[5]  # noqa
    lambda4i = Y0[6]  # noqa

    xf = Yf[0]  # noqa
    yf = Yf[1]
    Vxf = Yf[2]
    Vyf = Yf[3]
    lambda2f = Yf[4]
    lambda3f = Yf[5]
    lambda4f = Yf[6]

    costhetaf = -lambda3f/np.sqrt(lambda3f**2 + lambda4f**2)
    sinthetaf = -lambda4f/np.sqrt(lambda3f**2 + lambda4f**2)
    Acc = Thrust2Weight*g_accel
    Hf = lambda2f*Vxf + lambda3f * Acc * costhetaf + lambda4f * Acc * sinthetaf - lambda4f*g_accel
    PSI = np.array([
        xi - xi_d,
        yi - yi_d,
        Vxi - Vxi_d,
        Vyi - Vyi_d,
        yf - yf_d,
        Vxf - Vxf_d,
        Vyf - Vyf_d,
        Hf - Hf_d
    ])
    return PSI


# ----------------------------------------------------------------------------
# # Initial Guesses
# ----------------------------------------------------------------------------
# initial time
t0 = 0
# list initial conditions in yinit, use zero if unknown
yinit = [xi_d, yi_d, Vxi_d, Vyi_d, 0, 1, 1]  # guess for initial state and costate variables
tf_guess = 700  # sec, initial guess for final time

# # Page 218

# Because the final time is free, we must parameterize the problem by
# the unknown final time, tf. Create a nondimensional time vector,
# tau, with Nt linearly spaced elements. (tau = time/tf) We will pass the
# unknown final time to solve_bvp as an unknown parameter and the code will
# attempt to solve for the actual final time as it solves our TPBVP.
Nt = 41
tau = np.linspace(0, 1, Nt)  # nondimensional time vector
# Create an initial guess of the solution
# Python/scipy do not have an equivalent to bvpinit, so we make our own
# bvpinit basically just stores the mesh and the initial guess inside a struct
# after replicating the initial guess for each point on the mesh. solve_bvp
# takes the mesh as a parameter, so all we need to do is replicate the initial
# guess for each point of the original mesh and make sure the result is nxm, as
# solve_bvp expects, where n is the number of states and m is the mesh size
solinit = np.array([yinit for _ in range(Nt)]).T
# ----------------------------------------------------------------------------
# # Solution
# ----------------------------------------------------------------------------
# Call solve_bvp to solve the TPBVP. Point the solver to the functions
# containing the differential equations and the boundary conditions and
# provide it with the initial guess of the solution.
sol = solve_bvp(ascent_odes_tf, ascent_bcs_tf, tau, solinit, np.array([tf_guess]))
# Extract the final time from the solution:
tf = sol.p[0]
# # Evaluate the solution at all times in the nondimensional time vector tau
# # and store the state variables in the matrix Z.
Z = sol.sol(tau)
# # Convert back to dimensional time for plotting
time = t0 + tau*(tf-t0)
# # Extract the solution for each state variable from the matrix Z:
x_sol = Z[0, :]
y_sol = Z[1, :]
vx_sol = Z[2, :]
vy_sol = Z[3, :]
lambda2_sol = Z[4, :]
lambda3_sol = Z[5, :]
lambda4_sol = Z[6, :]
x = [vx_sol[0] * time[0]]
for i in range(1, len(time)):
    x.append(x[-1] + (vx_sol[i]+vx_sol[i-1])/2*(time[i]-time[i-1]))

# # Plots
fig = plt.figure('Optimal Ascent from Flat Moon')

# Indices are modifed as compared with the code in the book so that the
# plots appears in the same order as in the book
ax = fig.add_subplot(323)
ax.plot(time, x_sol/1000)
ax.set_ylabel('x, km')
ax.set_xlabel('Time, sec')
ax.grid()

ax = fig.add_subplot(324)
ax.plot(time, y_sol/1000)
ax.set_ylabel('y, km')
ax.set_xlabel('Time, sec')
ax.grid()

ax = fig.add_subplot(325)
ax.plot(time, vx_sol/1000)
ax.set_ylabel('V_x, km')
ax.set_xlabel('Time, sec')
ax.grid()

ax = fig.add_subplot(326)
ax.plot(time, vy_sol/1000)
ax.set_ylabel('V_y, km')
ax.set_xlabel('Time, sec')
ax.grid()

ax = fig.add_subplot(321)
ax.plot(time, np.rad2deg(np.arctan2(-lambda4_sol, -lambda3_sol)))
ax.set_ylabel(r'$\alpha$ deg')
ax.set_xlabel('Time, sec')
ax.grid()

ax = fig.add_subplot(322)
ax.plot(x_sol/1000, y_sol/1000)
ax.set_ylabel('y, km')
ax.set_xlabel('x, km')
ax.grid()
