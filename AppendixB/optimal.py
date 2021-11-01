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
yinit = np.array([xbar0, ybar0, Vxbar0, Vybar0, 0, -1, 0], dtype=np.float64)
tf_guess = 250 # sec, initial guess for final time
# Because the final time is free, we must parameterize the problem by
# the unknown final time, tf. Create a nondimensional time vector,
# tau, with Nt linearly spaced elements. (tau = time/tf) We will pass the
# unknown final time to bvp4c as an unknown parameter and the code will
# attempt to solve for the actual final time as it solves our TPBVP.
# Nt is the number of points that the TPBVP will be discretized into. The
# larger Nt is, the more accurate your solution. However be aware that
# if Nt is too large, the solver may not be able to compute a solution
# using its algorithm.
Nt = 162
tau = np.linspace(0,1,Nt) # nondimensional time vector
# Create an initial guess of the solution using the MATLAB function
# bvpinit, which requires as inputs the (nondimensional) time vector,
# initial states (or guesses, as applicable), and the guess for the final
# time. The initial guess of the solution is stored in the structure
# solinit.
solinit = np.array([yinit for _ in range(Nt)]).T
solinit[0, -1] = 5
solinit[1, -1] = 1
solinit[2, -1] = 1
solinit[3, -1] = 0
solinit[4, -1] = -0.095
solinit[5, -1] = -0.81
solinit[6, -1] = .2311
# solinit[0] = [0,1.701553e-04,6.815298e-04,1.535491e-03,2.733409e-03,4.276663e-03,6.166633e-03,8.404707e-03,1.099227e-02,1.393073e-02,1.722148e-02,2.086592e-02,2.486547e-02,2.922154e-02,3.393554e-02,3.900889e-02,4.444302e-02,5.023937e-02,5.639935e-02,6.292441e-02,6.981598e-02,7.707550e-02,8.470443e-02,9.270420e-02,1.010763e-01,1.098221e-01,1.189431e-01,1.284408e-01,1.383166e-01,1.485719e-01,1.592084e-01,1.702273e-01,1.816301e-01,1.934184e-01,2.055936e-01,2.181571e-01,2.311105e-01,2.444551e-01,2.581925e-01,2.723241e-01,2.868513e-01,3.017757e-01,3.170987e-01,3.328217e-01,3.489462e-01,3.654737e-01,3.824055e-01,3.997433e-01,4.174883e-01,4.356420e-01,4.542059e-01,4.731814e-01,4.925699e-01,5.123728e-01,5.325916e-01,5.532277e-01,5.742824e-01,5.957572e-01,6.176534e-01,6.399724e-01,6.627156e-01,6.858844e-01,7.094801e-01,7.335041e-01,7.579577e-01,7.828422e-01,8.081589e-01,8.339092e-01,8.600943e-01,8.867156e-01,9.137742e-01,9.412715e-01,9.692087e-01,9.975870e-01,1.026408e+00,1.055672e+00,1.085381e+00,1.115535e+00,1.146137e+00,1.177187e+00,1.208687e+00,1.229936e+00,1.251387e+00,1.273038e+00,1.294890e+00,1.316944e+00,1.339199e+00,1.361656e+00,1.384316e+00,1.407179e+00,1.441853e+00,1.476985e+00,1.512576e+00,1.548626e+00,1.585136e+00,1.622108e+00,1.659542e+00,1.697438e+00,1.735799e+00,1.774624e+00,1.813915e+00,1.853671e+00,1.893894e+00,1.934585e+00,1.975744e+00,2.017372e+00,2.059469e+00,2.102036e+00,2.145073e+00,2.188582e+00,2.232562e+00,2.277014e+00,2.321939e+00,2.367336e+00,2.413207e+00,2.459551e+00,2.506369e+00,2.553662e+00,2.601429e+00,2.649670e+00,2.698387e+00,2.747579e+00,2.797246e+00,2.847388e+00,2.898006e+00,2.949099e+00,3.000668e+00,3.052713e+00,3.105233e+00,3.158228e+00,3.211699e+00,3.265645e+00,3.320066e+00,3.374962e+00,3.430332e+00,3.486177e+00,3.542496e+00,3.599289e+00,3.656556e+00,3.714295e+00,3.772507e+00,3.831192e+00,3.890349e+00,3.949976e+00,4.010075e+00,4.070644e+00,4.131683e+00,4.193191e+00,4.255168e+00,4.317612e+00,4.380524e+00,4.443902e+00,4.507746e+00,4.572055e+00,4.636829e+00,4.702066e+00,4.767766e+00,4.833927e+00,4.900550e+00,4.967633e+00,5.035175e+00,5.103175e+00]
# solinit[1] = [0,9.848625e-05,3.930108e-04,8.821598e-04,1.564502e-03,2.438590e-03,3.502957e-03,4.756120e-03,6.196578e-03,7.822812e-03,9.633285e-03,1.162644e-02,1.380070e-02,1.615448e-02,1.868615e-02,2.139410e-02,2.427666e-02,2.733217e-02,3.055894e-02,3.395525e-02,3.751937e-02,4.124956e-02,4.514403e-02,4.920100e-02,5.341864e-02,5.779513e-02,6.232861e-02,6.701719e-02,7.185897e-02,7.685204e-02,8.199444e-02,8.728421e-02,9.271935e-02,9.829786e-02,1.040177e-01,1.098768e-01,1.158731e-01,1.220044e-01,1.282687e-01,1.346638e-01,1.411875e-01,1.478375e-01,1.546118e-01,1.615080e-01,1.685238e-01,1.756570e-01,1.829052e-01,1.902661e-01,1.977372e-01,2.053163e-01,2.130007e-01,2.207882e-01,2.286763e-01,2.366623e-01,2.447438e-01,2.529183e-01,2.611830e-01,2.695356e-01,2.779732e-01,2.864933e-01,2.950931e-01,3.037700e-01,3.125212e-01,3.213439e-01,3.302353e-01,3.391927e-01,3.482132e-01,3.572938e-01,3.664318e-01,3.756242e-01,3.848680e-01,3.941603e-01,4.034981e-01,4.128783e-01,4.222980e-01,4.317539e-01,4.412431e-01,4.507624e-01,4.603087e-01,4.698787e-01,4.794693e-01,4.858729e-01,4.922833e-01,4.986994e-01,5.051204e-01,5.115452e-01,5.179729e-01,5.244025e-01,5.308330e-01,5.372634e-01,5.469066e-01,5.565441e-01,5.661723e-01,5.757878e-01,5.853873e-01,5.949673e-01,6.045242e-01,6.140545e-01,6.235548e-01,6.330216e-01,6.424512e-01,6.518400e-01,6.611846e-01,6.704812e-01,6.797263e-01,6.889163e-01,6.980474e-01,7.071160e-01,7.161185e-01,7.250511e-01,7.339101e-01,7.426919e-01,7.513927e-01,7.600087e-01,7.685362e-01,7.769716e-01,7.853109e-01,7.935504e-01,8.016864e-01,8.097151e-01,8.176327e-01,8.254353e-01,8.331193e-01,8.406808e-01,8.481159e-01,8.554210e-01,8.625922e-01,8.696256e-01,8.765175e-01,8.832641e-01,8.898616e-01,8.963062e-01,9.025940e-01,9.087214e-01,9.146844e-01,9.204794e-01,9.261025e-01,9.315501e-01,9.368182e-01,9.419033e-01,9.468015e-01,9.515090e-01,9.560223e-01,9.603375e-01,9.644510e-01,9.683591e-01,9.720581e-01,9.755444e-01,9.788142e-01,9.818640e-01,9.846902e-01,9.872891e-01,9.896572e-01,9.917909e-01,9.936867e-01,9.953409e-01,9.967502e-01,9.979110e-01,9.988198e-01,9.994732e-01,9.998677e-01,1]
# solinit[2] = [0,4.991099e-03,1.000219e-02,1.503336e-02,2.008467e-02,2.515619e-02,3.024801e-02,3.536018e-02,4.049277e-02,4.564585e-02,5.081949e-02,5.601375e-02,6.122869e-02,6.646436e-02,7.172084e-02,7.699817e-02,8.229641e-02,8.761561e-02,9.295583e-02,9.831710e-02,1.036995e-01,1.091030e-01,1.145277e-01,1.199737e-01,1.254409e-01,1.309294e-01,1.364392e-01,1.419704e-01,1.475229e-01,1.530969e-01,1.586923e-01,1.643091e-01,1.699474e-01,1.756071e-01,1.812883e-01,1.869909e-01,1.927150e-01,1.984606e-01,2.042276e-01,2.100160e-01,2.158258e-01,2.216571e-01,2.275097e-01,2.333836e-01,2.392789e-01,2.451954e-01,2.511331e-01,2.570920e-01,2.630720e-01,2.690730e-01,2.750951e-01,2.811380e-01,2.872018e-01,2.932864e-01,2.993917e-01,3.055176e-01,3.116639e-01,3.178307e-01,3.240177e-01,3.302249e-01,3.364522e-01,3.426994e-01,3.489664e-01,3.552531e-01,3.615592e-01,3.678848e-01,3.742295e-01,3.805933e-01,3.869760e-01,3.933773e-01,3.997972e-01,4.062354e-01,4.126917e-01,4.191660e-01,4.256580e-01,4.321674e-01,4.386941e-01,4.452379e-01,4.517985e-01,4.583756e-01,4.649690e-01,4.693736e-01,4.737852e-01,4.782038e-01,4.826293e-01,4.870616e-01,4.915006e-01,4.959462e-01,5.003985e-01,5.048571e-01,5.115571e-01,5.182711e-01,5.249988e-01,5.317400e-01,5.384942e-01,5.452612e-01,5.520407e-01,5.588322e-01,5.656354e-01,5.724500e-01,5.792756e-01,5.861118e-01,5.929584e-01,5.998148e-01,6.066807e-01,6.135557e-01,6.204395e-01,6.273315e-01,6.342315e-01,6.411391e-01,6.480537e-01,6.549750e-01,6.619026e-01,6.688360e-01,6.757749e-01,6.827187e-01,6.896672e-01,6.966197e-01,7.035760e-01,7.105355e-01,7.174978e-01,7.244625e-01,7.314292e-01,7.383973e-01,7.453664e-01,7.523362e-01,7.593061e-01,7.662757e-01,7.732446e-01,7.802123e-01,7.871784e-01,7.941424e-01,8.011038e-01,8.080623e-01,8.150174e-01,8.219687e-01,8.289157e-01,8.358580e-01,8.427952e-01,8.497268e-01,8.566524e-01,8.635716e-01,8.704840e-01,8.773892e-01,8.842867e-01,8.911761e-01,8.980571e-01,9.049292e-01,9.117921e-01,9.186454e-01,9.254887e-01,9.323215e-01,9.391436e-01,9.459546e-01,9.527541e-01,9.595418e-01,9.663172e-01,9.730802e-01,9.798302e-01,9.865671e-01,9.932905e-01,1]
# solinit[3] = [0,2.883520e-03,5.746441e-03,8.588515e-03,1.140949e-02,1.420911e-02,1.698712e-02,1.974326e-02,2.247727e-02,2.518887e-02,2.787780e-02,3.054380e-02,3.318657e-02,3.580585e-02,3.840135e-02,4.097279e-02,4.351989e-02,4.604234e-02,4.853986e-02,5.101216e-02,5.345892e-02,5.587986e-02,5.827465e-02,6.064301e-02,6.298461e-02,6.529914e-02,6.758628e-02,6.984572e-02,7.207712e-02,7.428017e-02,7.645454e-02,7.859989e-02,8.071589e-02,8.280220e-02,8.485849e-02,8.688441e-02,8.887961e-02,9.084376e-02,9.277649e-02,9.467746e-02,9.654631e-02,9.838269e-02,1.001862e-01,1.019566e-01,1.036934e-01,1.053962e-01,1.070648e-01,1.086987e-01,1.102976e-01,1.118611e-01,1.133888e-01,1.148804e-01,1.163354e-01,1.177535e-01,1.191343e-01,1.204775e-01,1.217826e-01,1.230493e-01,1.242771e-01,1.254658e-01,1.266149e-01,1.277240e-01,1.287928e-01,1.298209e-01,1.308078e-01,1.317533e-01,1.326569e-01,1.335182e-01,1.343370e-01,1.351127e-01,1.358450e-01,1.365335e-01,1.371779e-01,1.377779e-01,1.383329e-01,1.388427e-01,1.393069e-01,1.397251e-01,1.400970e-01,1.404222e-01,1.407004e-01,1.408596e-01,1.409976e-01,1.411143e-01,1.412097e-01,1.412836e-01,1.413360e-01,1.413667e-01,1.413756e-01,1.413628e-01,1.413022e-01,1.411920e-01,1.410318e-01,1.408214e-01,1.405603e-01,1.402483e-01,1.398851e-01,1.394705e-01,1.390042e-01,1.384859e-01,1.379153e-01,1.372923e-01,1.366165e-01,1.358878e-01,1.351058e-01,1.342705e-01,1.333815e-01,1.324387e-01,1.314419e-01,1.303910e-01,1.292856e-01,1.281258e-01,1.269112e-01,1.256419e-01,1.243175e-01,1.229381e-01,1.215035e-01,1.200136e-01,1.184684e-01,1.168676e-01,1.152113e-01,1.134994e-01,1.117318e-01,1.099086e-01,1.080296e-01,1.060948e-01,1.041043e-01,1.020580e-01,9.995601e-02,9.779827e-02,9.558484e-02,9.331576e-02,9.099110e-02,8.861091e-02,8.617527e-02,8.368428e-02,8.113801e-02,7.853659e-02,7.588013e-02,7.316876e-02,7.040260e-02,6.758182e-02,6.470657e-02,6.177701e-02,5.879331e-02,5.575567e-02,5.266428e-02,4.951933e-02,4.632105e-02,4.306964e-02,3.976534e-02,3.640838e-02,3.299901e-02,2.953748e-02,2.602405e-02,2.245898e-02,1.884255e-02,1.517504e-02,1.145673e-02,7.687920e-03,3.868909e-03,0]
# solinit[4] = [-9.498222e-02,-9.498222e-02,-9.498222e-02,-9.498222e-02,-9.498222e-02,-9.498222e-02,-9.498222e-02,-9.498222e-02,-9.498222e-02,-9.498222e-02,-9.498222e-02,-9.498222e-02,-9.498222e-02,-9.498222e-02,-9.498222e-02,-9.498222e-02,-9.498222e-02,-9.498222e-02,-9.498222e-02,-9.498222e-02,-9.498222e-02,-9.498222e-02,-9.498222e-02,-9.498222e-02,-9.498222e-02,-9.498222e-02,-9.498222e-02,-9.498222e-02,-9.498222e-02,-9.498222e-02,-9.498222e-02,-9.498222e-02,-9.498222e-02,-9.498222e-02,-9.498222e-02,-9.498222e-02,-9.498222e-02,-9.498222e-02,-9.498222e-02,-9.498222e-02,-9.498222e-02,-9.498222e-02,-9.498222e-02,-9.498222e-02,-9.498222e-02,-9.498222e-02,-9.498222e-02,-9.498222e-02,-9.498222e-02,-9.498222e-02,-9.498222e-02,-9.498222e-02,-9.498222e-02,-9.498222e-02,-9.498222e-02,-9.498222e-02,-9.498222e-02,-9.498222e-02,-9.498222e-02,-9.498222e-02,-9.498222e-02,-9.498222e-02,-9.498222e-02,-9.498222e-02,-9.498222e-02,-9.498222e-02,-9.498222e-02,-9.498222e-02,-9.498222e-02,-9.498222e-02,-9.498222e-02,-9.498222e-02,-9.498222e-02,-9.498222e-02,-9.498222e-02,-9.498222e-02,-9.498222e-02,-9.498222e-02,-9.498222e-02,-9.498222e-02,-9.498222e-02,-9.498222e-02,-9.498222e-02,-9.498222e-02,-9.498222e-02,-9.498222e-02,-9.498222e-02,-9.498222e-02,-9.498222e-02,-9.498222e-02,-9.498222e-02,-9.498222e-02,-9.498222e-02,-9.498222e-02,-9.498222e-02,-9.498222e-02,-9.498222e-02,-9.498222e-02,-9.498222e-02,-9.498222e-02,-9.498222e-02,-9.498222e-02,-9.498222e-02,-9.498222e-02,-9.498222e-02,-9.498222e-02,-9.498222e-02,-9.498222e-02,-9.498222e-02,-9.498222e-02,-9.498222e-02,-9.498222e-02,-9.498222e-02,-9.498222e-02,-9.498222e-02,-9.498222e-02,-9.498222e-02,-9.498222e-02,-9.498222e-02,-9.498222e-02,-9.498222e-02,-9.498222e-02,-9.498222e-02,-9.498222e-02,-9.498222e-02,-9.498222e-02,-9.498222e-02,-9.498222e-02,-9.498222e-02,-9.498222e-02,-9.498222e-02,-9.498222e-02,-9.498222e-02,-9.498222e-02,-9.498222e-02,-9.498222e-02,-9.498222e-02,-9.498222e-02,-9.498222e-02,-9.498222e-02,-9.498222e-02,-9.498222e-02,-9.498222e-02,-9.498222e-02,-9.498222e-02,-9.498222e-02,-9.498222e-02,-9.498222e-02,-9.498222e-02,-9.498222e-02,-9.498222e-02,-9.498222e-02,-9.498222e-02,-9.498222e-02,-9.498222e-02,-9.498222e-02,-9.498222e-02,-9.498222e-02,-9.498222e-02,-9.498222e-02,-9.498222e-02,-9.498222e-02]
# solinit[5] = [-8.100355e-01,-8.100355e-01,-8.100355e-01,-8.100355e-01,-8.100355e-01,-8.100355e-01,-8.100355e-01,-8.100355e-01,-8.100355e-01,-8.100355e-01,-8.100355e-01,-8.100355e-01,-8.100355e-01,-8.100355e-01,-8.100355e-01,-8.100355e-01,-8.100355e-01,-8.100355e-01,-8.100355e-01,-8.100355e-01,-8.100355e-01,-8.100355e-01,-8.100355e-01,-8.100355e-01,-8.100355e-01,-8.100355e-01,-8.100355e-01,-8.100355e-01,-8.100355e-01,-8.100355e-01,-8.100355e-01,-8.100355e-01,-8.100355e-01,-8.100355e-01,-8.100355e-01,-8.100355e-01,-8.100355e-01,-8.100355e-01,-8.100355e-01,-8.100355e-01,-8.100355e-01,-8.100355e-01,-8.100355e-01,-8.100355e-01,-8.100355e-01,-8.100355e-01,-8.100355e-01,-8.100355e-01,-8.100355e-01,-8.100355e-01,-8.100355e-01,-8.100355e-01,-8.100355e-01,-8.100355e-01,-8.100355e-01,-8.100355e-01,-8.100355e-01,-8.100355e-01,-8.100355e-01,-8.100355e-01,-8.100355e-01,-8.100355e-01,-8.100355e-01,-8.100355e-01,-8.100355e-01,-8.100355e-01,-8.100355e-01,-8.100355e-01,-8.100355e-01,-8.100355e-01,-8.100355e-01,-8.100355e-01,-8.100355e-01,-8.100355e-01,-8.100355e-01,-8.100355e-01,-8.100355e-01,-8.100355e-01,-8.100355e-01,-8.100355e-01,-8.100355e-01,-8.100355e-01,-8.100355e-01,-8.100355e-01,-8.100355e-01,-8.100355e-01,-8.100355e-01,-8.100355e-01,-8.100355e-01,-8.100355e-01,-8.100355e-01,-8.100355e-01,-8.100355e-01,-8.100355e-01,-8.100355e-01,-8.100355e-01,-8.100355e-01,-8.100355e-01,-8.100355e-01,-8.100355e-01,-8.100355e-01,-8.100355e-01,-8.100355e-01,-8.100355e-01,-8.100355e-01,-8.100355e-01,-8.100355e-01,-8.100355e-01,-8.100355e-01,-8.100355e-01,-8.100355e-01,-8.100355e-01,-8.100355e-01,-8.100355e-01,-8.100355e-01,-8.100355e-01,-8.100355e-01,-8.100355e-01,-8.100355e-01,-8.100355e-01,-8.100355e-01,-8.100355e-01,-8.100355e-01,-8.100355e-01,-8.100355e-01,-8.100355e-01,-8.100355e-01,-8.100355e-01,-8.100355e-01,-8.100355e-01,-8.100355e-01,-8.100355e-01,-8.100355e-01,-8.100355e-01,-8.100355e-01,-8.100355e-01,-8.100355e-01,-8.100355e-01,-8.100355e-01,-8.100355e-01,-8.100355e-01,-8.100355e-01,-8.100355e-01,-8.100355e-01,-8.100355e-01,-8.100355e-01,-8.100355e-01,-8.100355e-01,-8.100355e-01,-8.100355e-01,-8.100355e-01,-8.100355e-01,-8.100355e-01,-8.100355e-01,-8.100355e-01,-8.100355e-01,-8.100355e-01,-8.100355e-01,-8.100355e-01,-8.100355e-01,-8.100355e-01,-8.100355e-01]
# solinit[6] = [-7.928242e-01,-7.863437e-01,-7.798631e-01,-7.733826e-01,-7.669021e-01,-7.604215e-01,-7.539410e-01,-7.474604e-01,-7.409799e-01,-7.344994e-01,-7.280188e-01,-7.215383e-01,-7.150577e-01,-7.085772e-01,-7.020967e-01,-6.956161e-01,-6.891356e-01,-6.826550e-01,-6.761745e-01,-6.696940e-01,-6.632134e-01,-6.567329e-01,-6.502523e-01,-6.437718e-01,-6.372913e-01,-6.308107e-01,-6.243302e-01,-6.178496e-01,-6.113691e-01,-6.048886e-01,-5.984080e-01,-5.919275e-01,-5.854469e-01,-5.789664e-01,-5.724859e-01,-5.660053e-01,-5.595248e-01,-5.530442e-01,-5.465637e-01,-5.400832e-01,-5.336026e-01,-5.271221e-01,-5.206415e-01,-5.141610e-01,-5.076805e-01,-5.011999e-01,-4.947194e-01,-4.882389e-01,-4.817583e-01,-4.752778e-01,-4.687972e-01,-4.623167e-01,-4.558362e-01,-4.493556e-01,-4.428751e-01,-4.363945e-01,-4.299140e-01,-4.234335e-01,-4.169529e-01,-4.104724e-01,-4.039918e-01,-3.975113e-01,-3.910308e-01,-3.845502e-01,-3.780697e-01,-3.715891e-01,-3.651086e-01,-3.586281e-01,-3.521475e-01,-3.456670e-01,-3.391864e-01,-3.327059e-01,-3.262254e-01,-3.197448e-01,-3.132643e-01,-3.067837e-01,-3.003032e-01,-2.938227e-01,-2.873421e-01,-2.808616e-01,-2.743810e-01,-2.700607e-01,-2.657403e-01,-2.614200e-01,-2.570996e-01,-2.527792e-01,-2.484589e-01,-2.441385e-01,-2.398182e-01,-2.354978e-01,-2.290173e-01,-2.225367e-01,-2.160562e-01,-2.095756e-01,-2.030951e-01,-1.966146e-01,-1.901340e-01,-1.836535e-01,-1.771729e-01,-1.706924e-01,-1.642119e-01,-1.577313e-01,-1.512508e-01,-1.447702e-01,-1.382897e-01,-1.318092e-01,-1.253286e-01,-1.188481e-01,-1.123675e-01,-1.058870e-01,-9.940647e-02,-9.292593e-02,-8.644539e-02,-7.996485e-02,-7.348431e-02,-6.700377e-02,-6.052323e-02,-5.404269e-02,-4.756215e-02,-4.108161e-02,-3.460107e-02,-2.812053e-02,-2.163999e-02,-1.515945e-02,-8.678910e-03,-2.198370e-03,4.282170e-03,1.076271e-02,1.724325e-02,2.372379e-02,3.020433e-02,3.668487e-02,4.316541e-02,4.964595e-02,5.612649e-02,6.260703e-02,6.908757e-02,7.556811e-02,8.204865e-02,8.852919e-02,9.500973e-02,1.014903e-01,1.079708e-01,1.144513e-01,1.209319e-01,1.274124e-01,1.338930e-01,1.403735e-01,1.468540e-01,1.533346e-01,1.598151e-01,1.662957e-01,1.727762e-01,1.792567e-01,1.857373e-01,1.922178e-01,1.986984e-01,2.051789e-01,2.116594e-01,2.181400e-01,2.246205e-01,2.311011e-01]
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
    
    Vxbardot = np.zeros(len(tau))
    Vybardot = np.zeros(len(tau))
    lambda_2_bardot = np.zeros(len(tau))
    lambda_3_bardot = np.zeros(len(tau))
    lambda_4_bardot = np.zeros(len(tau))
    for i in range(len(tau)):
        xbar = X[0, i]
        ybar = X[1, i]
        Vxbar = X[2, i]
        Vybar = X[3, i]
        lambda_2_bar = X[4, i]
        lambda_3_bar = X[5, i]
        lambda_4_bar = X[6, i]
        m = m0 - abs(mdot)*tau[i]*tf
        Vxbardot[i] = (f/Vc*(-lambda_3_bar/np.sqrt(lambda_3_bar**2 + lambda_4_bar**2)) \
            - eta*np.exp(-ybar*beta)*Vxbar*np.sqrt(Vxbar**2 + Vybar**2)*Vc)/m
        Vybardot[i] = (f/Vc*(-lambda_4_bar/np.sqrt(lambda_3_bar**2 + lambda_4_bar**2)) \
            - eta*np.exp(-ybar*beta)*Vybar*np.sqrt(Vxbar**2 + Vybar**2)*Vc)/m - g_accel/Vc
        if np.sqrt(Vxbar**2 + Vybar**2) == 0:
            lambda_2_bardot[i] = 0
            lambda_3_bardot[i] = 0
            lambda_4_bardot[i] = -lambda_2_bar*Vc/h
        else:
            lambda_2_bardot[i] = \
                -(lambda_3_bar*Vxbar+lambda_4_bar*Vybar)*eta*beta*np.exp(-ybar*beta)*np.sqrt(Vxbar**2+Vybar**2)*Vc/m
            lambda_3_bardot[i] = eta*np.exp(-ybar*beta)*Vc*(lambda_3_bar*(2*Vxbar**2+Vybar**2) \
                + lambda_4_bar*Vxbar*Vybar)/np.sqrt(Vxbar**2+Vybar**2)/m
            lambda_4_bardot[i] = -lambda_2_bar*Vc/h+eta*np.exp(-ybar*beta)*Vc*(lambda_4_bar*(Vxbar**2
                + 2*Vybar**2) \
                + lambda_3_bar*Vxbar*Vybar)/np.sqrt(Vxbar**2+Vybar**2)/m

    dX_dtau = tf * np.array([xbardot, ybardot, Vxbardot, Vybardot, lambda_2_bardot, lambda_3_bardot, lambda_4_bardot])
    return dX_dtau

def ascent_bcs_tf(ya, yb, p):
    xbari = ya[0]  # use i instead of 0 to distinguish from desired initial
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
    PSI = np.array([
        xbari - xbar0,
        ybari - ybar0,
        Vxbari - Vxbari,
        Vybari - Vybari,
        ybarff - ybarf,
        Vxbarff - Vxbarf,
        Vybarff - Vybarf,
        Hf
    ])
    return PSI
    


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
