#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  9 14:44:47 2021

@author: nickolai
"""

from pyomo.dae import ContinuousSet, DerivativeVar
from pyomo.environ import ConcreteModel, TransformationFactory, Var, \
                          NonNegativeReals, Expression, Constraint, \
                          SolverFactory, Objective, cos, sin, Param, minimize, \
                          NonNegativeReals
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import simpson

# Define parameters of the problem
h = 185.2e3 # meters, final altitude (100 nmi circular orbit)
Vc = 1.627e3 # m/s, Circular speed at 100 nmi
g_accel = 1.62 # m/sˆ2, gravitational acceleration of Moon
Thrust2Weight = 3 # Thrust to Weight ratio for Ascent Vehicle, in lunar G's
F = Thrust2Weight * g_accel

model = ConcreteModel("rocket")
model.T = Var(domain=NonNegativeReals)
tf = 1
model.t = ContinuousSet(bounds=(0, tf))
model.x = Var(model.t, domain=NonNegativeReals)
model.y = Var(model.t, domain=NonNegativeReals)
model.xdot = DerivativeVar(model.x, wrt=model.t, domain=NonNegativeReals)
model.xdoubledot = DerivativeVar(model.xdot, wrt=model.t)
model.ydot = DerivativeVar(model.y, wrt=model.t, domain=NonNegativeReals)
model.ydoubledot = DerivativeVar(model.ydot, wrt=model.t)
model.theta = Var(model.t, bounds=(-np.pi, np.pi))


def xoderule (m, t) :
  return m.xdoubledot[t] == (F*cos(m.theta[t]))*m.T**2
model.xode = Constraint(model.t, rule=xoderule)


def yoderule (m, t) :
  return m.ydoubledot[t] == (F*sin(m.theta[t]) - g_accel)*m.T**2
model.yode = Constraint(model.t, rule=yoderule)
model.x[0].fix(0)
model.y[0].fix(0)
model.xdot[0].fix(0)
model.ydot[0].fix(0)
model.y[tf].fix(h)
# model.xdot[tf].fix(Vc)
model.ydot[tf].fix(0)

model.xdotfinal = Constraint(model.t, rule=lambda m, t: Constraint.Skip if t != m.t.last() else m.xdot[t] == Vc*m.T)



discretizer = TransformationFactory('dae.collocation')
discretizer.apply_to(model, wrt=model.t, nfe=30, ncp=6)
discretizer.reduce_collocation_points(model, var=model.theta, ncp=1, contset=model.t)

model.obj = Objective(expr=model.T, sense=minimize)

solver = SolverFactory('ipopt')
results = solver.solve(model)
print(results)

tf = model.T()
tsim = [t*tf for t in model.t]
thetasim = [model.theta[t]() for t in model.t]
xsim = [model.x[t]() for t in model.t]
ysim = [model.y[t]() for t in model.t]
xdotsim = [model.xdot[t]()/tf for t in model.t]
ydotsim = [model.ydot[t]()/tf for t in model.t]
xdoubledotsim = [model.xdoubledot[t]() for t in model.t]
ydoubledotsim = [model.ydoubledot[t]() for t in model.t]

plt.subplot(421)
plt.plot(tsim, xsim, 'b')
plt.ylabel('x')
plt.grid()
plt.subplot(422)
plt.plot(tsim, ysim, 'r')
plt.ylabel('y')
plt.grid()
plt.subplot(423)
plt.plot(tsim, xdotsim, 'g')
plt.ylabel('xdot')
plt.grid()
plt.subplot(424)
plt.plot(tsim, ydotsim, 'k')
plt.ylabel('ydot')
plt.grid()
plt.subplot(425)
plt.plot(tsim, np.rad2deg(thetasim), 'm+')
plt.ylabel('theta')
plt.grid()