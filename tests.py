#!/usr/bin/env python

import gym
from ilqr import FiniteDiff, TwoArgFiniteDiff, ILQR
import numpy as np
import math
import time

def makepoint(lst):
    '''
    makes a column vector out of a list of points
    '''
    return np.matrix(lst).T

def test0():
    '''
    tests finite difference class
    '''
    def f0(x):
        return x*x*3.0
    
    fd1 = FiniteDiff(f0, 1, 1, 1e-6)
    f0_grad_est = fd1.gradient()
    f0_grad = lambda x: 6.0*x

    points = [0.0, 1.0, 2.0, -1.0]
    testpoints = map(makepoint, points)
    for x in testpoints:
        true_grad = f0_grad(x)
        est_grad = f0_grad_est(x)
        assert np.allclose(true_grad, est_grad)

def test1():
    '''
    more extensive testing of finite differences
    '''
        
    def f1(x):
        assert np.shape(x) == (2,1)
        G = np.matrix([[1.0,2.0],[3.0,4.0]])
        out = x.T*G*x
        assert np.shape(out) == (1,1)
        return out

    def f1_grad(x):
        G = np.matrix([[1.0,2.0],[3.0,4.0]])    
        return (G+G.T)*x

    def f1_hess(x):
        G = np.matrix([[1.0,2.0],[3.0,4.0]])
        return G+G.T

    fd1 = FiniteDiff(f1, 2, 1, 1e-3)
    f1_grad_est = fd1.gradient()

    points = [
        [0.0, 0.0],
        [1.0, 1.0],
        [0.0, 1.0],
        [1.0, 0.0]
    ]
    testpoints = map(makepoint, points)
    
    for x in testpoints:
        true_grad = f1_grad(x)
        est_grad = f1_grad_est(x)
        assert np.allclose(true_grad, est_grad)

    f1_hess_est = fd1.hessian()
    for x in testpoints:
        true_hess = f1_hess(x)
        est_hess = f1_hess_est(x)
        assert np.allclose(true_hess, est_hess)

def test2():
    '''
    tests two arg finite diff class
    '''
    w = makepoint([2.0, 3.0])
    v = makepoint([4.0, 5.0])
    A = np.matrix([[0.5, 1.5], [2.5, 3.5]])
    B = np.matrix([[4.5, 5.5], [6.5, 7.5]])
    C = np.matrix([[-1.0, -2.0], [3.0, 4.0]])
    D = np.matrix([[-5.0, -6.0], [7.0, 8.0]])
    c = np.matrix([[3.0]])

    def f2(x, u):
        return (w.T*x + v.T*u + c +
                x.T*A*x + u.T*B*u +
                x.T*C*u + u.T*D*x)

    def f2_gradx(x, u):
        return w+(A+A.T)*x + D.T*u + C*u

    def f2_gradu(x, u):
        return v+(B+B.T)*u + C.T*x + D*x

    def f2_jacx(x, u):
        return f2_gradx(x,u).T

    def f2_jacu(x, u):
        return f2_gradu(x,u).T

    def f2_hessxx(x, u):
        return (A+A.T)

    def f2_hessuu(x, u):
        return (B+B.T)

    def f2_hessxu(x, u):
        return D.T+C

    def f2_hessux(x, u):
        return C.T+D

    M = TwoArgFiniteDiff(f2, 2, 2, 1)
    
    points = [
        [0.0, 0.0],
        [1.0, 1.0],
        [0.0, 1.0],
        [1.0, 0.0],

        [-0.5, -0.5],
        [-2.0, 3.0],
        [3.0, -1.0],
        [5.0, 20.0]
    ]
    testpoints = map(makepoint, points)

    fnpairs = [
        (f2_gradx, M.gradient1),
        (f2_gradu, M.gradient2),
        (f2_jacx, M.jacobian1),
        (f2_jacu, M.jacobian2),
        (f2_hessxx, M.hessian11),
        (f2_hessuu, M.hessian22),
        (f2_hessxu, M.hessian12),
        (f2_hessux, M.hessian21)
    ]

    for fn1, fn2 in fnpairs:
        for x in testpoints:
            for u in testpoints:
                true_val = fn1(x, u)
                est_val = fn2(x, u)

                assert np.allclose(true_val, est_val)

def ILQR_test(dynamics, cost,
              state_dim, act_dim,
              start = None,
              horizon = 50,
              iters = 10,
              eps = 1e-3,
              expected_position = 0.0,
              expected_cost = None):
    '''creates a new ilqr test given the args'''

    solver = ILQR(state_dim, act_dim,
                 dynamics, None, None,
                 cost, None, None,
                 None, None, None, None)

    if start is None:
        start = np.matrix(np.ones(state_dim)*3.0).T

    initial_actions = [np.matrix(np.zeros(act_dim)).T for i in range(horizon)]
    solver.config(start, horizon, initial_actions, iters, eps)
    solution = solver.solve_iterative()

    x = start
    acc_cost = 0
    for act in solution:
        acc_cost += cost(x, act)
        x = dynamics(x, act)

    if expected_position is not None:
        assert np.allclose(x, expected_position,
                           rtol = 0.01, atol = 0.01)

    if expected_cost is not None:
        assert np.allclose(acc_cost, expected_cost,
                           rtol = 0.1, atol = 1.0)
    
def test3():
    ''' 
    ILQR
    tests minimization of a simple quadratic function
    '''

    def dynamics(state, action):
        return state+action

    def cost(state, action):
        return 2.0*state.T*state + 5.0*action.T*action

    ILQR_test(dynamics, cost, 3, 3, expected_cost = 116.5)

def test4():
    '''
    ILQR
    tests system with more complex dynamics
    '''
    
    def dynamics(state, action):
        pos = state[0:2]
        vel = state[2:4]
        return np.concatenate([pos+vel, vel+action])

    def cost(state, action):
        pos = state[0:2]
        vel = state[2:4]
        return 0.1*pos.T*pos + vel[0]*vel[1] + action.T*action

    start = np.matrix([1.0, 2.0, 0.0, 0.0]).T

    ILQR_test(dynamics, cost,
              4, 2, start)

def test5():
    '''Cartpole with ILQR'''

    def angle_normalize(x):
        return (((x+np.pi) % (2*np.pi)) - np.pi)

    def cos_sin_to_theta(costheta, sintheta):
        costheta = np.clip(costheta, -1, 1)
        theta = math.acos(costheta)
        if sintheta >= 0:
            ans = theta
        else:
            ans = 2*math.pi-theta
        return angle_normalize(ans)

    def read_state(state):
        theta = cos_sin_to_theta(state[0], state[1])
        thdot = state[2] #angular velocity
        return theta, thdot

    def softclip(x, _min, _max):
        if _min <= x <= _max:
            return float(x)

        if _min > x:
            extra = _min-x
        elif x > _max:
            extra = x-_max

        if _min > x:
            return _min - math.sqrt(extra)
        elif x > _max:
            return _max + math.sqrt(extra)

    def dynamics(state, action):
        theta, thdot = read_state(state)
        
        g = 10.0
        m = 1.0
        l = 1.0
        dt = 0.05
        max_torque = 2.0
        max_speed = 8.0

        action = softclip(action, -max_torque, max_torque)
        newthdot = thdot + (-3*g/(2*l) * np.sin(theta + np.pi) + 3./(m*l**2)*action) * dt
        newtheta = theta + newthdot*dt
        newthdot = softclip(newthdot, -max_speed, max_speed) 
        newstate = np.matrix([math.cos(newtheta), math.sin(newtheta), newthdot])
        return newstate.T
    
    def cost(state, action):
        theta, thdot = read_state(state)
        #cost = angle_normalize(theta)**2 + 0.1*thdot**2 + 0.001*(action.T*action)
        cost = np.matrix(thdot**2) + 0.01*action.T*action
        return cost

    env = gym.make('Pendulum-v0')
    start_state = env.reset()
    start_state = np.matrix(start_state).T
    
    solver = ILQR(3, 1,
                  dynamics, None, None,
                  cost, None, None,
                  None, None, None, None,
                  eps = 1E-2)

    
    solver.config(start_state, horizon = 30, num_iters = 20, damping = 0.0)
    solution = solver.solve_iterative()

    print solution

    env.render()
    for step in solution:
        time.sleep(0.1)
        env.step(step[0])
        env.render()
        
if __name__ == '__main__':
    # print 'running test 0'
    # test0()
    # print 'running test 1'    
    # test1()
    # print 'running test 2'    
    # test2()
    # print 'running test 3'
    # test3()
    # print 'running test 4'
    # test4()
    print 'running test 5'
    test5()

