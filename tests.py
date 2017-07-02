#!/usr/bin/env python

from ilqr import FiniteDiff, TwoArgFiniteDiff, ILQR
import numpy as np

def makepoint(lst):
    return np.matrix(lst).T

def test0():
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

    #not so sure about these two!
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

                '''
                print '#'*20
                print 'functions:', fn1, fn2
                print 'x/u:'
                print x
                print u
                print 'values:'
                print true_val
                print est_val
                '''
                
                assert np.allclose(true_val, est_val)

def test3():
    def dynamics(state, action):
        return state+action

    def cost(state, action):
        return 2.0*state.T*state + 5.0*action.T*action

    n = 3
    
    solver = ILQR(n, n,
                 dynamics, None, None,
                 cost, None, None,
                 None, None, None, None)

    start = np.matrix(np.ones(n)*3.0).T
    horizon = 50
    iters = 10
    initial_actions = [np.matrix(np.zeros(n)).T for i in range(horizon)]
    solver.config(start, horizon, initial_actions, iters, 1E-3)
    
    solution = solver.solve_iterative()

    x = start
    acc_cost = 0
    for act in solution:
        acc_cost += cost(x, act)
        x = dynamics(x, act)

    assert np.allclose(x, 0.0)
    assert 116 < acc_cost < 117
    
    
if __name__ == '__main__':
    print 'running test 0'
    test0()
    print 'running test 1'    
    test1()
    print 'running test 2'    
    test2()
    print 'running test 3'
    test3()
