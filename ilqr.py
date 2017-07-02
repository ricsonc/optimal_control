#!/usr/bin/env python

import numpy as np
from abc import ABCMeta
import time

class FiniteDiff: 
    def __init__(self, fn, dom_dim, range_dim, eps = 1E-3):
        self.fn = fn #input: two arguments
        self.dom_dim = dom_dim
        self.range_dim = range_dim
        self.eps = eps
        #R^d to R^r
        self.verify_fn()

    def verify_fn(self):
        zeros = np.matrix(np.zeros(self.dom_dim)).T
        out = self.fn(zeros)
        assert np.shape(out) == (self.range_dim, 1)

    #R^d to R^(r*d)
    def diff(self, x):
        assert np.shape(x) == (self.dom_dim, 1)

        deltas = map(lambda x: np.matrix(x).T,
                     list(np.identity(self.dom_dim)*self.eps))

        yslopes = []
        for delta in deltas:
            yp = self.fn(x+delta)
            ym = self.fn(x-delta)
            
            yslope = (yp-ym)/(2*self.eps) #R^r
            yslopes.append(yslope)

        dydx = np.concatenate(yslopes, axis = 1) #R^(r*d)

        assert np.shape(dydx) == (self.range_dim, self.dom_dim)
        return dydx

    def jacobian(self):
        return self.diff

    def gradient(self):
        assert self.range_dim == 1
        return lambda x: self.diff(x).T

    def hessian(self):
        F = FiniteDiff(self.gradient(), self.dom_dim, self.dom_dim)
        return F.jacobian()

class TwoArgFiniteDiff:
    def __init__(self, fn, dom1_dim, dom2_dim, range_dim, eps = 1E-3):
        self.fn = fn
        #we fix arg2 and consider the derivative wrt arg1, and vice versa
        self.fd1 = lambda arg2: FiniteDiff(lambda arg1: fn(arg1, arg2), dom1_dim, range_dim, eps)
        self.fd2 = lambda arg1: FiniteDiff(lambda arg2: fn(arg1, arg2), dom2_dim, range_dim, eps)
        self.dom1_dim = dom1_dim
        self.dom2_dim = dom2_dim
        self.eps = eps

    def diff1(self, arg1, arg2):
        return self.fd1(arg2).diff(arg1)

    def diff2(self, arg1, arg2):
        return self.fd2(arg1).diff(arg2)

    def gradient1(self, arg1, arg2):
        return self.fd1(arg2).gradient()(arg1)

    def gradient2(self, arg1, arg2):
        return self.fd2(arg1).gradient()(arg2)

    def jacobian1(self, arg1, arg2):
        return self.fd1(arg2).jacobian()(arg1)

    def jacobian2(self, arg1, arg2):
        return self.fd2(arg1).jacobian()(arg2)

    def hessian11(self, arg1, arg2):
        return self.fd1(arg2).hessian()(arg1)

    def hessian22(self, arg1, arg2):
        return self.fd2(arg1).hessian()(arg2)

    def hessian12(self, arg1, arg2):
        tafd = TwoArgFiniteDiff(self.gradient1,
                                self.dom1_dim, self.dom2_dim, self.dom1_dim,
                                self.eps)
        hessian = tafd.jacobian2(arg1, arg2)
        assert np.shape(hessian) == (self.dom1_dim, self.dom2_dim)
        return hessian

    def hessian21(self, arg1, arg2):
        tafd = TwoArgFiniteDiff(self.gradient2,
                                self.dom2_dim, self.dom1_dim, self.dom2_dim,
                                self.eps)
        hessian = tafd.jacobian1(arg1, arg2)
        assert np.shape(hessian) == (self.dom2_dim, self.dom1_dim)
        return hessian
        
class DDP:
    def __init__(self, state_dim, act_dim,
                 dynamics, dyn_state_jac, dyn_act_jac,
                 cost, cost_state_grad, cost_act_grad,
                 cost_state_state_hess, cost_state_act_hess,
                 cost_act_state_hess, cost_act_act_hess):

        #an integer > 0
        self.state_dim = state_dim 
        self.act_dim = act_dim

        dyn_fd = TwoArgFiniteDiff(dynamics, state_dim, act_dim, state_dim)
        cost_fd = TwoArgFiniteDiff(cost, state_dim, act_dim, 1)
        
        #R^s x R^a -> R^a
        self.dynamics = dynamics
        #R^s x R^a -> R^(s*s)
        self.dyn_state_jac = dyn_state_jac if dyn_state_jac else dyn_fd.jacobian1
        #R^s x R^a -> R^(s*a)
        self.dyn_act_jac = dyn_act_jac if dyn_act_jac else dyn_fd.jacobian2

        #R^s x R^a -> R        
        self.cost = cost
        #R^s x R^a -> R^s
        self.cost_state_grad = cost_state_grad if cost_state_grad else cost_fd.gradient1
        #R^s x R^a -> R^a        
        self.cost_act_grad = cost_act_grad if cost_act_grad else cost_fd.gradient2
        #R^s x R^a -> R^(s*s)
        self.cost_state_state_hess = (cost_state_state_hess
                                      if cost_state_state_hess
                                      else cost_fd.hessian11)
        #R^s x R^a -> R^(s*a)
        self.cost_state_act_hess = (cost_state_act_hess
                                    if cost_state_act_hess
                                    else cost_fd.hessian12)
        #R^s x R^a -> R^(a*s)
        self.cost_act_state_hess = (cost_act_state_hess
                                    if cost_act_state_hess
                                    else cost_fd.hessian21)
        #R^s x R^a -> R^(a*a)
        self.cost_act_act_hess = (cost_act_act_hess
                                  if cost_act_act_hess
                                  else cost_fd.hessian22)

    def config(self, start_state, horizon,
               initial_actions, num_iters,
               ilqr_tol = 1E-3):

        #R^s
        self.start_state = start_state
        self.horizon = horizon
        #R^(H*a)        
        self.initial_actions = initial_actions
        self.num_iters = num_iters
        self.ilqr_tol = ilqr_tol

    #returns R^(H*a)
    def solve_single_iteration(self, acts):

        #first run the dynamics forward
        xs = [self.start_state]
        for act in acts:
            x_ = self.dynamics(xs[-1], act)
            xs.append(x_)
        xs.pop() #we don't need H+1

        #initialize P values
        P_mat = np.matrix(np.zeros((self.state_dim, self.state_dim)))
        p_vec = np.matrix(np.zeros((1,self.state_dim))).T
        p_sca = 0.0

        K_mats = []
        k_vecs = []
        for i in list(reversed(range(self.horizon))):

            print P_mat
            print p_vec
            print p_sca
            

            #we should be around here
            x_pred = xs[i] 
            u_pred = acts[i]
            
            #first linearize dynamics
            d_ = self.dynamics(x_pred, u_pred)            
            D_x = self.dyn_state_jac(x_pred, u_pred)
            D_u = self.dyn_state_jac(x_pred, u_pred)
            
            #quadratic approximation of cost
            c_ = self.cost(x_pred, u_pred)
            c_x = self.cost_state_grad(x_pred, u_pred)
            c_u = self.cost_act_grad(x_pred, u_pred)
            C_xx = self.cost_state_state_hess(x_pred, u_pred)
            C_xu = self.cost_state_act_hess(x_pred, u_pred)
            C_ux = self.cost_act_state_hess(x_pred, u_pred)
            C_uu = self.cost_act_act_hess(x_pred, u_pred)

            '''
            print '#'*10
            print c_
            print c_x
            print c_u
            print C_xx
            print C_xu
            print C_ux
            print C_uu
            print '@'*10
            '''
            
            #now we should compute #K and k
            C_uu_inv = C_uu.I #this is not cached
            K_mat = -C_uu_inv * (D_u * P_mat + C_ux)
            k_vec = -0.5 * (D_u * p_vec + c_u) #wait...should be a C_uu_inv here!

            #Now it's time for #M
            M_mat = D_x + D_u * K_mat
            m_vec = D_u * k_vec + d_

            #finally we can obtain the new #P's
            P_mat = (C_xx +
                     K_mat.T*C_uu*K_mat +
                     K_mat.T*C_ux + C_xu*K_mat +
                     M_mat.T*P_mat*M_mat)
            
            p_vec = (c_x +
                     K_mat.T*c_u +
                     2*K_mat.T*C_uu*k_vec +
                     2*C_xu*k_vec +
                     2*M_mat.T*P_mat*m_vec +
                     M_mat*p_vec)

            p_sca = (c_u.T*k_vec +
                     k_vec.T*C_uu*k_vec +
                     p_vec.T*m_vec +
                     p_sca)

            #we know what the optimal action will be now, given x
            K_mats.append(K_mat)
            k_vecs.append(k_vec)

        K_mats = list(reversed(K_mats))
        k_vecs = list(reversed(k_vecs))

        #rollout once
        acts = []
        x = self.start_state
        for (K_mat, k_vec) in zip(K_mats, k_vecs):
            act = K_mat * x + k_vec
            acts.append(act)
            x = self.dynamics(x, act)

        return acts

    def solve_iterative(self):
        #basically ilqr

        def acts_diff(acts1, acts2):
            def act_diff(act1, act2):
                return np.sum(np.abs(act1-act2))
            return sum((act_diff(act1, act2)
                        for (act1, act2) in zip(acts1, acts2)))
        
        acts = self.initial_actions
        count = 0
        while count < self.num_iters:
            count += 1
            new_acts = self.solve_single_iteration(acts)
            if acts_diff(acts, new_acts) < self.ilqr_tol:
                return new_acts
            acts = new_acts
        return acts

    def solve_session(self):
        #mpc: helpful when dynamics are uncertain
        pass

if __name__ == '__main__':

    def dynamics(state, action):
        return state+action

    def cost(state, action):
        return 2.0*state.T*state + 5.0*action.T*action

    n = 2
    
    solver = DDP(n, n,
                 dynamics, None, None,
                 cost, None, None,
                 None, None, None, None)

    start = np.matrix(np.ones(n)*3.0).T
    horizon = 2
    initial_actions = [np.matrix(np.zeros(n)).T for i in range(horizon)]
    solver.config(start, horizon, initial_actions, 1, 1E-3)
    
    solution = solver.solve_iterative()
    print solution
 