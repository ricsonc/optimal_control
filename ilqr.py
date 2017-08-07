#!/usr/bin/env python

import numpy as np
from abc import ABCMeta
import time
from utils import check_vals

class FiniteDiff: 
    def __init__(self, fn, dom_dim, range_dim, eps = 1E-3, ns = 1):
        '''
        fn : the function to differentiated, has one argument
        dom_dim : dimensionality of the domain of fn
        range_dim : dimentionality of the codomain of fn
        eps : the epsilon used when computing gradients
        ns : number of smaples, should be > 1 for stochastic fns
        '''
        self.fn = fn 
        self.dom_dim = dom_dim
        self.range_dim = range_dim
        self.eps = eps
        self.ns = ns 
        self.verify_fn()

    def verify_fn(self):
        zeros = np.matrix(np.zeros(self.dom_dim)).T
        out = self.fn(zeros)
        assert np.shape(out) == (self.range_dim, 1)

    def diff(self, x):
        '''
        x : a point in the domain of fn
        outputs shape [range_dim, dom_dim]
        '''
        
        assert np.shape(x) == (self.dom_dim, 1)

        deltas = map(lambda x: np.matrix(x).T,
                     list(np.identity(self.dom_dim)*self.eps))

        yslopes = []
        for delta in deltas:
            xp = x+delta
            xm = x-delta
            
            if self.ns == 1:
                yp = self.fn(xp)
                ym = self.fn(xm)
            else:
                yp = sum((self.fn(xp) for i in range(self.ns)))/float(self.ns)
                ym = sum((self.fn(xm) for i in range(self.ns)))/float(self.ns)
            
            yslope = (yp-ym)/(2*self.eps) #R^r
            yslopes.append(yslope)

        dydx = np.concatenate(yslopes, axis = 1) #R^(r*d)

        assert np.shape(dydx) == (self.range_dim, self.dom_dim)
        return dydx

    def jacobian(self):
        '''returns a function which computes the jacobian'''
        return self.diff

    def gradient(self):
        '''
        returns a function which computes the gradient
        note that the range of the function must be single-dimension
        '''
        assert self.range_dim == 1
        return lambda x: self.diff(x).T

    def hessian(self):
        '''returns a function which computes the hessi'''
        F = FiniteDiff(self.gradient(), self.dom_dim, self.dom_dim,
                       self.eps, self.ns)
        return F.jacobian()

class TwoArgFiniteDiff:
    def __init__(self, fn, dom1_dim, dom2_dim, range_dim, eps = 1E-3, ns = 1):
        '''
        fn : the function to be differentiated, has two arguments
        dom1_dim, dom2_dim : dimensionality of domain of fn
        range_dim : dimensionality of the codomain of fn
        eps : the epsilon used when computing gradients
        ns : number of samples, should be > 1 for stochastic fns
        '''
        
        self.fn = fn
        #we fix arg2 and consider the derivative wrt arg1, and vice versa
        self.fd1 = lambda arg2: FiniteDiff(lambda arg1: fn(arg1, arg2), dom1_dim, range_dim, eps, ns)
        self.fd2 = lambda arg1: FiniteDiff(lambda arg2: fn(arg1, arg2), dom2_dim, range_dim, eps, ns)
        self.dom1_dim = dom1_dim
        self.dom2_dim = dom2_dim
        self.eps = eps
        self.ns = ns

    def diff1(self, arg1, arg2):
        '''
        arg1 : the first argument of the fn
        arg2 : the second argument of the fn
        output : difference wrt x1 at x1 = arg1, x2 = arg2
        '''
        return self.fd1(arg2).diff(arg1)

    def diff2(self, arg1, arg2):
        '''output : difference wrt x2 at x1 = arg1, x2 = arg2'''
        return self.fd2(arg1).diff(arg2)

    def gradient1(self, arg1, arg2):
        '''output : gradient wrt x1 at x1 = arg1, x2 = arg2'''
        return self.fd1(arg2).gradient()(arg1)

    def gradient2(self, arg1, arg2):
        '''output : gradient wrt x2 at x1 = arg1, x2 = arg2'''
        return self.fd2(arg1).gradient()(arg2)

    def jacobian1(self, arg1, arg2):
        '''output : jacobian wrt x2 at x1 = arg1, x2 = arg2'''
        return self.fd1(arg2).jacobian()(arg1)

    def jacobian2(self, arg1, arg2):
        '''output : jacobian wrt x2 at x1 = arg1, x2 = arg2'''        
        return self.fd2(arg1).jacobian()(arg2)

    def hessian11(self, arg1, arg2):
        '''output : hessian wrt x1, x1'''
        return self.fd1(arg2).hessian()(arg1)

    def hessian22(self, arg1, arg2):
        '''output : hessian wrt x2, x2'''
        return self.fd2(arg1).hessian()(arg2)

    def hessian12(self, arg1, arg2):
        '''hessian wrt x1, x2'''
        tafd = TwoArgFiniteDiff(self.gradient1,
                                self.dom1_dim, self.dom2_dim, self.dom1_dim,
                                self.eps, self.ns)
        hessian = tafd.jacobian2(arg1, arg2)
        assert np.shape(hessian) == (self.dom1_dim, self.dom2_dim)
        return hessian

    def hessian21(self, arg1, arg2):
        '''hessian wrt x2, x1'''
        tafd = TwoArgFiniteDiff(self.gradient2,
                                self.dom1_dim, self.dom2_dim, self.dom2_dim,
                                self.eps, self.ns)
        hessian = tafd.jacobian1(arg1, arg2)
        assert np.shape(hessian) == (self.dom2_dim, self.dom1_dim)
        return hessian
        
class ILQR:
    def __init__(self, state_dim, act_dim,
                 dynamics, dyn_state_jac, dyn_act_jac,
                 cost, cost_state_grad, cost_act_grad,
                 cost_state_state_hess, cost_state_act_hess,
                 cost_act_state_hess, cost_act_act_hess,
                 eps = 1e-3, ns = 1):
        '''
        state_dim : state dimension of the env
        act_dim : action dimension of the env
        dynamics : fn which maps (state, action) to new state
        dyn_state_jac : jacobian of the dynamics wrt state, optional
        dyn_act_jac : jacobian of the dynamics wrt action, optional
        cost : fn which maps (state, action) to cost
        cost_state_grad : gradient of the cost fn wrt state, optional
        cost_act_grad : gradient of the cost fn wrt action, optional
        cost_*_*_hess : hessian of the cost fn, optional
        
        for optional arguments, pass in None if unkown
        they will be computed numerically in that case

        eps : epsilon used for finite differencing if necessary
        ns : number of samples used for finite differencing
        '''

        #an integer > 0
        self.state_dim = state_dim 
        self.act_dim = act_dim

        dyn_fd = TwoArgFiniteDiff(dynamics, state_dim, act_dim, state_dim, eps, ns)
        cost_fd = TwoArgFiniteDiff(cost, state_dim, act_dim, 1, eps, ns)
        
        #s x a -> a
        self.dynamics = dynamics
        #s x a -> (s,s)
        self.dyn_state_jac = dyn_state_jac if dyn_state_jac else dyn_fd.jacobian1
        #s x a -> (s,a)
        self.dyn_act_jac = dyn_act_jac if dyn_act_jac else dyn_fd.jacobian2

        #s x a -> ()
        self.cost = cost
        #s x a -> s
        self.cost_state_grad = cost_state_grad if cost_state_grad else cost_fd.gradient1
        #s x a -> a
        self.cost_act_grad = cost_act_grad if cost_act_grad else cost_fd.gradient2
        #s x a -> (s,s)
        self.cost_state_state_hess = (cost_state_state_hess
                                      if cost_state_state_hess
                                      else cost_fd.hessian11)
        #s x a -> (s,a)
        self.cost_state_act_hess = (cost_state_act_hess
                                    if cost_state_act_hess
                                    else cost_fd.hessian12)
        #s x a -> (a,s)
        self.cost_act_state_hess = (cost_act_state_hess
                                    if cost_act_state_hess
                                    else cost_fd.hessian21)
        #s x a -> (a,a)
        self.cost_act_act_hess = (cost_act_act_hess
                                  if cost_act_act_hess
                                  else cost_fd.hessian22)

    def config(self, start_state, horizon = 50,
               initial_actions = None, num_iters = 10,
               ilqr_tol = 1E-3, damping = 0.0):
        '''
        start_state : the start state for the mdp
        horizon : integer, how far out to optimize
        initial_actions : initial trajectory to start optimizing from
        num_iters : maximum iterations to go for
        ilqr_tol : if total action difference is under this threshold, stop
        damping : update the actions with a damping factor for more stability
        '''

        self.start_state = start_state #shape (s,)
        self.horizon = horizon
        if initial_actions:
            self.initial_actions = initial_actions #shape (H,a)
        else:
            self.initial_actions = [np.matrix(np.ones(self.act_dim)).T for i in range(horizon)]

        self.num_iters = num_iters
        self.ilqr_tol = ilqr_tol
        self.damping = damping

    def solve_single_iteration(self, acts):
        '''
        acts : initial guess of actions, shape (H,a)
        output : actions
        '''

        DEBUGMODE = True

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

            if DEBUGMODE:
                print 'on loop %d' % i

            #we should be around here
            x0 = xs[i] 
            u0 = acts[i]
            
            #first linearize dynamics
            d_ = self.dynamics(x0, u0)
            D_x = self.dyn_state_jac(x0, u0)
            D_u = self.dyn_act_jac(x0, u0)

            #switching from x0 and u0 to x and u basis
            d_ -= D_x*x0 + D_u*u0

            if DEBUGMODE:
                check_vals(d_)
                check_vals(D_x)
                check_vals(D_u)
            
            #quadratic approximation of cost
            c_ = self.cost(x0, u0)
            c_x = self.cost_state_grad(x0, u0)
            c_u = self.cost_act_grad(x0, u0)
            C_xx = self.cost_state_state_hess(x0, u0)
            C_xu = self.cost_state_act_hess(x0, u0)
            C_ux = self.cost_act_state_hess(x0, u0)
            C_uu = self.cost_act_act_hess(x0, u0)

            if DEBUGMODE:
                check_vals(c_)
                check_vals(c_x)
                check_vals(c_u)
                check_vals(C_xx)
                check_vals(C_xu)
                check_vals(C_ux)
                check_vals(C_uu)

            #switching from x0 and u0 to x and u center
            c_ += (0.5*x0.T*C_xx*x0 + 0.5*u0.T*C_uu*u0 + x0.T*C_xu*u0 -
                   c_x.T*x0 - c_u.T*u0)
            c_x -= C_xx*x0 + C_xu*u0
            c_u -= C_uu*u0 + C_ux*x0

            if DEBUGMODE:
                check_vals(c_)
                check_vals(c_x)
                check_vals(c_u)
                
            D_u_T_P = D_u.T*P_mat
            Q_u = D_u_T_P*D_u + C_uu
            Q_x = D_u_T_P*D_x + C_ux
            q_ = D_u_T_P*d_ + D_u.T*p_vec

            #now we should compute #K and k
            try:
                Q_u_inv = Q_u.I
            except:
                Q_u_inv = (Q_u + 1E-3*np.identity(self.act_dim)).I
                
            K_mat = -Q_u_inv * Q_x
            k_vec = -Q_u_inv * q_

            if DEBUGMODE:
                check_vals(K_mat)
                check_vals(k_vec)
                
            #Now it's time for #M
            M_mat = D_x + D_u * K_mat
            m_vec = D_u * k_vec + d_

            if DEBUGMODE:
                check_vals(M_mat)
                check_vals(m_vec)

            #finally we can obtain the new #P's
            P_mat = (C_xx +
                     K_mat.T*C_uu*K_mat +
                     K_mat.T*C_ux + C_xu*K_mat +
                     M_mat.T*P_mat*M_mat)
            
            p_vec = (c_x +
                     K_mat.T*c_u +
                     K_mat.T*C_uu*k_vec +
                     C_xu*k_vec +
                     M_mat.T*P_mat*m_vec +
                     M_mat*p_vec)

            p_sca = (c_u.T*k_vec +
                     0.5*k_vec.T*C_uu*k_vec +
                     p_vec.T*m_vec +
                     p_sca)


            if DEBUGMODE:
                check_vals(P_mat)
                check_vals(p_vec)
                check_vals(p_sca)
            
            #we know what the optimal action will be now, given x
            K_mats.append(K_mat)
            k_vecs.append(k_vec)

        K_mats = list(reversed(K_mats))
        k_vecs = list(reversed(k_vecs))

        #rollout once
        newacts = []
        x = self.start_state
        acc_cost = 0
        for i, (K_mat, k_vec) in enumerate(zip(K_mats, k_vecs)):
            act = (1.0-self.damping)*(K_mat * x + k_vec) + self.damping*acts[i]
            
            if DEBUGMODE:
                check_vals(act)
                
            newacts.append(act)
            acc_cost += self.cost(x, act)
            x = self.dynamics(x, act)

        print 'cost was', acc_cost

        return newacts

    def solve_iterative(self):
        '''
        performs ILQR, returns good set of actions
        '''

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

class MPC:
    '''
    model predicative control
    due to the interactive way that MPC works, the interface is a bit different from ILQR
    one must first initialize with start_session
    then alternate calls to get_next_move and update_state
    '''
    
    def __init__(self, ilqr_system):
        '''
        set up MPC system given an ILQR instance
        '''
        self.system = ilqr_system
        self.start = self.system.start_state

    def start_session(self):
        '''start an MPC solving session'''
        self.system.start_state = self.start

    def get_next_move(self):
        '''ask for the next move'''
        actions = self.system.solve_iterative()
        return actions[0]

    def update_state(self, state):
        '''return the state which happens'''
        self.system.state = state
        self.system.horizon -= 1
        
if __name__ == '__main__':
    pass

#make mpc test
#need a family with time varying dynamics and costs
#3-argument dynamics/cost fn, with optional time
