#!/usr/bin/env python

import numpy as np

class FiniteDiff:
    def __init__(self, fn, dom_dim, range_dim, eps = 1E-3):
        self.fn = fn
        self.dom_dim = dom_dim
        self.range_dim = range_dim
        self.eps = eps
        #R^d to R^r

    #R^d to R^(r*d)
    def diff(self, x):

        deltas = list(np.identity(dom_dim)*self.eps)

        yslopes = []
        for delta in deltas:
            yp = self.fn(x+delta)
            ym = self.fn(x-delta)
            yslope = (yp-ym)/(2*self.eps) #R^r
            yslopes.append(yslope)

        dydx = np.stack(yslopes, axis = 1) #R^(r*d)
        return dydx

def maybe_jac(f, dom_dim, range_dim):
    if f is not None:
        return f
    F = FiniteDiff(f, dom_dim, range_dim)
    return F.diff

def maybe_grad(f, dom_dim):
    if f is not None:
        return f
    F = FiniteDiff(f, dom_dim)
    return F.diff
    #do we need to transpose?

def maybe_hess(f, dom_dim):
    if f is not None:
        return f
    grad = maybe_grad(f, dom_dim)
    return maybe_jac(f, dom_dim, dom_dim)

class DDP:
    def __init__(self, state_dim, act_dim,
                 dynamics, dyn_state_jac, dyn_act_jac,
                 cost, cost_state_grad, cost_act_grad,
                 cost_state_state_hess, cost_state_act_hess,
                 cost_act_state_hess, cost_act_act_hess):

        #an integer > 0
        self.state_dim = state_dim 
        self.act_dim = act_dim

        #R^s x R^a -> R^a
        self.dynamics = dynamics
        #R^s x R^a -> R^(s*s)
        self.dyn_state_jac = maybe_jac(dyn_state_jac)
        #R^s x R^a -> R^(s*a)
        self.dyn_act_jac = maybe_jac(dyn_act_jac)

        #R^s x R^a -> R        
        self.cost = cost
        #R^s x R^a -> R^s
        self.cost_state_grad = maybe_grad(cost_state_grad)
        #R^s x R^a -> R^a        
        self.cost_act_grad = maybe_grad(cost_act_grad)
        #R^s x R^a -> R^(s*s)
        self.cost_state_state_hess = maybe_hess(cost_state_state_hess)
        #R^s x R^a -> R^(s*a)
        self.cost_state_act_hess = maybe_hess(cost_state_act_hess)
        #R^s x R^a -> R^(a*s)
        self.cost_act_state_hess = maybe_hess(cost_act_state_hess)
        #R^s x R^a -> R^(a*a)
        self.cost_act_act_hess = maybe_hess(cost_act_act_hess)

    def config(self, start_state, horizon,
               initial_actions, num_iters,
               ilqr_tol = 1E-3):
        self.start_state = start_state
        self.horizon = horizon
        self.initial_actions = initial_actions
        self.num_iters = num_iters
        self.ilqr_tol = ilqr_tol

    #returns R^(H*a)
    def solve_single_iteration(self, acts):

        #first run the dynamics forward
        xs = [self.start_state]
        for act in acts:
            x_ = self.dynamics(x, act)
            xs.append(x_)
        xs.pop() #we don't need H+1

        #initialize P values
        P_mat = np.matrix(np.zeros((self.state_dim, self.state_dim)))
        p_vec = np.matrix(np.zeros((1,self.state_dim))).T
        p_sca = 0.0

        K_mats = []
        k_vecs = []
        for i in list(reversed(range(self.horizon))):

            #we should be around here
            x_pred = xs[i] 
            u_pred = acts[i]
            
            #first linearize dynamics
            d_ = self.dynamics(x_pred, u_pred)            
            D_s = self.dyn_state_jac(x_pred, u_pred)
            D_u = self.dyn_state_jac(x_pred, u_pred)
            
            #quadratic approximation of cost
            c_ = self.cost(x_pred, u_pred)
            c_x = self.cost_state_grad(x_pred, u_pred)
            c_u = self.cost_act_grad(x_pred, u_pred)
            C_xx = self.cost_state_state_hess(x_pred, u_pred)
            C_xu = self.cost_state_act_hess(x_pred, u_pred)
            C_ux = self.cost_act_state_hess(x_pred, u_pred)
            C_uu = self.cost_act_act_hess(x_pred, u_pred)

            #now we should compute #K and k
            C_uu_inv = C_uu.I #this is not cached
            K_mat = -C_uu_inv * (D_u * P_mat + C_ux)
            k_vec = -0.5 * (D_u * p_vec + c_u)

            #Now it's time for #M
            M_mat = D_x + D_u * K_mat
            M_vec = D_u * k_vec + d_

            #finally we can obtain the new #P's
            P_mat = (C_xx +
                     K_mat.T*C_uu*K_mat +
                     K_mat.T*C_ux + C_xu*K_mat +
                     M_mat.T*P_mat*M_mat)
            
            p_vec = (C_x +
                     K_mat.T*c_u +
                     2*K_mat.T*C_uu*k_vec +
                     2*C_xu*k_vec +
                     2*M_mat.T*P_mat*m_vec +
                     M_mat*p_vec)

            p_sca = (C_u.T*k_vec +
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
                return np.reduce_sum(np.abs(act1-act2))
            return sum((act_diff(act1, act2)
                        for (act1, act2) in zip(acts1, acts2)))
        
        acts = self.initial_actions
        while 1:
            new_acts = self.solve_single_iteration(acts)
            if acts_diff(acts, new_acts) < self.ilqr_tol:
                return new_acts
            acts = new_acts

    def solve_session(self):
        #mpc: helpful when dynamics are uncertain
        pass
        
if __name__ == '__main__':
    solver = DDP(4, 4,
                 None, None, None,
                 None, None, None,
                 None, None, None, None)

    solver.config(None, None, None, None)
    solver.solve()

#now we just need a test problem.
#uh we need to check the correctness of numerical differentiators too
