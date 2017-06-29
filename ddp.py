#!/usr/bin/env python2

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
        self.dyn_state_jac = dyn_state_jac
        #R^s x R^a -> R^(s*a)
        self.dyn_act_jac = dyn_act_jac

        #R^s x R^a -> R        
        self.cost = cost
        #R^s x R^a -> R^s
        self.cost_state_grad = cost_state_grad
        #R^s x R^a -> R^a        
        self.cost_act_grad = cost_act_grad
        #R^s x R^a -> R^(s*s)
        self.cost_state_state_hess = cost_state_state_hess
        #R^s x R^a -> R^(s*a)
        self.cost_state_act_hess = cost_state_act_hess
        #R^s x R^a -> R^(a*s)
        self.cost_act_state_hess = cost_act_state_hess
        #R^s x R^a -> R^(a*a)
        self.cost_act_act_hess = cost_act_act_hess

    def config(self, start_state, horizon, initial_actions, num_iters):
        self.start_state = start_state
        self.horizon = horizon
        self.initial_actions = initial_actions
        self.num_iters = num_iters

    #returns R^(H*a)
    def solve(self):
        pass
