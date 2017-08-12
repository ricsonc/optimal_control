import numpy as np

def check_vals(mat, _min = -1000.0, _max = 1000.0):
    if np.min(mat) < _min:
        print mat
        assert False, 'value was too small'

    if np.max(mat) > _max:
        print mat
        assert False, 'value was too large'

    if np.isnan(mat).any():
        print mat
        assert False, 'value has nans'
    
