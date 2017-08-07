import numpy as np

def check_vals(mat, _min = -100.0, _max = 100.0):
    if np.min(mat) < _min:
        print mat
        assert False, 'value was too small!'

    if np.max(mat) > _max:
        print mat
        assert False, 'value was too large!'
