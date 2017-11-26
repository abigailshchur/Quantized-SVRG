import numpy as np

"""
Casts any low precision vector or matrix to a float matrix
"""
def low_precision_to_float(m, s):
	m = m.astype(float)
	return m*s

"""
Casts any 8 bit vector or matrix to 16 bits
"""
def cast_8b_to_16b(m):
	return m.astype(np.int16)
