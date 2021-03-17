import numpy as np


x = np.array([[[1,2],[3,4]]])
print(x, x.shape)
hi = np.repeat(x, 3, axis=1)
print(hi, hi.shape)
