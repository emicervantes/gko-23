import numpy as np

def gk(A, x, y, TOL):
  k=1
  m, n = A.shape
  x_old = np.zeros(n)
  x_lst = [x_old]
  ap_error = []
  ar = (np.linalg.norm(x_old-x))**2
  ap_error.append(ar)
  while True:
    r = (A@x_old - y)**2
    i = np.argmax(r)
    ai = A[i,:]
    x_new = x_old - ((np.transpose(ai)@x_old - y[i]) /  np.linalg.norm(ai)**2 * ai)
    x_lst.append(x_new)
    x_old = x_new
    ar = (np.linalg.norm(x_old-x))**2
    ap_error.append(ar)
    k+=1
    if ar < TOL:
      break
  return k, ap_error
