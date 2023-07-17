import numpy as np

def mwrko(A, x, y, TOL):
  k=1
  m, n = A.shape
  inner_p = A@np.transpose(A)
  x_old = np.zeros(n)
  ap_error = []
  ar = (np.linalg.norm(x_old-x))**2
  ap_error.append(ar)
  row_lst = []
  # pick the row based on maximal weighted residual selection rule
  resid = abs(A@x_old - y)
  denom = np.sum(np.abs(A)**2,axis=-1)**(1./2)
  i1 = np.argmax(resid/denom)
  # compute the next row
  a1 = A[i1,:]
  x1 = x_old - ((a1@x_old - y[i1]) / np.linalg.norm(a1)**2) * np.transpose(a1)
  x_old = x1
  row_lst.append(i1)
  ar = (np.linalg.norm(x_old-x))**2
  ap_error.append(ar)
  k += 1

  while True:
    ik = row_lst[-1]
    resid = abs(A@x_old - y)
    i_k1 = np.argmax(resid/denom)
    row_lst.append(i_k1)
    # Oblique projection with the new selected row
    r = A[i_k1,:]@x_old - y[i_k1]
    w = A[i_k1,:] - ((inner_p[ik, i_k1] / np.linalg.norm( A[ik,:])**2) * A[ik,:])
    t = r / np.linalg.norm(w)**2
    xk = x_old - t*w
    x_old = xk
    ar = (np.linalg.norm(x_old-x))**2
    ap_error.append(ar)
    k+=1

    if ar < TOL:
      break
  return k, ap_error