import numpy as np

def mwrko(A, x, y, TOL):
  k=1
  m, n = A.shape
  inner_p = A@np.transpose(A)
  x_old = np.zeros(n)
  x_lst = [x_old]
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
  x_lst.append(x_old)
  ar = (np.linalg.norm(x_old-x))**2
  ap_error.append(ar)
  k += 1

  while True:
    t_lst = []
    ik = row_lst[-1]
    resid = abs(A@x_old - y)
    i_k1 = np.argmax(resid/denom)
    row_lst.append(i_k1)
    # Oblique projection with the new selected row
    ai = A[ik,:] # a_k-1
    ai_k = A[i_k1,:] # a_k
    D_ik = ai@ai_k
    r = y[i_k1] - ai_k@x_old
    w = ai_k - ((D_ik / np.linalg.norm(ai)**2) * ai)
    h_ik = np.linalg.norm(w)**2
    alpha = r / h_ik
    xk = x_old + alpha*w
    x_old = xk
    x_lst.append(x_old)
    ar = approximation_error(x_old, x)
    ap_error.append(ar)
    k+=1

    if ar < TOL:
      break
  return k, ap_error