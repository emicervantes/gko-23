
def gk(A, x, y, TOL):
  k=1
  m, n = A.shape
  x_old = np.zeros(n)
  x_lst = [x_old]
  ap_error = []
  ar = approximation_error(x_old, x)
  ap_error.append(ar)
  while True:
    r = []
    for i in range(m):
      resid = (np.transpose(A[i,:])@x_old - y[i])**2
      r.append(resid)
    i = np.argmax(r)
    ai = A[i,:]
    x_new = x_old - ((np.transpose(ai)@x_old - y[i]) * ai)
    x_lst.append(x_new)
    x_old = x_new
    ar = approximation_error(x_old, x)
    ap_error.append(ar)
    k+=1
    if ar < TOL:
      break

  return k, ap_error