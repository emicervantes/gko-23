
def mwrko(A, x, y, TOL = 0.01):
  k=1
  m, n = A.shape
  x_old = np.zeros(n)
  x_lst = [x_old]
  ap_error = []
  t_lst = []
  ar = approximation_error(x_old, x)
  ap_error.append(ar)
  row_lst = []
  for i in range(m):
      r = abs(y[i] - A[i,:]@x_lst[-1])
      w = np.linalg.norm(A[i,:])
      t = r / w
      t_lst.append([i,t])
  [ik, t_lst] = max(t_lst,key = lambda x : x[1])
  row_lst.append(ik)
  a1 = A[ik,:]
  x1 = x_old + ((y[ik] - a1@x_old) / np.linalg.norm(a1)**2) * np.transpose(a1)
  x_lst.append(x1)
  x_old = x1

  while True:
    t_lst = []
    ik = row_lst[-1]
    for i in range(m):
      if i != ik:
        r = abs(y[i] - A[i,:]@x_lst[-1])
        w = np.linalg.norm(A[i,:])
        t = r / w
        t_lst.append([i,t])

    [i_k1,t_lst] = max(t_lst,key = lambda x : x[1])
    row_lst.append(i_k1)
    ai = A[ik,:] 
    ai_k = A[i_k1,:] 
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
