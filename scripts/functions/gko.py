

def gko(A, x, y, TOL):
  k=1
  m, n = A.shape
  x_old = np.zeros(n)
  x_lst = [x_old]
  ap_error = []
  ar = approximation_error(x_old, x)
  ap_error.append(ar)
  row_lst = []
  [ik, i_k1] = np.random.choice(range(m), 2)
  row_lst.extend([ik, i_k1])
  a1 = A[ik,:]
  x1 = x_old + ((y[ik] - a1@x_old) / np.linalg.norm(a1)**2) * np.transpose(a1)
  x_lst.append(x1)
  a2 = A[i_k1,:]
  x2 = x1 + ((y[i_k1] - a2@x1) / (np.linalg.norm(a2)**2 - (((a1@a2)**2)/np.linalg.norm(a1)**2))) * ((a2 - (a1@a2)/np.linalg.norm(a1)**2) * a1)
  x_lst.append(x2)
  x_old = x2

  while True:
    t_lst = []
    ik = row_lst[-1]
    for i in range(m):
      if i != ik:
        r = y[i] - A[i,:]@x_lst[-1]
        w = A[i,:] - ( ( A[ik,:]@A[i,:] / np.linalg.norm(A[ik,:])**2 ) * A[ik,:] )
        t = abs(r) / np.linalg.norm(w)**2
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
