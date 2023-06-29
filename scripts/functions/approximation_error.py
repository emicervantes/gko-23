def approximation_error(xk, xnew):
  a_error = (np.linalg.norm(xk-xnew))**2
  return a_error
