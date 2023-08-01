import numpy as np

def method2(A, x, y, case, LH1, LH, TOL):
    
    # LH is the left hand side of convergence rate: 
    ## LH1 is for first iterate and LH is for the rest
    
    k=1
    m, n = A.shape
    x_old = np.zeros(n)
    ap_error = []
    ar = (np.linalg.norm(x_old-x))**2
    ap_error.append(ar)
    upper_bd = []
    # add upper_bd
    bd = LH1 * ( np.linalg.norm(x_old - x)**2 )
    upper_bd.append(bd)
    
    if case == 'GKO' or case == 'MWRKO':
        inner_p = A@np.transpose(A)
        row_lst = [] 
        resid = abs(A@x_old - y)
        denom = np.sum(np.abs(A)**2,axis=-1)**(1./2)
        i1 = np.argmax(resid/denom)
        a1 = A[i1,:]
        x1 = x_old - ((a1@x_old - y[i1]) / np.linalg.norm(a1)**2) * np.transpose(a1)
        # add upper_bd
        bd = LH1 * ( np.linalg.norm(x_old - x)**2 )
        upper_bd.append(bd)
        # update x
        x_old = x1
        row_lst.append(i1)
        ar = (np.linalg.norm(x_old-x))**2
        ap_error.append(ar)
        k += 1
        ik = row_lst[-1]
    
    while True:
        rhat = A@x_old - y
        match case:
            case "GK":
                r = (rhat)**2
                i = np.argmax(r)
                ai = A[i,:]
                xk = x_old - ((np.transpose(ai)@x_old - y[i]) /  np.linalg.norm(ai)**2 * ai)
            case "GKO":
                inner_dig = np.delete(inner_p.diagonal(), ik)
                all_comb = np.delete(inner_p[:,ik], ik)
                denom = inner_dig - np.square(all_comb) / np.linalg.norm(A[ik,:])**2
                resid  = np.delete(abs(rhat), ik)
                i_k1 = np.argmax(resid/np.sqrt(denom))
                if (i_k1 >= ik):
                  i_k1 += 1
                row_lst.append(i_k1)
                r = A[i_k1,:]@x_old - y[i_k1]
                w = A[i_k1,:] - ((inner_p[ik, i_k1] / np.linalg.norm( A[ik,:])**2) * A[ik,:])
                t = r / np.linalg.norm(w)**2
                xk = x_old - t*w
            case "MWRKO":
                resid = abs(rhat)
                i_k1 = np.argmax(resid/denom)
                row_lst.append(i_k1)
                r = A[i_k1,:]@x_old - y[i_k1]
                w = A[i_k1,:] - ((inner_p[ik, i_k1] / np.linalg.norm( A[ik,:])**2) * A[ik,:])
                t = r / np.linalg.norm(w)**2
                xk = x_old - t*w 
  
        # compute upper_bd
        bd = LH * ( np.linalg.norm(x_old - x)**2 )
        upper_bd.append(bd)
        # update x
        x_old = xk
        ar = (np.linalg.norm(x_old-x))**2
        ap_error.append(ar)
        k+=1
        
        if ar < TOL or k == 100000:
            break
            
    return k, ap_error, upper_bd