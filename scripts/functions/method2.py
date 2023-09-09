import numpy as np

def method2(A, x, y, case, LH1, LH2, LHk, TOL):
    
    # LH is the left hand side of convergence rate: 
    ## LH1 is for first iterate
    ## LH2 is for the second iterate (used for GKO and MWRKO only)
    ## LHk is for the k-th iterate
    
    k=1
    m, n = A.shape
    x_old = np.zeros(n)
    ap_error = []
    # e0 = ||x0 - x*||^2
    ar = (np.linalg.norm(x_old-x))**2
    #ap_error.append(ar)
    upper_bd = []
    
    # first iterate (k = 1)
    if case == 'GKO' or case == 'MWRKO':
        inner_p = A@np.transpose(A)
        row_lst = [] 
        resid = abs(A@x_old - y)
        denom = np.sum(np.abs(A)**2,axis=-1)**(1./2)
        i1 = np.argmax(resid/denom)
        a1 = A[i1,:]
        x1 = x_old - ((a1@x_old - y[i1]) / np.linalg.norm(a1)**2) * np.transpose(a1)
        # add upper_bd: ||x1 - x*||^2 <= LH1 * ||x0 - x*||^2
        bd = LH1 * ar
        upper_bd.append(bd)
        # update x
        x_old = x1
        row_lst.append(i1)
        ar = (np.linalg.norm(x_old-x))**2
        #ap_error.append(ar)
        k += 1
        ik = row_lst[-1]
    count = 0
    while True:
        rhat = A@x_old - y
        match case:
            case "GK":
                r = (rhat)**2
                i = np.argmax(r)
                ai = A[i,:]
                xk = x_old - ((np.transpose(ai)@x_old - y[i]) /  np.linalg.norm(ai)**2 * ai)
                # compute upper_bd: ||xk - x*||^2 <= LH * ||x(k-1) - x*||^2
                bd = ( LHk ** k ) * (np.linalg.norm(x_old-x))**2
                upper_bd.append(bd)
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
                print(max(w)**2)
                t = r / np.linalg.norm(w)**2
                xk = x_old - t*w
                # compute upper_bd: ||xk - x*||^2 <= LH * ||x(k-1) - x*||^2
                # compute dynamic range
                gamma = np.linalg.norm(A@x_old - y, np.inf)**2 / np.linalg.norm(A@x_old - y)**2
                if k == 2:
                    bd = ( 1 - LH2 / gamma ) * ar
                    upper_bd.append(bd)
                else:
                    bd = ( 1 - LH2 / gamma ) * ar
                    upper_bd.append(bd)
                print(LH2 / gamma)
                print(gamma)
                print("")
                count += 1
            case "MWRKO":
                resid = abs(rhat)
                i_k1 = np.argmax(resid/denom)
                row_lst.append(i_k1)
                r = A[i_k1,:]@x_old - y[i_k1]
                w = A[i_k1,:] - ((inner_p[ik, i_k1] / np.linalg.norm( A[ik,:])**2) * A[ik,:])
                t = r / np.linalg.norm(w)**2
                xk = x_old - t*w 
                # compute upper_bd: ||xk - x*||^2 <= LH * ||x(k-1) - x*||^2
                if k == 2:
                    bd = LH2 * ar
                    upper_bd.append(bd)
                else:
                    bd = LHk * ar
                    upper_bd.append(bd)
  
        # update x and approximation error
        x_old = xk
        ar = (np.linalg.norm(x_old-x))**2
        gamma = np.linalg.norm(A@x_old - y, np.inf)**2 / np.linalg.norm(A@x_old - y)**2
        ap_error.append(ar)
        k+=1
        
        if count == 5:
            break
        
        if ar < TOL or k == 100000:
            if case == 'GK':
            # compute upper_bd: ||xk - x*||^2 <= LH * ||x(k-1) - x*||^2
                bd = ( LHk ** k ) * ap_error[0]
                upper_bd.append(bd)
            if case == 'GKO':
                bd = ( 1 - LHk / gamma ) * ar
                upper_bd.append(bd)
            break
            
    return k, ap_error, upper_bd
