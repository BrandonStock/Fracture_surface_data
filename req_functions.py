"""
Functions required for aperture generation model, does not include number swapping algorithm or calculating H
"""
import sys
import numpy as np
import matplotlib.pyplot as plt

def RMS_COR(trace, plot=None):   
        """
        
        Calculates the RMS_COR slope and intercept from trace, trace must have dimensions (n,1)
        
        """
        from scipy import signal
        trace=signal.detrend(trace,axis=0) ## detrending data, should have no effect unless there is a linear trend, in which case it is removed
        # Linear least squares fit to data is subtracted from the data (Marsch 2021)
        maxBin=np.int(len(trace)/2)
        nofData=len(trace)
        # 'generate Step sizes (1 2 4 8 16 32 64 128 ... etc)'
        # steps=[]
        # s=1
        # while s <=maxBin:
        #     s*=2
        #     steps.append(s)
        # steps=np.insert(steps,0,1)[:-1]
        
        ## Steps going up in intervals of one, not sure why Martin didnt do this
        steps=np.arange(1,nofData,1)
        
        ##added so only use less than 20% of length scale, suggested by (Malinverno. 1990)
        # step=[]
        # step[:]=[x for x in steps if x < (0.2*nofData)]  
        # steps=np.asarray(step) 
        
        ## However Marsch 2021 states 10% of max is better, removes first dents in sigma(dh) vs dv plot
        step=[]
        step[:]=[x for x in steps if x < (0.1*nofData)]  
        steps=np.asarray(step) 
        
        'Calculate the std og height differences for the different step sizes, eq 5-4'        
        offset_value=[]
        diff=[]
        std=[]
        for i in range(0,len(steps)):
            # print (i)
            zero_array=np.zeros([steps[i]])
            zero_array=zero_array.reshape(len(zero_array),1)
            new=np.vstack([zero_array,trace])[:-len(zero_array)]
            offset_value.append(new)
            d=trace-offset_value[i]
            di=np.delete(d,[np.arange(0,steps[i])])
            diff.append(di)
            st=np.std(diff[i])
            std.append(st)
        
        'Calculate slope and intercept'        
        m,c=np.polyfit(np.log(steps),np.log(std),1)
        y_fit=np.exp(m*np.log(steps)+c)
        c=np.exp(c)
        # print (m)
        
        ## correction based of Marsch 2021 paper
        if m > 0.5:
            m=np.log(m)+1.18
        else:
            m==m
            
        def pl(steps,std):
            plt.scatter(steps,std, label='Data',color='red', marker='+')
            plt.plot(steps,y_fit,label='infered slope')
            plt.xscale('log')
            plt.yscale('log')
            plt.xlabel('length difference')
            plt.ylabel('std of height difference')
            plt.plot([],[],' ',label='slope = %s'% float('%.3g' % m))
            plt.plot([],[],' ',label='intercept = %s'% float('%.3g' % c))
            plt.legend()
            plt.show()

        if plot==True:
            return m, c, std,pl(steps,std)
        return m,c,std
    
    
def RMS_COR_5(trace, plot=None):   
        """
        
        Calculates the RMS_COR slope and intercept from trace, trace must have dimensions (n,1)
        
        """
        from scipy import signal
        trace=signal.detrend(trace,axis=0) ## detrending data, should have no effect unless there is a linear trend, in which case it is removed
        # Linear least squares fit to data is subtracted from the data (Marsch 2021)
        maxBin=np.int(len(trace)/2)
        nofData=len(trace)
        # 'generate Step sizes (1 2 4 8 16 32 64 128 ... etc)'
        # steps=[]
        # s=1
        # while s <=maxBin:
        #     s*=2
        #     steps.append(s)
        # steps=np.insert(steps,0,1)[:-1]
        
        ## Steps going up in intervals of one, not sure why Martin didnt do this
        steps=np.arange(1,nofData,1)
        
        ##added so only use less than 20% of length scale, suggested by (Malinverno. 1990)
        # step=[]
        # step[:]=[x for x in steps if x < (0.2*nofData)]  
        # steps=np.asarray(step) 
        
        ## However Marsch 2021 states 10% of max is better, removes first dents in sigma(dh) vs dv plot
        step=[]
        step[:]=[x for x in steps if x < (0.05*nofData)]  
        steps=np.asarray(step) 
        
        'Calculate the std og height differences for the different step sizes, eq 5-4'        
        offset_value=[]
        diff=[]
        std=[]
        for i in range(0,len(steps)):
            # print (i)
            zero_array=np.zeros([steps[i]])
            zero_array=zero_array.reshape(len(zero_array),1)
            new=np.vstack([zero_array,trace])[:-len(zero_array)]
            offset_value.append(new)
            d=trace-offset_value[i]
            di=np.delete(d,[np.arange(0,steps[i])])
            diff.append(di)
            st=np.std(diff[i])
            std.append(st)
        
        'Calculate slope and intercept'        
        m,c=np.polyfit(np.log(steps),np.log(std),1)
        y_fit=np.exp(m*np.log(steps)+c)
        c=np.exp(c)
        # print (m)
        
        ## correction based of Marsch 2021 paper
        if m > 0.5:
            m=np.log(m)+1.18
        else:
            m==m
            
        def pl(steps,std):
            plt.scatter(steps,std, label='Data',color='red', marker='+')
            plt.plot(steps,y_fit,label='infered slope')
            plt.xscale('log')
            plt.yscale('log')
            plt.xlabel('length difference')
            plt.ylabel('std of height difference')
            plt.plot([],[],' ',label='slope = %s'% float('%.3g' % m))
            plt.plot([],[],' ',label='intercept = %s'% float('%.3g' % c))
            plt.legend()
            plt.show()

        if plot==True:
            return m, c, std,pl(steps,std)
        return m,c,std
    
def RMS_COR_20(trace, plot=None):   
        """
        
        Calculates the RMS_COR slope and intercept from trace, trace must have dimensions (n,1)
        
        """
        from scipy import signal
        trace=signal.detrend(trace,axis=0) ## detrending data, should have no effect unless there is a linear trend, in which case it is removed
        # Linear least squares fit to data is subtracted from the data (Marsch 2021)
        maxBin=np.int(len(trace)/2)
        nofData=len(trace)
        # 'generate Step sizes (1 2 4 8 16 32 64 128 ... etc)'
        # steps=[]
        # s=1
        # while s <=maxBin:
        #     s*=2
        #     steps.append(s)
        # steps=np.insert(steps,0,1)[:-1]
        
        ## Steps going up in intervals of one, not sure why Martin didnt do this
        steps=np.arange(1,nofData,1)
        
        ##added so only use less than 20% of length scale, suggested by (Malinverno. 1990)
        # step=[]
        # step[:]=[x for x in steps if x < (0.2*nofData)]  
        # steps=np.asarray(step) 
        
        ## However Marsch 2021 states 10% of max is better, removes first dents in sigma(dh) vs dv plot
        step=[]
        step[:]=[x for x in steps if x < (0.2*nofData)]  
        steps=np.asarray(step) 
        
        'Calculate the std og height differences for the different step sizes, eq 5-4'        
        offset_value=[]
        diff=[]
        std=[]
        for i in range(0,len(steps)):
            # print (i)
            zero_array=np.zeros([steps[i]])
            zero_array=zero_array.reshape(len(zero_array),1)
            new=np.vstack([zero_array,trace])[:-len(zero_array)]
            offset_value.append(new)
            d=trace-offset_value[i]
            di=np.delete(d,[np.arange(0,steps[i])])
            diff.append(di)
            st=np.std(diff[i])
            std.append(st)
        
        'Calculate slope and intercept'        
        m,c=np.polyfit(np.log(steps),np.log(std),1)
        y_fit=np.exp(m*np.log(steps)+c)
        c=np.exp(c)
        # print (m)
        
        ## correction based of Marsch 2021 paper
        if m > 0.5:
            m=np.log(m)+1.18
        else:
            m==m
            
        def pl(steps,std):
            plt.scatter(steps,std, label='Data',color='red', marker='+')
            plt.plot(steps,y_fit,label='infered slope')
            plt.xscale('log')
            plt.yscale('log')
            plt.xlabel('length difference')
            plt.ylabel('std of height difference')
            plt.plot([],[],' ',label='slope = %s'% float('%.3g' % m))
            plt.plot([],[],' ',label='intercept = %s'% float('%.3g' % c))
            plt.legend()
            plt.show()

        if plot==True:
            return m, c, std,pl(steps,std)
        return m,c,std
        
def RMS_COR_1(trace, plot=None):   
        """
        
        Calculates the RMS_COR slope and intercept from trace, trace must have dimensions (n,1)
        
        """
        from scipy import signal
        trace=signal.detrend(trace,axis=0) ## detrending data, should have no effect unless there is a linear trend, in which case it is removed
        # Linear least squares fit to data is subtracted from the data (Marsch 2021)
        maxBin=np.int(len(trace)/2)
        nofData=len(trace)
        # 'generate Step sizes (1 2 4 8 16 32 64 128 ... etc)'
        # steps=[]
        # s=1
        # while s <=maxBin:
        #     s*=2
        #     steps.append(s)
        # steps=np.insert(steps,0,1)[:-1]
        
        ## Steps going up in intervals of one, not sure why Martin didnt do this
        steps=np.arange(1,nofData,1)
        
        ##added so only use less than 20% of length scale, suggested by (Malinverno. 1990)
        # step=[]
        # step[:]=[x for x in steps if x < (0.2*nofData)]  
        # steps=np.asarray(step) 
        
        ## However Marsch 2021 states 10% of max is better, removes first dents in sigma(dh) vs dv plot
        step=[]
        step[:]=[x for x in steps if x < (0.01*nofData)]  
        steps=np.asarray(step) 
        
        'Calculate the std og height differences for the different step sizes, eq 5-4'        
        offset_value=[]
        diff=[]
        std=[]
        for i in range(0,len(steps)):
            # print (i)
            zero_array=np.zeros([steps[i]])
            zero_array=zero_array.reshape(len(zero_array),1)
            new=np.vstack([zero_array,trace])[:-len(zero_array)]
            offset_value.append(new)
            d=trace-offset_value[i]
            di=np.delete(d,[np.arange(0,steps[i])])
            diff.append(di)
            st=np.std(diff[i])
            std.append(st)
        
        'Calculate slope and intercept'        
        m,c=np.polyfit(np.log(steps),np.log(std),1)
        y_fit=np.exp(m*np.log(steps)+c)
        c=np.exp(c)
        # print (m)
        
        ## correction based of Marsch 2021 paper
        if m > 0.5:
            m=np.log(m)+1.18
        else:
            m==m
            
        def pl(steps,std):
            plt.scatter(steps,std, label='Data',color='red', marker='+')
            plt.plot(steps,y_fit,label='infered slope')
            plt.xscale('log')
            plt.yscale('log')
            plt.xlabel('length difference')
            plt.ylabel('std of height difference')
            plt.plot([],[],' ',label='slope = %s'% float('%.3g' % m))
            plt.plot([],[],' ',label='intercept = %s'% float('%.3g' % c))
            plt.legend()
            plt.show()

        if plot==True:
            return m, c, std,pl(steps,std)
        return m,c,std
    
def psd_data(surface=None):
    """
    https://bertvandenbroucke.netlify.app/2019/05/24/computing-a-power-spectrum-in-python/ also Candela et al 2009 paper
    
    Get wavenumber and bins for plotting PSD, input surface must be square

    """
    import scipy.stats as stats
    from scipy import signal
    surface=signal.detrend(surface)
    narray=surface.shape[0]
    narray1=surface.shape[1]

    fourier_upper = np.fft.fftn(surface) #Fourier transform, complex values, only interested in amplituude
    fourier_amplitudes = np.abs(fourier_upper)**2 #compute variance

    kfreq = np.fft.fftfreq(narray) * narray #1d array containing wave vectors for fft.fftn call in correct order, multiply by narray so 2d array
    kfreq1=np.fft.fftfreq(narray1) * narray1
    kfreq2D = np.meshgrid(kfreq1, kfreq)
    knrm = np.sqrt(kfreq2D[0]**2 + kfreq2D[1]**2) #Calculate norm

    knrm = knrm.flatten()
    fourier_amplitudes = fourier_amplitudes.flatten()

    kbins = np.arange(0.5, narray1//2+1, 1.) #Integer k value bins
    kvals = 0.5 * (kbins[1:] + kbins[:-1])
    Abins, _, _ = stats.binned_statistic(knrm, fourier_amplitudes, statistic = "mean", bins = kbins)
    Abins *= np.pi * (kbins[1:]**2 - kbins[:-1]**2)
    return kvals, Abins

 
# Function to sort arr1
# according to arr2
def solve_R(arr1, arr2):
    """
    Reorder R_unique to the correct order for how the appear in R_values
    """
    from collections import Counter

    # Our output array
    res = []
    # Counting Frequency of each
    # number in arr1
    f = Counter(arr1)
    # Iterate over arr2 and append all
    # occurrences of element of
    # arr2 from arr1
    for e in arr2:
        # Appending element 'e',
        # f[e] number of times
        res.extend([e]*f[e])
        # Count of 'e' after appending is zero
        f[e] = 0
    # Remaining numbers in arr1 in sorted
    # order (Numbers with non-zero frequency)
    rem = list(sorted(filter(
      lambda x: f[x] != 0, f.keys())))
    # Append them also
    for e in rem:
        res.extend([e]*f[e])        
    return res


def WavenumberGrid_flattened(nval=None,cutoff_length=None,R_values=None,K_cutoff=None): 
    """
    Parameters
    ----------
    nval : Number of values along edge of grid
    R_values : Correlation values for wavenumbers based off PSDR (correlationValues function)
    K_cutoff : Cutoff wavenumber, after which the correlation will remain constant

    Returns
    -------
    K : A list of arrays containing the locations of wavenumbers from the grid

    """
    
    up=np.arange(1,int(cutoff_length)+1,1) # changed this to start from one so didnt get divide by zero error in rad
    down=np.arange(1,(int(cutoff_length))+1-1,1)[::-1][:-1] # also removed zero
    updown=np.append(up,down)
    multi=int(nval/cutoff_length)+4
    updown=np.tile(updown,multi)[0:nval]
    K=[]
    for i in range(0,len(updown)):
        for j in range(0,len(updown)):
            t=np.sqrt(updown[i]*updown[i]+updown[j]*updown[j])
            K.append(t)
    K_combine1=np.asarray(K)#.reshape(len(updown),len(updown))
    ## wavenumber where the correlation values change,
    wk=np.where(R_values[:-1] != R_values[1:])[0]
    wk=wk[1::]
    ## locations (or just amounnt) of values falling within this wavenumber
    K_first=np.where(K_combine1<wk[0]) #first value as it is written different than middle values and harder to put into a loop
    K_cut_rev=len(R_values)-1-K_cutoff
    K_last=np.where(K_combine1>=wk[-K_cut_rev]) #same reasonas above, all values above last wk value, for this casse above 312
    K=[]
    for i in range (0, len(wk)-K_cut_rev): 
        # print (i)
        K_loop=np.where((K_combine1>=wk[i]) & (K_combine1<wk[i+1]))
        K.append(K_loop)
    K.insert(0,K_first)
    K.insert(len(K),K_last)
    return K, K_combine1


def getAandBforRoughsurf(reordered_R=None,R_unique=None,K=None,all_A=None,all_B=None):
    """
    Parameters
    ----------
    reordered_R : reordered_R from function
    R_unique : R_unique from function
    K : K from function
    all_A : correlated array A for every wavenumber
    all_B : correlated array B for every wavenumber

    Returns
    -------
    A_new : Semi correlated array A used for Candela input
    B_new : Semi correlated array B used for Candela input

    """
    index=[]
    for i in range (0,len(reordered_R)):
        result=np.where(R_unique == reordered_R[i])
        index.append(result[0][0])
    
    ## Sort all_A and all_B so that the order is correct for the relveant correlation and wavelength
    all_A=all_A[index,:]
    all_B=all_B[index,:]
    
    # R_unique=reordered_R # highligh first section and keep this for artificial R_values where they are already in order
    
    ## Check that the array has actually being reorded to the correct positions
    # pos=0
    # test_corr=stats.pearsonr(all_B[pos,:],all_A[pos,:])[0]
    # print ('position',pos)
    # print ('target', reordered_R[pos])
    # print ('calculated correlation', test_corr)
    
    B=[]
    for i in range(0,len(K)):
        # print (i)
        B_loop=all_B[i,:][K[i]]
        B.append(B_loop)
        
    A=[]
    for i in range(0,len(K)):
        A_loop=all_A[i,:][K[i]]
        A.append(A_loop)
    
    B_test=np.concatenate(B, axis=0)#[::-1]
    A_test=np.concatenate(A, axis=0)#[::-1]
    
    args3=[]
    for i in range(0, len(K)):
        a=K[i][0]
        args3.append(a)
        
    all_Ks=np.concatenate(args3)
    
    B_new=B_test[all_Ks]
    A_new=A_test[all_Ks]
    return A_new, B_new


def makeFracSurf_updated(N=None, H=None, anisotropy=None, phase1=None,wavenumber_grid=None):#, phase2=None,wavenumber_grid=None):
    '''
    This is the final version, this is how the surface is generated
    '''
    rad = np.power( wavenumber_grid**2/(anisotropy*anisotropy), -(H+1.0)/2.0 ).reshape(N,N)
    
    N2=int(N/2) 
    A = np.zeros( (N,N), dtype=np.complex )
    # np.random.seed(seed)
    N2=int(N/2)
    for i in range(1,N2,1):
      for j in range(1,N2,1):
          p1=phase1[i,j]
          r1=rad[i,j]
          A[i,j]     = complex( r1*np.cos(p1),  r1*np.sin(p1) )
          
          p12=phase1[N-i,N-j]
          r12=rad[N-i,N-j]
          A[N-i,N-j] = complex( r12*np.cos(p12), -r12*np.sin(p12) )
          
          p2=phase1[i,N-j]
          r2=rad[i,N-j]
          A[i,N-j]   = complex( r2*np.cos(p2),  r2*np.sin(p2) )
          
          p22=phase1[N-i,j] 
          r22=rad[N-i,j] 
          A[N-i,j]   = complex( r22*np.cos(p22), -r22*np.sin(p22) )
    A = np.fft.ifft2( A )
    return A.real

def correlationValues(lower,upper):
    """
    Finds the correlation for each wavenumber from the PSDR of the real data
    """    
    aper=upper-lower
    aper[aper<0] = 0
    kvals_lower,abins_lower=psd_data(lower)
    kvals_upper,abins_upper=psd_data(upper)
    kvals_aper,abins_aper=psd_data(aper)
    PR_r=abins_aper/(abins_upper+abins_lower)
    
    lamda=((2*np.pi)/kvals_lower)
    wl=(1/kvals_lower)*len(lower[0,:])/10 
    
    logx=np.log(wl)
    logy=np.log(PR_r)
    
    R_values=[]
    for i in range(1,21):
        coeffs=np.polyfit(logx,logy,deg=i)
        poly=np.poly1d(coeffs)
        yfit = lambda wl: np.exp(poly(np.log(wl)))
        test_corr=np.corrcoef(yfit(wl),PR_r)[0,1]**2
        R_values.append(test_corr)
    R_values=np.asarray(R_values)
    deg=np.where(R_values>=R_values[-1])[0][0]  ## Cmake sure 0.93 is the same number as that used in the number swapping algorithm to get R
    
    coeffs=np.polyfit(logx,logy,deg)
    poly=np.poly1d(coeffs)
    yfit = lambda wl: np.exp(poly(np.log(wl)))
    best_fit=yfit(wl)
    # plt.plot(kvals_aper, PR_r,label='PSDR')
    # plt.plot(kvals_aper,yfit(wl), label='best fit')
    # plt.xscale('log')
    # plt.ylabel("$PSDR(k)$")
    # plt.xlabel("$k$")
    # plt.legend()
    # # plt.savefig('1N1_psdr_best_fit_k.png', dpi=600, bbox_inches='tight')
    # plt.show()
    return 1-best_fit
    
def RMS_COR_for_scaling(trace, plot=None):   
        """
        
        Calculates the RMS_COR slope and intercept from trace, trace must have dimensions (n,1)
        
        """
        from scipy import signal
        trace=signal.detrend(trace,axis=0) ## detrending data, should have no effect unless there is a linear trend, in which case it is removed
        # Linear least squares fit to data is subtracted from the data (Marsch 2021)
        maxBin=np.int(len(trace)/2)
        nofData=len(trace)
        'generate Step sizes (1 2 4 8 16 32 64 128 ... etc)'
        # steps=[]
        # s=1
        # while s <=maxBin:
        #     s*=2
        #     steps.append(s)
        # steps=np.array(steps)
        # steps=np.insert(steps,0,1)[:-1]
        
        ## Steps going up in intervals of one, not sure why Martin didnt do this
        steps=np.arange(1,nofData,1)
        
        ##added so only use less than 20% of length scale, suggested by (Malinverno. 1990)
        # step=[]
        # step[:]=[x for x in steps if x < (0.2*nofData)]  
        # steps=np.asarray(step) 
        
        ## However Marsch 2021 states 10% of max is better, removes first dents in sigma(dh) vs dv plot
        step=[]
        step[:]=[x for x in steps if x < (0.1*nofData)]  
        steps=np.asarray(step) 
        
        'Calculate the std og height differences for the different step sizes, eq 5-4'        
        offset_value=[]
        diff=[]
        std=[]
        for i in range(0,len(steps)):
            # print (i)
            zero_array=np.zeros([steps[i]])
            zero_array=zero_array.reshape(len(zero_array),1)
            new=np.vstack([zero_array,trace])[:-len(zero_array)]
            offset_value.append(new)
            d=trace-offset_value[i]
            di=np.delete(d,[np.arange(0,steps[i])])
            diff.append(di)
            st=np.std(diff[i])
            std.append(st)
        
        'Calculate slope and intercept'        
        m,c=np.polyfit(np.log(steps),np.log(std),1)
        y_fit=np.exp(m*np.log(steps)+c)
        c=np.exp(c)
        # print (m)
        
        ## correction based of Marsch 2021 paper
        if m > 0.5:
            m=np.log(m)+1.18
        else:
            m==m
            
        ## find Y value respective to X value, use for rescaling different trace lengths (test)
        # import numpy as np
        val=np.interp((1000/513),steps,y_fit) ## edited
        val=np.exp(val)

        def pl(steps,std):
            plt.scatter(steps,std, label='Data',color='red', marker='+')
            plt.plot(steps,y_fit,label='infered slope')
            plt.xscale('log')
            plt.yscale('log')
            plt.xlabel('length difference')
            plt.ylabel('std of height difference')
            plt.plot([],[],' ',label='slope = %s'% float('%.3g' % m))
            plt.plot([],[],' ',label='intercept = %s'% float('%.3g' % c))
            plt.legend()
            plt.show()

        if plot==True:
            return m, c, std,pl(steps,std)
        return m,c,std,val
 
def calculate_scaling_gen_surf(surface_real,surface_gen):
    '''
    Function using RMS-COR to calculate intercept, scaling parameters, from surface so can be used for rescaling in 
    generation code, this is used for correctly rescaling surfaces

    '''
    x_length=len(surface_gen[1,:])
    y_length=len(surface_gen[:,1])   
        
    # H and intercept
    H_x=[]
    int_x=[]
    H_y=[]
    int_y=[]

    for i in range(0,y_length):
        t=surface_gen[i,:].reshape(x_length,1)
        h,intercept,std=RMS_COR(t)
        H_x.append(h)
        intercept=intercept*(len(surface_real)**2/len(surface_gen)**2)
        int_x.append(intercept) 
    # y
    for i in range(0,x_length):
        t=surface_gen[:,i].reshape(y_length,1)
        h,intercept,std=RMS_COR(t)
        H_y.append(h)
        intercept=intercept*(len(surface_real)**2/len(surface_gen)**2)
        int_y.append(intercept)
    
    int_surface_all_array=int_x+int_y
    med_int_surface=np.median(int_surface_all_array)
    
    return med_int_surface

def calculate_scaling_testing(surface):
    '''
    Function using RMS-COR to calculate intercept, scaling parameters, from surface so can be used for rescaling in 
    generation code, this is needed for testing that the rescaling is doinf the correct things

    '''
    x_length=len(surface[1,:])
    y_length=len(surface[:,1])   
        
    # H and intercept
    H_x=[]
    int_x=[]
    H_y=[]
    int_y=[]
    val_x=[]
    val_y=[]
    for i in range(0,y_length):
        # print (i)
        t=surface[i,:].reshape(x_length,1)
        h,intercept,std,val=RMS_COR_for_scaling(t)
        H_x.append(h)
        int_x.append(intercept) 
        val_x.append(val)
    # y
    for i in range(0,x_length):
        t=surface[:,i].reshape(y_length,1)
        h,intercept,std,val=RMS_COR_for_scaling(t)
        H_y.append(h)
        int_y.append(intercept)
        val_x.append(val)
    
    val_array=val_x+val_y
    med_val=np.median(val_array)
    
    return med_val

def calculate_scaling(surface):
    '''
    Function using RMS-COR to calculate intercept, scaling parameters, from surface so can be used for rescaling in 
    generation code, this is the original

    '''
    x_length=len(surface[1,:])
    y_length=len(surface[:,1])   
        
    # H and intercept
    H_x=[]
    int_x=[]
    H_y=[]
    int_y=[]
    
    for i in range(0,y_length):
        t=surface[i,:].reshape(x_length,1)
        h,intercept,std=RMS_COR(t)
        H_x.append(h)
        int_x.append(intercept) 
    # y
    for i in range(0,x_length):
        t=surface[:,i].reshape(y_length,1)
        h,intercept,std=RMS_COR(t)
        H_y.append(h)
        int_y.append(intercept)
    
    int_surface_all_array=int_x+int_y
    med_int_surface=np.median(int_surface_all_array)
    
    return med_int_surface
