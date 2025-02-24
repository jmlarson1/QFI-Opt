import os
USE_DIFFRAX = bool(os.getenv("USE_DIFFRAX"))

if USE_DIFFRAX:
    import diffrax
    import jax
    import jax.numpy as np
    from jax.scipy.linalg import expm
    from jax.numpy import linalg as LA
    jax.config.update("jax_enable_x64", True)
    COMPLEX_TYPE = np.complex128

else:
    import numpy as np
    from scipy.integrate import solve_ivp

###################################
## Define functions for building the matrix of Hamiltonian and dissipation
###################################
## Dicke basis, ordered as |0,0>, |1,-1>,|1,0>... |J-1,J-1>,|J,-J>,...|J,J>
## Dissipation: single site and collective Sz, Sp, Sm
## Hamiltonian: OAT, TAT, LMG
## Solver: scipy solve_ivp

 
# dephasing with spontaneous emission/absorption
def dephasing_DisMat(gamma_p, gamma_m, gamma_z, gamma_pall, gamma_mall, gamma_zall, N):
    # table for j, mj
    tablej = []
    tablemj1 = []
    tablemj2 = []
    for i in range(0, N + 1):
        for j in range(0, 2 * i + 1):
            for k in range(j, 2 * i + 1):
                tablej.append(i)
                tablemj1.append(j - i)
                tablemj2.append(k - i)
    dimension = len(tablej)
    resultmat = []
    resultmat2 = []  # complex conjugate
    resultmatloc = []
    resultmatloc2 = []

    def Fgamma1(tablej, tablemj1, tablemj2, N):
        gamma1 = 0.5 * gamma_mall * (
                    (tablej + tablemj1) * (tablej - tablemj1 + 1) + (tablej + tablemj2) * (tablej - tablemj2 + 1))
        gamma1 += 0.5 * gamma_pall * (
                    (tablej - tablemj1) * (tablej + tablemj1 + 1) + (tablej - tablemj2) * (tablej + tablemj2 + 1))
        gamma1 += 0.5 * gamma_zall * (tablemj1 - tablemj2) * (tablemj1 - tablemj2)
        gamma1 += 0.5 * gamma_m * (2 * N + tablemj1 + tablemj2)
        gamma1 += 0.5 * gamma_p * (2 * N - tablemj1 - tablemj2)
        if (tablej != 0):
            gamma1 += 0.5 * gamma_z * (N - tablemj1 * tablemj2 * (N + 1) / tablej / (tablej + 1))
        return gamma1

    def Fgamma2(tablej, tablemj1, tablemj2, N):
        gamma2 = gamma_mall * np.sqrt(
            (tablej + tablemj1) * (tablej - tablemj1 + 1) * (tablej + tablemj2) * (tablej - tablemj2 + 1))
        gamma2 += 0.5 * gamma_m * np.sqrt(
            (tablej + tablemj1) * (tablej - tablemj1 + 1) * (tablej + tablemj2) * (tablej - tablemj2 + 1)) * (
                              N + 1) / tablej / (tablej + 1)
        return gamma2

    def Fgamma8(tablej, tablemj1, tablemj2, N):
        gamma8 = gamma_pall * np.sqrt(
            (tablej - tablemj1) * (tablej + tablemj1 + 1) * (tablej - tablemj2) * (tablej + tablemj2 + 1))
        gamma8 += 0.5 * gamma_p * np.sqrt(
            (tablej - tablemj1) * (tablej + tablemj1 + 1) * (tablej - tablemj2) * (tablej + tablemj2 + 1)) * (
                              N + 1) / tablej / (tablej + 1)
        return gamma8

    def Fgamma5(tablej, tablemj1, tablemj2, N):
        gamma5 = 0.5 * gamma_z * np.sqrt((tablej + tablemj1) * (tablej - tablemj1)) * np.sqrt(
            (tablej + tablemj2) * (tablej - tablemj2)) * (N + tablej + 1) / tablej / (2 * tablej + 1)
        return gamma5

    def Fgamma3(tablej, tablemj1, tablemj2, N):
        gamma3 = 0.5 * gamma_m * np.sqrt(
            (tablej + tablemj1) * (tablej + tablemj1 - 1) * (tablej + tablemj2) * (tablej + tablemj2 - 1)) * (
                             N + tablej + 1) / tablej / (2 * tablej + 1)
        return gamma3

    def Fgamma7(tablej, tablemj1, tablemj2, N):
        gamma7 = 0.5 * gamma_p * np.sqrt(
            (tablej - tablemj1) * (tablej - tablemj1 - 1) * (tablej - tablemj2) * (tablej - tablemj2 - 1)) * (
                             N + tablej + 1) / tablej / (2 * tablej + 1)
        return gamma7

    def Fgamma6(tablej, tablemj1, tablemj2, N):
        gamma6 = 0.5 * gamma_z * np.sqrt(
            (tablej + tablemj1 + 1) * (tablej - tablemj1 + 1) * (tablej + tablemj2 + 1) * (tablej - tablemj2 + 1)) * (
                             N - tablej) / (tablej + 1) / (2 * tablej + 1)
        return gamma6

    def Fgamma4(tablej, tablemj1, tablemj2, N):
        gamma4 = 0.5 * gamma_m * np.sqrt(
            (tablej - tablemj1 + 1) * (tablej - tablemj1 + 2) * (tablej - tablemj2 + 1) * (tablej - tablemj2 + 2)) * (
                             N - tablej) / (tablej + 1) / (2 * tablej + 1)
        return gamma4

    def Fgamma9(tablej, tablemj1, tablemj2, N):
        gamma9 = 0.5 * gamma_p * np.sqrt(
            (tablej + tablemj1 + 1) * (tablej + tablemj1 + 2) * (tablej + tablemj2 + 1) * (tablej + tablemj2 + 2)) * (
                             N - tablej) / (tablej + 1) / (2 * tablej + 1)
        return gamma9

    for i in range(0, dimension):
        resultmat.append([])
        resultmatloc.append([])
        for j in range(0, dimension):
            # p(j,m,m')
            if (i == j):
                gamma1 = Fgamma1(tablej[i], tablemj1[i], tablemj2[i], N)
                if (abs(gamma1) >= 1e-10):
                    resultmat[i].append(-gamma1)
                    resultmatloc[i].append(j)
            elif (tablej[i] == tablej[j]):
                # p(j,m+1,m'+1)
                if (tablemj1[i] + 1 == tablemj1[j] and tablemj2[i] + 1 == tablemj2[j]):
                    gamma2 = Fgamma2(tablej[i], tablemj1[i] + 1, tablemj2[i] + 1, N)
                    if (abs(gamma2) >= 1e-10):
                        resultmat[i].append(gamma2)
                        resultmatloc[i].append(j)
                # p(j,m-1,m'-1)
                elif (tablemj1[i] - 1 == tablemj1[j] and tablemj2[i] - 1 == tablemj2[j]):
                    gamma8 = Fgamma8(tablej[i], tablemj1[i] - 1, tablemj2[i] - 1, N)
                    if (abs(gamma8) >= 1e-10):
                        resultmat[i].append(gamma8)
                        resultmatloc[i].append(j)

            elif (tablej[i] + 1 == tablej[j]):
                # p(j+1,m,m')
                if (tablemj1[i] == tablemj1[j] and tablemj2[i] == tablemj2[j]):
                    gamma5 = Fgamma5(tablej[i] + 1, tablemj1[i], tablemj2[i], N)
                    if (abs(gamma5) >= 1e-10):
                        resultmat[i].append(gamma5)
                        resultmatloc[i].append(j)
                # p(j+1,m+1,m'+1)
                elif (tablemj1[i] + 1 == tablemj1[j] and tablemj2[i] + 1 == tablemj2[j]):
                    gamma3 = Fgamma3(tablej[i] + 1, tablemj1[i] + 1, tablemj2[i] + 1, N)
                    if (abs(gamma3) >= 1e-10):
                        resultmat[i].append(gamma3)
                        resultmatloc[i].append(j)
                # p(j+1,m-1,m'-1)
                elif (tablemj1[i] - 1 == tablemj1[j] and tablemj2[i] - 1 == tablemj2[j]):
                    gamma7 = Fgamma7(tablej[i] + 1, tablemj1[i] - 1, tablemj2[i] - 1, N)
                    if (abs(gamma7) >= 1e-10):
                        resultmat[i].append(gamma7)
                        resultmatloc[i].append(j)
            elif (tablej[i] - 1 == tablej[j]):  # and tablej[j]!=0):
                # p(j-1,m,m')
                if (tablemj1[i] == tablemj1[j] and tablemj2[i] == tablemj2[j]):
                    gamma6 = Fgamma6(tablej[i] - 1, tablemj1[i], tablemj2[i], N)
                    if (abs(gamma6) >= 1e-10):
                        resultmat[i].append(gamma6)
                        resultmatloc[i].append(j)
                # p(j-1,m+1,m'+1)
                elif (tablemj1[i] + 1 == tablemj1[j] and tablemj2[i] + 1 == tablemj2[j]):
                    gamma4 = Fgamma4(tablej[i] - 1, tablemj1[i] + 1, tablemj2[i] + 1, N)
                    if (abs(gamma4) >= 1e-10):
                        resultmat[i].append(gamma4)
                        resultmatloc[i].append(j)
                # p(j-1,m-1,m'-1)
                elif (tablemj1[i] - 1 == tablemj1[j] and tablemj2[i] - 1 == tablemj2[j]):
                    gamma9 = Fgamma9(tablej[i] - 1, tablemj1[i] - 1, tablemj2[i] - 1, N)
                    if (abs(gamma9) >= 1e-10):
                        resultmat[i].append(gamma9)
                        resultmatloc[i].append(j)
    del tablej, tablemj1, tablemj2
    return resultmat, resultmatloc, resultmat2, resultmatloc2, dimension


# dephasing along all axes
def axes_DisMat(gamma_x, gamma_y, gamma_z, N):
    #table for j, mj
    tablej=[]
    tablemj1=[]
    tablemj2=[]
    for i in range(0,N+1):
        for j in range(0,2*i+1):
            for k in range(j,2*i+1):
                tablej.append(i)
                tablemj1.append(j-i)
                tablemj2.append(k-i)
    dimension=len(tablej)
    resultmat=[]
    resultmatloc=[]
    resultmat2=[]
    resultmatloc2=[]
    
    def Fgamma1(tablej,tablemj1,tablemj2,N):
        gamma1=0.5*(gamma_x+gamma_y)*N
        gamma1+=0.5*(gamma_z)*N
        if(tablej!=0):
            gamma1+=0.5*gamma_z*(-tablemj1*tablemj2*(N+1)/tablej/(tablej+1))
        #print(gamma1)
        return gamma1
        
    def Fgamma2(tablej,tablemj1,tablemj2,N):
        gamma2=0.5*(gamma_x+gamma_y)/4.*np.sqrt((tablej+tablemj1)*(tablej-tablemj1+1)*(tablej+tablemj2)*(tablej-tablemj2+1))*(N+1)/tablej/(tablej+1)
        return gamma2
        
    def Fgamma8(tablej,tablemj1,tablemj2,N):
        gamma8=0.5*(gamma_x+gamma_y)/4.*np.sqrt((tablej-tablemj1)*(tablej+tablemj1+1)*(tablej-tablemj2)*(tablej+tablemj2+1))*(N+1)/tablej/(tablej+1)
        return gamma8
        
    def Fgamma5(tablej,tablemj1,tablemj2,N):
        gamma5=0.5*gamma_z*np.sqrt((tablej+tablemj1)*(tablej-tablemj1))*np.sqrt((tablej+tablemj2)*(tablej-tablemj2))*(N+tablej+1)/tablej/(2*tablej+1)
        return gamma5
        
    def Fgamma3(tablej,tablemj1,tablemj2,N):
        gamma3=0.5*(gamma_x+gamma_y)/4.*np.sqrt((tablej+tablemj1)*(tablej+tablemj1-1)*(tablej+tablemj2)*(tablej+tablemj2-1))*(N+tablej+1)/tablej/(2*tablej+1)
        return gamma3
        
    def Fgamma7(tablej,tablemj1,tablemj2,N):
        gamma7=0.5*(gamma_x+gamma_y)/4.*np.sqrt((tablej-tablemj1)*(tablej-tablemj1-1)*(tablej-tablemj2)*(tablej-tablemj2-1))*(N+tablej+1)/tablej/(2*tablej+1)
        return gamma7
                
    def Fgamma6(tablej,tablemj1,tablemj2,N):
        gamma6=0.5*gamma_z*np.sqrt((tablej+tablemj1+1)*(tablej-tablemj1+1)*(tablej+tablemj2+1)*(tablej-tablemj2+1))*(N-tablej)/(tablej+1)/(2*tablej+1)
        return gamma6
                    
    def Fgamma4(tablej,tablemj1,tablemj2,N):
        gamma4=0.5*(gamma_x+gamma_y)/4.*np.sqrt((tablej-tablemj1+1)*(tablej-tablemj1+2)*(tablej-tablemj2+1)*(tablej-tablemj2+2))*(N-tablej)/(tablej+1)/(2*tablej+1)
        return gamma4
                        
    def Fgamma9(tablej,tablemj1,tablemj2,N):
        gamma9=0.5*(gamma_x+gamma_y)/4.*np.sqrt((tablej+tablemj1+1)*(tablej+tablemj1+2)*(tablej+tablemj2+1)*(tablej+tablemj2+2))*(N-tablej)/(tablej+1)/(2*tablej+1)
        return gamma9
        
        
    def Fgamma13(tablej,tablemj1,tablemj2,N):
        gamma13=0.5*(gamma_x-gamma_y)/4.*np.sqrt((tablej-tablemj1)*(tablej+tablemj1+1)*(tablej+tablemj2)*(tablej-tablemj2+1))*(N+1)/tablej/(tablej+1)
        return gamma13
        
    def Fgamma14(tablej,tablemj1,tablemj2,N):
        gamma14=0.5*(gamma_x-gamma_y)/4.*(-1)*np.sqrt((tablej-tablemj1)*(tablej-tablemj1-1)*(tablej+tablemj2)*(tablej+tablemj2-1))*(N+tablej+1)/tablej/(2*tablej+1)
        return gamma14
        
    def Fgamma15(tablej,tablemj1,tablemj2,N):
        gamma15=0.5*(gamma_x-gamma_y)/4.*(-1)*np.sqrt((tablej+tablemj1+1)*(tablej+tablemj1+2)*(tablej-tablemj2+1)*(tablej-tablemj2+2))*(N-tablej)/(tablej+1)/(2*tablej+1)
        return gamma15
        
    def Fgamma10(tablej,tablemj1,tablemj2,N):
        gamma10=0.5*(gamma_x-gamma_y)/4.*np.sqrt((tablej+tablemj1)*(tablej-tablemj1+1)*(tablej-tablemj2)*(tablej+tablemj2+1))*(N+1)/tablej/(tablej+1)
        return gamma10
        
    def Fgamma11(tablej,tablemj1,tablemj2,N):
        gamma11=0.5*(gamma_x-gamma_y)/4.*(-1)*np.sqrt((tablej+tablemj1)*(tablej+tablemj1-1)*(tablej-tablemj2)*(tablej-tablemj2-1))*(N+tablej+1)/tablej/(2*tablej+1)
        return gamma11
        
    def Fgamma12(tablej,tablemj1,tablemj2,N):
        gamma12=0.5*(gamma_x-gamma_y)/4.*(-1)*np.sqrt((tablej-tablemj1+1)*(tablej-tablemj1+2)*(tablej+tablemj2+1)*(tablej+tablemj2+2))*(N-tablej)/(tablej+1)/(2*tablej+1)
        return gamma12
    
    for i in range(0,dimension):
        resultmat.append([])
        resultmatloc.append([])
        resultmat2.append([])
        resultmatloc2.append([])
        for j in range(0,dimension):
            #p(j,m,m')
            if(i==j):
                gamma1=Fgamma1(tablej[i],tablemj1[i],tablemj2[i],N)
                if(abs(gamma1)>=1e-10):
                    resultmat[i].append(-gamma1)
                    resultmatloc[i].append(j)
            if(tablej[i]==tablej[j]):
                #p(j,m+1,m'+1)
                if(tablemj1[i]+1==tablemj1[j] and tablemj2[i]+1==tablemj2[j]):
                    gamma2=Fgamma2(tablej[i],tablemj1[i]+1,tablemj2[i]+1,N)
                    if(abs(gamma2)>=1e-10):
                        resultmat[i].append(gamma2)
                        resultmatloc[i].append(j)
                #p(j,m-1,m'-1)
                if(tablemj1[i]-1==tablemj1[j] and tablemj2[i]-1==tablemj2[j]):
                    gamma8=Fgamma8(tablej[i],tablemj1[i]-1,tablemj2[i]-1,N)
                    if(abs(gamma8)>=1e-10):
                        resultmat[i].append(gamma8)
                        resultmatloc[i].append(j)
                #p(j,m+1,m'-1)
                if(tablemj1[i]+1==tablemj1[j] and tablemj2[i]-1==tablemj2[j]):
                    gamma10=Fgamma10(tablej[i],tablemj1[i]+1,tablemj2[i]-1,N)
                    if(abs(gamma10)>=1e-10):
                        resultmat[i].append(gamma10)
                        resultmatloc[i].append(j)
                if(tablemj1[i]==tablemj2[i] or tablemj1[i]+1==tablemj2[i]):
                #p(j,m+1,m'-1)
                    if(tablemj1[i]+1==tablemj2[j] and tablemj2[i]-1==tablemj1[j]):
                        gamma10=Fgamma10(tablej[i],tablemj1[i]+1,tablemj2[i]-1,N)
                        if(abs(gamma10)>=1e-10):
                            resultmat2[i].append(gamma10)
                            resultmatloc2[i].append(j)
                #p(j,m-1,m'+1)
                if(tablemj1[i]-1==tablemj1[j] and tablemj2[i]+1==tablemj2[j]):
                    gamma13=Fgamma13(tablej[i],tablemj1[i]-1,tablemj2[i]+1,N)
                    if(abs(gamma13)>=1e-10):
                        resultmat[i].append(gamma13)
                        resultmatloc[i].append(j)
                        

                        
                        
            elif(tablej[i]+1==tablej[j]):
                #p(j+1,m,m')
                if(tablemj1[i]==tablemj1[j] and tablemj2[i]==tablemj2[j]):
                    gamma5=Fgamma5(tablej[i]+1,tablemj1[i],tablemj2[i],N)
                    if(abs(gamma5)>=1e-10):
                        resultmat[i].append(gamma5)
                        resultmatloc[i].append(j)
                #p(j+1,m+1,m'+1)
                if(tablemj1[i]+1==tablemj1[j] and tablemj2[i]+1==tablemj2[j]):
                    gamma3=Fgamma3(tablej[i]+1,tablemj1[i]+1,tablemj2[i]+1,N)
                    if(abs(gamma3)>=1e-10):
                        resultmat[i].append(gamma3)
                        resultmatloc[i].append(j)
                #p(j+1,m-1,m'-1)
                if(tablemj1[i]-1==tablemj1[j] and tablemj2[i]-1==tablemj2[j]):
                    gamma7=Fgamma7(tablej[i]+1,tablemj1[i]-1,tablemj2[i]-1,N)
                    if(abs(gamma7)>=1e-10):
                        resultmat[i].append(gamma7)
                        resultmatloc[i].append(j)
                #p(j+1,m+1,m'-1)
                if(tablemj1[i]+1==tablemj1[j] and tablemj2[i]-1==tablemj2[j]):
                    gamma11=Fgamma11(tablej[i]+1,tablemj1[i]+1,tablemj2[i]-1,N)
                    if(abs(gamma11)>=1e-10):
                        resultmat[i].append(gamma11)
                        resultmatloc[i].append(j)
                if(tablemj1[i]==tablemj2[i] or tablemj1[i]+1==tablemj2[i]):
                #p(j+1,m+1,m'-1)
                    if(tablemj1[i]+1==tablemj2[j] and tablemj2[i]-1==tablemj1[j]):
                        gamma11=Fgamma11(tablej[i]+1,tablemj1[i]+1,tablemj2[i]-1,N)
                        if(abs(gamma11)>=1e-10):
                            resultmat2[i].append(gamma11)
                            resultmatloc2[i].append(j)
                #p(j+1,m-1,m'+1)
                if(tablemj1[i]-1==tablemj1[j] and tablemj2[i]+1==tablemj2[j]):
                    gamma14=Fgamma14(tablej[i]+1,tablemj1[i]-1,tablemj2[i]+1,N)
                    if(abs(gamma14)>=1e-10):
                        resultmat[i].append(gamma14)
                        resultmatloc[i].append(j)
                        
                
                        
            elif(tablej[i]-1==tablej[j]):# and tablej[j]!=0):
                #p(j-1,m,m')
                if(tablemj1[i]==tablemj1[j] and tablemj2[i]==tablemj2[j]):
                    gamma6=Fgamma6(tablej[i]-1,tablemj1[i],tablemj2[i],N)
                    if(abs(gamma6)>=1e-10):
                        resultmat[i].append(gamma6)
                        resultmatloc[i].append(j)
                #p(j-1,m+1,m'+1)
                if(tablemj1[i]+1==tablemj1[j] and tablemj2[i]+1==tablemj2[j]):
                    gamma4=Fgamma4(tablej[i]-1,tablemj1[i]+1,tablemj2[i]+1,N)
                    if(abs(gamma4)>=1e-10):
                        resultmat[i].append(gamma4)
                        resultmatloc[i].append(j)
                #p(j-1,m-1,m'-1)
                if(tablemj1[i]-1==tablemj1[j] and tablemj2[i]-1==tablemj2[j]):
                    gamma9=Fgamma9(tablej[i]-1,tablemj1[i]-1,tablemj2[i]-1,N)
                    if(abs(gamma9)>=1e-10):
                        resultmat[i].append(gamma9)
                        resultmatloc[i].append(j)
                #p(j-1,m+1,m'-1)
                if(tablemj1[i]+1==tablemj1[j] and tablemj2[i]-1==tablemj2[j]):
                    gamma12=Fgamma12(tablej[i]-1,tablemj1[i]+1,tablemj2[i]-1,N)
                    if(abs(gamma12)>=1e-10):
                        #print(gamma12)
                        resultmat[i].append(gamma12)
                        resultmatloc[i].append(j)
                if(tablemj1[i]==tablemj2[i] or tablemj1[i]+1==tablemj2[i]):
                #p(j-1,m+1,m'-1)
                    if(tablemj1[i]+1==tablemj2[j] and tablemj2[i]-1==tablemj1[j]):
                        gamma12=Fgamma12(tablej[i]-1,tablemj1[i]+1,tablemj2[i]-1,N)
                        if(abs(gamma12)>=1e-10):
                        #print(gamma12)
                            resultmat2[i].append(gamma12)
                            resultmatloc2[i].append(j)
                #p(j-1,m-1,m'+1)
                if(tablemj1[i]-1==tablemj1[j] and tablemj2[i]+1==tablemj2[j]):
                    gamma15=Fgamma15(tablej[i]-1,tablemj1[i]-1,tablemj2[i]+1,N)
                    if(abs(gamma15)>=1e-10):
                        resultmat[i].append(gamma15)
                        resultmatloc[i].append(j)
                        
                        
    del tablej,tablemj1,tablemj2
    return resultmat,resultmatloc,resultmat2,resultmatloc2,dimension
                
#Hamiltonian matrix
def OATMat(chi, N): #H=\chi Sz^2
    #table for j, mj
    tablej=[]
    tablemj1=[]
    tablemj2=[]
    for i in range(0,N+1):
        for j in range(0,2*i+1):
            for k in range(j,2*i+1):
                tablej.append(i)
                tablemj1.append(j-i)
                tablemj2.append(k-i)
    dimension=len(tablej)
    resultmat=[]
    resultmatloc=[]
    resultmat2=[]
    resultmatloc2=[]
    for i in range(0,dimension):
        resultmat.append([])
        resultmat2.append([])
        resultmatloc.append([])
        resultmatloc2.append([])
        for j in range(0,dimension):
            #p(j,m,m') only for Sz^2
            if(i==j):
                gamma0=chi*(tablemj2[i]*tablemj2[i]-tablemj1[i]*tablemj1[i])*1j
                if(abs(gamma0)>=1e-10):
                    resultmat[i].append(gamma0)
                    resultmatloc[i].append(j)
    del tablej,tablemj1,tablemj2
    return resultmat,resultmatloc,resultmat2,resultmatloc2,dimension

def TATMat(chi, N): #H=\chi (SxSy+SySx)
    #table for j, mj
    tablej=[]
    tablemj1=[]
    tablemj2=[]
    for i in range(0,N+1):
        for j in range(0,2*i+1):
            for k in range(j,2*i+1):
                tablej.append(i)
                tablemj1.append(j-i)
                tablemj2.append(k-i)
    dimension=len(tablej)
    resultmat=[]
    resultmatloc=[]
    resultmat2=[]
    resultmatloc2=[]
    for i in range(0,dimension):
        resultmat.append([])
        resultmat2.append([])
        resultmatloc.append([])
        resultmatloc2.append([])
        for j in range(0,dimension):
            if(tablej[i]==tablej[j]):
            #p(j,m-2,m')
                if(tablemj1[i]-2==tablemj1[j] and tablemj2[i]==tablemj2[j]):
                    gamma0=-0.5*chi*np.sqrt((tablej[i]*(tablej[i]+1)-(tablemj1[i]-2)*(tablemj1[i]-1))*(tablej[i]*(tablej[i]+1)-(tablemj1[i]-1)*tablemj1[i]))
                    if(abs(gamma0)>=1e-10):
                        # TODO: MODIFIED gamma0 -> gamma0/2 TO RETURN HALF OF THE HAMILTONIAN
                        resultmat[i].append(gamma0/2)
                        resultmatloc[i].append(j)
            #p(j,m+2,m')
                if(tablemj1[i]+2==tablemj1[j] and tablemj2[i]==tablemj2[j]):
                    gamma0=0.5*chi*np.sqrt((tablej[i]*(tablej[i]+1)-(tablemj1[i]+2)*(tablemj1[i]+1))*(tablej[i]*(tablej[i]+1)-(tablemj1[i]+1)*tablemj1[i]))
                    if(abs(gamma0)>=1e-10):
                        # TODO: MODIFIED gamma0 -> gamma0/2 TO RETURN HALF OF THE HAMILTONIAN
                        resultmat[i].append(gamma0/2)
                        resultmatloc[i].append(j)
            #p(j,m,m'-2)
                if(tablemj1[i]==tablemj1[j] and tablemj2[i]-2==tablemj2[j]):
                    gamma0=-0.5*chi*np.sqrt((tablej[i]*(tablej[i]+1)-(tablemj2[i]-2)*(tablemj2[i]-1))*(tablej[i]*(tablej[i]+1)-(tablemj2[i]-1)*tablemj2[i]))
                    if(abs(gamma0)>=1e-10):
                        # TODO: MODIFIED gamma0 -> gamma0/2 TO RETURN HALF OF THE HAMILTONIAN
                        resultmat[i].append(gamma0/2)
                        resultmatloc[i].append(j)
             #p(j,m,m'+2)
                if(tablemj1[i]==tablemj1[j] and tablemj2[i]+2==tablemj2[j]):
                    gamma0=0.5*chi*np.sqrt((tablej[i]*(tablej[i]+1)-(tablemj2[i]+2)*(tablemj2[i]+1))*(tablej[i]*(tablej[i]+1)-(tablemj2[i]+1)*tablemj2[i]))
                    if(abs(gamma0)>=1e-10):
                        # TODO: MODIFIED gamma0 -> gamma0/2 TO RETURN HALF OF THE HAMILTONIAN
                        resultmat[i].append(gamma0/2)
                        resultmatloc[i].append(j)
            
                if(tablemj1[i]==tablemj2[i] or tablemj1[i]+1==tablemj2[i]):
                #p(j,m+2,m')
                    if(tablemj1[i]+2==tablemj2[j] and tablemj2[i]==tablemj1[j]):
                        gamma0=0.5*chi*np.sqrt((tablej[i]*(tablej[i]+1)-(tablemj1[i]+2)*(tablemj1[i]+1))*(tablej[i]*(tablej[i]+1)-(tablemj1[i]+1)*tablemj1[i]))
                        if(abs(gamma0)>=1e-10):
                            # TODO: MODIFIED gamma0 -> gamma0/2 TO RETURN HALF OF THE HAMILTONIAN
                            resultmat2[i].append(gamma0/2)
                            resultmatloc2[i].append(j)
                #p(j,m,m'-2)
                    elif(tablemj1[i]==tablemj2[j] and tablemj2[i]-2==tablemj1[j]):
                        gamma0=-0.5*chi*np.sqrt((tablej[i]*(tablej[i]+1)-(tablemj2[i]-2)*(tablemj2[i]-1))*(tablej[i]*(tablej[i]+1)-(tablemj2[i]-1)*tablemj2[i]))
                        if(abs(gamma0)>=1e-10):
                            # TODO: MODIFIED gamma0 -> gamma0/2 TO RETURN HALF OF THE HAMILTONIAN
                            resultmat2[i].append(gamma0/2)
                            resultmatloc2[i].append(j)
                        
    del tablej,tablemj1,tablemj2
    return resultmat,resultmatloc,resultmat2,resultmatloc2, dimension

def LMGMat(chi, Omega, N): #-chi/(2N)*Sx^2+Omega*S_z
    #table for j, mj
    chi=chi/2
    tablej=[]
    tablemj1=[]
    tablemj2=[]
    for i in range(0,N+1):
        for j in range(0,2*i+1):
            for k in range(j,2*i+1):
                tablej.append(i)
                tablemj1.append(j-i)
                tablemj2.append(k-i)
    dimension=len(tablej)
    resultmat=[]
    resultmatloc=[]
    resultmat2=[]
    resultmatloc2=[]
    for i in range(0,dimension):
        resultmat.append([])
        resultmat2.append([])
        resultmatloc.append([])
        resultmatloc2.append([])
        for j in range(0,dimension):
            if(i==j):
            #p(j,m,m')
                gamma0=-1j*Omega*(tablemj1[i]-tablemj2[i])
                gamma0+=0.5*1j*chi/N*(tablemj2[i]*tablemj2[i]-tablemj1[i]*tablemj1[i])
                if(abs(gamma0)>=1e-10):
                    resultmat[i].append(gamma0)
                    resultmatloc[i].append(j)
            elif(tablej[i]==tablej[j]):
            #p(j,m-2,m')
                if(tablemj1[i]-2==tablemj1[j] and tablemj2[i]==tablemj2[j]):
                    gamma0=0.25*1j*chi/N*np.sqrt((tablej[i]*(tablej[i]+1)-(tablemj1[i]-2)*(tablemj1[i]-1))*(tablej[i]*(tablej[i]+1)-(tablemj1[i]-1)*tablemj1[i]))
                    if(abs(gamma0)>=1e-10):
                        resultmat[i].append(gamma0)
                        resultmatloc[i].append(j)
            #p(j,m+2,m')
                if(tablemj1[i]+2==tablemj1[j] and tablemj2[i]==tablemj2[j]):
                    gamma0=0.25*1j*chi/N*np.sqrt((tablej[i]*(tablej[i]+1)-(tablemj1[i]+2)*(tablemj1[i]+1))*(tablej[i]*(tablej[i]+1)-(tablemj1[i]+1)*tablemj1[i]))
                    if(abs(gamma0)>=1e-10):
                        resultmat[i].append(gamma0)
                        resultmatloc[i].append(j)
            #p(j,m,m'-2)
                if(tablemj1[i]==tablemj1[j] and tablemj2[i]-2==tablemj2[j]):
                    gamma0=-0.25*1j*chi/N*np.sqrt((tablej[i]*(tablej[i]+1)-(tablemj2[i]-2)*(tablemj2[i]-1))*(tablej[i]*(tablej[i]+1)-(tablemj2[i]-1)*tablemj2[i]))
                    if(abs(gamma0)>=1e-10):
                        resultmat[i].append(gamma0)
                        resultmatloc[i].append(j)
             #p(j,m,m'+2)
                if(tablemj1[i]==tablemj1[j] and tablemj2[i]+2==tablemj2[j]):
                    gamma0=-0.25*1j*chi/N*np.sqrt((tablej[i]*(tablej[i]+1)-(tablemj2[i]+2)*(tablemj2[i]+1))*(tablej[i]*(tablej[i]+1)-(tablemj2[i]+1)*tablemj2[i]))
                    if(abs(gamma0)>=1e-10):
                        resultmat[i].append(gamma0)
                        resultmatloc[i].append(j)

                if(tablemj1[i]==tablemj2[i] or tablemj1[i]+1==tablemj2[i]):
                #p(j,m+2,m')
                    if(tablemj1[i]+2==tablemj2[j] and tablemj2[i]==tablemj1[j]):
                        gamma0=0.25*1j*chi/N*np.sqrt((tablej[i]*(tablej[i]+1)-(tablemj1[i]+2)*(tablemj1[i]+1))*(tablej[i]*(tablej[i]+1)-(tablemj1[i]+1)*tablemj1[i]))
                        if(abs(gamma0)>=1e-10):
                            resultmat2[i].append(gamma0)
                            resultmatloc2[i].append(j)
                #p(j,m,m'-2)
                    elif(tablemj1[i]==tablemj2[j] and tablemj2[i]-2==tablemj1[j]):
                        gamma0=-0.25*1j*chi/N*np.sqrt((tablej[i]*(tablej[i]+1)-(tablemj2[i]-2)*(tablemj2[i]-1))*(tablej[i]*(tablej[i]+1)-(tablemj2[i]-1)*tablemj2[i]))
                        if(abs(gamma0)>=1e-10):
                            resultmat2[i].append(gamma0)
                            resultmatloc2[i].append(j)
                        
    del tablej,tablemj1,tablemj2
    return resultmat,resultmatloc, resultmat2,resultmatloc2,dimension

def LMGMat2(chi, Omega, N): #-chi/(2N)*Sz^2-Omega*S_x
    #table for j, mj
    #chi=chi/2
    tablej=[]
    tablemj1=[]
    tablemj2=[]
    for i in range(0,N+1):
        for j in range(0,2*i+1):
            for k in range(j,2*i+1):
                tablej.append(i)
                tablemj1.append(j-i)
                tablemj2.append(k-i)
    dimension=len(tablej)
    resultmat=[]
    resultmatloc=[]
    resultmat2=[]
    resultmatloc2=[]
    for i in range(0,dimension):
        resultmat.append([])
        resultmatloc.append([])
        resultmat2.append([])
        resultmatloc2.append([])
        for j in range(0,dimension):
            if(i==j):
            #p(j,m,m')
                gamma0=0.5*1j*chi/N*(tablemj1[i]*tablemj1[i]-tablemj2[i]*tablemj2[i])
                if(abs(gamma0)>=1e-10):
                    resultmat[i].append(gamma0)
                    resultmatloc[i].append(j)
            elif(tablej[i]==tablej[j]):
            #p(j,m-1,m')
                if(tablemj1[i]-1==tablemj1[j] and tablemj2[i]==tablemj2[j]):
                    gamma0=0.5*1j*Omega*np.sqrt(tablej[i]*(tablej[i]+1)-(tablemj1[i]-1)*tablemj1[i])
                    if(abs(gamma0)>=1e-10):
                        resultmat[i].append(gamma0)
                        resultmatloc[i].append(j)
            #p(j,m+1,m')
                if(tablemj1[i]+1==tablemj1[j] and tablemj2[i]==tablemj2[j]):
                    gamma0=0.5*1j*Omega*np.sqrt(tablej[i]*(tablej[i]+1)-(tablemj1[i]+1)*tablemj1[i])
                    if(abs(gamma0)>=1e-10):
                        resultmat[i].append(gamma0)
                        resultmatloc[i].append(j)
            #p(j,m,m'-1)
                if(tablemj1[i]==tablemj1[j] and tablemj2[i]-1==tablemj2[j]):
                    gamma0=-0.5*1j*Omega*np.sqrt(tablej[i]*(tablej[i]+1)-(tablemj2[i]-1)*tablemj2[i])
                    if(abs(gamma0)>=1e-10):
                        resultmat[i].append(gamma0)
                        resultmatloc[i].append(j)
             #p(j,m,m'+1)
                if(tablemj1[i]==tablemj1[j] and tablemj2[i]+1==tablemj2[j]):
                    gamma0=-0.5*1j*Omega*np.sqrt(tablej[i]*(tablej[i]+1)-(tablemj2[i]+1)*tablemj2[i])
                    if(abs(gamma0)>=1e-10):
                        resultmat[i].append(gamma0)
                        resultmatloc[i].append(j)
            
                if(tablemj1[i]==tablemj2[i]):
                    #p(j,m,m'-1)
                    if(tablemj1[i]==tablemj2[j] and tablemj2[i]-1==tablemj1[j]):
                        gamma0=-0.5*1j*Omega*np.sqrt(tablej[i]*(tablej[i]+1)-(tablemj2[i]-1)*tablemj2[i])
                        if(abs(gamma0)>=1e-10):
                            resultmat2[i].append(gamma0)
                            resultmatloc2[i].append(j)
                    #p(j,m+1,m')
                    elif(tablemj1[i]+1==tablemj2[j] and tablemj2[i]==tablemj1[j]):
                        gamma0=0.5*1j*Omega*np.sqrt(tablej[i]*(tablej[i]+1)-(tablemj1[i]+1)*tablemj1[i])
                        if(abs(gamma0)>=1e-10):
                            resultmat2[i].append(gamma0)
                            resultmatloc2[i].append(j)
             
                        
    del tablej,tablemj1,tablemj2
    return resultmat,resultmatloc,resultmat2,resultmatloc2,dimension


def Perm_solver(rho0, tmax, Dmat, Dmatloc, Dmat2, Dmatloc2, Hmat, Hmatloc, Hmat2, Hmatloc2, dimension, Ntime):
    def func(t, rho, Dmat, Dmatloc, Dmat2, Dmatloc2, Hmat, Hmatloc, Hmat2, Hmatloc2, dimension):
        drhodt=np.zeros(dimension, dtype=np.complex128)
        for i in range(0,dimension):
            for j in range(0,len(Dmat[i])):
                drhodt[i]+=rho[Dmatloc[i][j]]*Dmat[i][j]
            for j in range(0,len(Dmat2[i])):
                drhodt[i]+=np.conj(rho[Dmatloc2[i][j]])*Dmat2[i][j]
            for j in range(0,len(Hmat[i])):
                drhodt[i]+=rho[Hmatloc[i][j]]*Hmat[i][j]
            for j in range(0,len(Hmat2[i])):
                drhodt[i]+=np.conj(rho[Hmatloc2[i][j]])*Hmat2[i][j]
        return drhodt
        '''
        #drhodt=np.zeros(dimension, dtype=np.complex128)
        drhodt=[]
        for i in range(0,dimension):
            drhodt.append(0+0j)
            for j in range(0,len(Dmat[i])):
                drhodt[i]+=rho[Dmatloc[i][j]]*Dmat[i][j]
                """
                if USE_DIFFRAX == False:
                    drhodt[i]+=rho[Dmatloc[i][j]]*Dmat[i][j]
                else:
                    drhodt = drhodt.at[i].add(rho[Dmatloc[i][j]]*Dmat[i][j])
                """
            for j in range(0,len(Hmat[i])):
                drhodt[i]+=rho[Hmatloc[i][j]]*Hmat[i][j]
                """
                if USE_DIFFRAX == False:
                    drhodt[i]+=rho[Hmatloc[i][j]]*Hmat[i][j]
                else:
                    drhodt = drhodt.at[i].add(rho[Hmatloc[i][j]]*Hmat[i][j])
                """
        #print("drhodt.shape", drhodt.shape)
        return drhodt
        '''

    if USE_DIFFRAX == False:
        atol=1e-10
        rtol=1e-10
        #method = DEFAULT_INTEGRATION_METHOD
        teval=np.linspace(0, tmax, Ntime+1, endpoint=True)
        sol = solve_ivp(func, [0,tmax], rho0,  args=(Dmat, Dmatloc, Dmat2, Dmatloc2, Hmat, Hmatloc, Hmat2, Hmatloc2, dimension),t_eval=teval,rtol=rtol,
            atol=atol)#,method=method)
        return sol
    else:

        def _func(t, rho, args):
            return func(t, rho, args[0], args[1], args[2], args[3], args[4])
            """
            drhodt=np.zeros(dimension, dtype=np.complex128)
            for i in range(0,dimension):
                for j in range(0,len(Dmat[i])):
                    drhodt[i]+=rho[Dmatloc[i][j]]*Dmat[i][j]
                for j in range(0,len(Hmat[i])):
                    drhodt[i]+=rho[Hmatloc[i][j]]*Hmat[i][j]
            return drhodt
            """
        # set initial time step size
        diffrax_kwargs = {}
        diffrax_kwargs["dt0"] = tmax.real/Ntime

        term = diffrax.ODETerm(_func)
        solver = diffrax.Tsit5()  # try also diffrax.Dopri8()
        solver_args = dict(t0=0.0, t1=tmax.real, y0=rho0, args=(Dmat, Dmatloc,Hmat, Hmatloc, dimension))
        #if FORWARD_MODE:
        diffrax_kwargs["max_steps"] = diffrax_kwargs.get("max_steps", None)
        solver_args |= dict(adjoint=diffrax.DirectAdjoint())
        solution = diffrax.diffeqsolve(term, solver, **solver_args, **diffrax_kwargs)
        return solution.ys #[-1]
