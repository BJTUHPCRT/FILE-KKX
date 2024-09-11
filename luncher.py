import math
import numpy as np
import matplotlib.pyplot as plt
import sys
import pandas as pa

def printWeekDataset_tmp(dataset,day,week):
    import matplotlib.pyplot as plt
    import pandas
    plt.clf()
    plt.plot(np.arange(len(dataset)), dataset)
    plt.title('workload')
    plt.ylabel('cpu_request')
    plt.xlabel('time slot')
    plt.show()

def Pettitt_change_point_detection(inputdata):
    inputdata = np.array(inputdata)
    n  = inputdata.shape[0]
    k = range(n)
    inputdataT = pa.Series(inputdata)
    r = inputdataT.rank()
    Uk = [2*np.sum(r[0:x])-x*(n + 1) for x in k]
    Uka = list(np.abs(Uk))
    U = np.max(Uka)
    K = Uka.index(U)
    pvalue = 2 * np.exp((-6 * (U**2))/(n**3 + n**2))
    if pvalue <= 0.05:
        change_point_desc = '显著'
    else:
        change_point_desc = '不显著'
    Pettitt_result = {'突变点位置':K,'突变程度':change_point_desc}
    return K, Pettitt_result
#--------------------------------------------
def Kendall_change_point_detection(inputdata):
    inputdata = np.array(inputdata)
    n=inputdata.shape[0]
   
    Sk             = [0]
   
    UFk            = [0]
   
    s              =  0
    Exp_value      = [0]
    Var_value      = [0]
   
    for i in range(1,n):
        for j in range(i):
            if inputdata[i] > inputdata[j]:
                s = s+1
            else:
                s = s+0
        Sk.append(s)
        Exp_value.append((i+1)*(i+2)/4 )                   
        Var_value.append((i+1)*i*(2*(i+1)+5)/72 )        
        UFk.append((Sk[i]-Exp_value[i])/np.sqrt(Var_value[i]))
   
    Sk2             = [0]
  
    UBk             = [0]
    UBk2            = [0]
  
    s2              =  0
    Exp_value2      = [0]
    Var_value2      = [0]
  
    inputdataT = list(reversed(inputdata))
    
    for i in range(1,n):
        for j in range(i):
            if inputdataT[i] > inputdataT[j]:
                s2 = s2+1
            else:
                s2 = s2+0
        Sk2.append(s2)
        Exp_value2.append((i+1)*(i+2)/4 )                    
        Var_value2.append((i+1)*i*(2*(i+1)+5)/72 )           
        UBk.append((Sk2[i]-Exp_value2[i])/np.sqrt(Var_value2[i]))
        UBk2.append(-UBk[i])
    
    UBkT = list(reversed(UBk2))
    diff = np.array(UFk) - np.array(UBkT)
    K    = list()
  
    for k in range(1,n):
        if diff[k-1]*diff[k]<0:
            K.append(k)
    
    plt.figure(figsize=(10,5))
    plt.plot(range(1,n+1) ,UFk  ,label='UFk') # UFk
    plt.plot(range(1,n+1) ,UBkT ,label='UBk') # UBk
    plt.ylabel('UFk-UBk')
    x_lim = plt.xlim()
    plt.plot(x_lim,[-1.96,-1.96],'m--',color='r')
    plt.plot(x_lim,[  0  ,  0  ],'m--')
    plt.plot(x_lim,[+1.96,+1.96],'m--',color='r')
    plt.legend(loc=2) 
    plt.show()
    return K

task_events = []
path = sys.path[0]
task0 = pa.read_csv(path + '/day_datas/9.csv', sep=',', encoding='utf-8', engine='python')

tasks = task0.values
for i in range(len(tasks)):
    task_events.append(tasks[i, 0] + 1)
# maax = len(task_events)
task_event = [math.log2(x) for x in task_events]
maax = 40

import five_regression
for i in range(0, len(task_event), maax):
    task_cpu_requests = task_event[i:i+maax]
    print(Pettitt_change_point_detection(task_cpu_requests)) 
    five_regression.fr(task_cpu_requests, str(i)) 
    printWeekDataset_tmp(task_cpu_requests,str(9), str(i)) 

