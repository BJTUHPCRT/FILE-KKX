
import pandas as pa
import os
import sys

def test_workload():

    print(os.getcwd())
    # machine = pa.read_csv('part-00000-of-00500.csv',sep=',',encoding='utf-8',engine='python')
    # machine_event = machine.values
    #print(machine_event[0,:])

    # task = pa.read_csv('workload.csv', sep=',', encoding='utf-8', engine='python')
    # task = pa.read_csv('part-00001-of-00500.csv', sep=',', encoding='utf-8', engine='python')
    task0 = pa.read_csv('5.csv', sep=',', encoding='utf-8', engine='python')
    task1 = pa.read_csv('2.csv', sep=',', encoding='utf-8', engine='python')
    task_event0 = task0.values
    task_event1 = task1.values
    task_cpu_requests = []
    import math
    for i in range(len(task0)):
        if not math.isnan(task_event0[i, 9]):
            task_cpu_requests.append(task_event0[i, 9])
        else:
            task_cpu_requests.append(0)
    # for i in range(len(task1)):
    #     if not math.isnan(task_event1[i, 9]):
    #         task_cpu_requests.append(task_event1[i, 9])
    #     else:
    #         task_cpu_requests.append(0)
    #print(task_event[0, :])

    return task_cpu_requests

def test_workloads():
    print(os.getcwd())
    task0 = pa.read_csv('workload.csv', sep=',', encoding='utf-8', engine='python')
    # task1 = pa.read_csv('part-00000-of-00500.csv', sep=',', encoding='utf-8', engine='python')
    # task2 = pa.read_csv('part-00000-of-00500.csv', sep=',', encoding='utf-8', engine='python')
    # task3 = pa.read_csv('part-00000-of-00500.csv', sep=',', encoding='utf-8', engine='python')
    # task4 = pa.read_csv('part-00000-of-00500.csv', sep=',', encoding='utf-8', engine='python')

    task_event0 = task0.values
    # task_event1 = task1.values
    # task_event2 = task2.values
    # task_event3 = task3.values
    # task_event4 = task4.values

    #print(task_event[0, :])
    task_cpu_requests = []
    import math
    for i in range(len(task0)):
        if not math.isnan(task_event0[i, 9]):
            task_cpu_requests.append(task_event0[i, 9])
    # for i in range(len(task1)):
    #     if not math.isnan(task_event0[i, 9]):
    #         task_cpu_requests.append(task_event1[i, 9])
    # for i in range(len(task2)):
    #     if not math.isnan(task_event0[i, 9]):
    #         task_cpu_requests.append(task_event2[i, 9])
    # for i in range(len(task3)):
    #     if not math.isnan(task_event0[i, 9]):
    #         task_cpu_requests.append(task_event3[i, 9])
    # for i in range(len(task4)):
    #     if not math.isnan(task_event0[i, 9]):
    #         task_cpu_requests.append(task_event4[i, 9])

    import pandas
    import numpy as np
    import matplotlib.pyplot as plt
    # save results into file
    # dataframe = pandas.DataFrame(data=list(task_cpu_requests), index=None, columns=None)
    # dataframe.to_csv('task_0.csv', index=False, sep=',')
    plt.plot(np.arange(len(task_cpu_requests)), task_cpu_requests)
    plt.title('task')
    plt.ylabel('task')
    plt.xlabel('number of tasks')
    # plt.savefig('C:/Users/beauty/Desktop/DRL/experiment_eunm/resultPicture/waiting_time.png')
    plt.show()
# def change_detection(task_cpu_requests):

# test_workload()
