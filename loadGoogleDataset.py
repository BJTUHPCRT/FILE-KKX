"""
Loading Google cluster-2011 dataset for experiment
"""
from __future__ import print_function

import sys
import os
import time

import numpy as np

import gzip


def load_eventsTable(path):

    files = os.listdir(path)
    job_events = []
    machine_attributes = []
    machine_events = []
    task_constraints = []
    task_events = []
    task_usage = []

    for file in files:
        if file == 'task_events':
            #if os.path.exists(path + "\\" + file) and os.path.isdir(path + "\\" + file):
            tempPath = path + "\\" + file
            subFiles = os.listdir(tempPath)
            for subFile in subFiles:
                #if os.path.exists(tempPath + "\\" + subFile):
                with gzip.open(tempPath + "\\" + subFile, 'rb') as f:
                    tempdata = f.read()
                    data = np.frombuffer(f.read(), np.uint8, offset=16)
                task_events.append(data)
                    # elif file == 'machine_attributes':
                    #     machine_attributes.append(data)
        elif file == 'machine_events':
            # if os.path.exists(path + "\\" + file) and os.path.isdir(path + "\\" + file):
            tempPath = path + "\\" + file
            subFiles = os.listdir(tempPath)
            for subFile in subFiles:
                # if os.path.exists(tempPath + "\\" + subFile):
                with gzip.open(tempPath + "\\" + subFile, 'rb') as f:
                    data = np.frombuffer(f.read(), np.uint8, offset=16)
                machine_events.append(data)
        # elif file == 'task_constraints':
            #     task_constraints.append(data)
        # elif file == 'job_events':
            #     job_events.append(data)
        # elif file == 'task_usage':
            #     task_usage.append(data)
         # else:
         #    print('the path [{}] is not exist!'.format(tempPath + "\\" + subFile))
        else:
            #print('the path [{}] is not exist!'.format(path))
            print('#################load datasets successful finished!####################')

    return task_events,machine_events
#
# path = "E:\experiment\clusterdata-2011-2"
# task_events, machine_events = load_eventsTable(path)
# print(machine_events,task_events)