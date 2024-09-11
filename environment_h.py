"""
   environment is  simulating the datacenter,also as the DRL environment.
"""
from loadDataset import smalltest
from entities import machine
from entities import task
import numpy as np
import matplotlib.pyplot as plt
import math
import time
import logging
class environment:
    def __init__(self, end='no_new_task'):
        self.tasks = []
        self.task_reset = []
        self.current_task = None
        self.task_number = 0
        self.machines = []
        self.machines_reset = []
        self.action_len = 0
        self.state_len = 0
        self.total_power = []
        self.total_job_latency = []
        self.reward = []
        self.theta = 0.3 #trade-off of power and job latncy
        #self.task_allocated
        self.task_allocate_counter = 0
        #self.machine_available

        self.action_space = []

        self.unchange_task_number = 0 
        self.delaytime_sum = 0
        self.detection_number = 0
        self.stage_total_task_cpu = 0
        self.average_task_cpu = 0
        #initialize system
        task_events, machine_events = smalltest.test_workload()

        for i in range(3000):
            if not math.isnan(task_events[i, 9]):
                self.tasks.append(task.Task(task_events[i, :]))
            else:
                continue
   
        self.task_number = len(self.tasks)
        self.tasks.sort(key=lambda x: x.timestamp, reverse=False)  #workload
        self.task_reset = self.tasks

        machine_CPUCapcity = [0.5, 0.7, 0.6, 0.9, 0.3, 0.8, 0.5, 0.4, 0.8, 0.5,
                              0.5, 0.7, 0.6, 0.9, 0.3, 0.8, 0.5, 0.4, 0.8, 0.5]
        for i in range(20):
            machine_temp = machine.Machine(machine_events[i, :])
            machine_temp.machineID = i
            machine_temp.CPUCapacity = machine_CPUCapcity[i]
            self.machines.append(machine_temp)
            self.machines_reset.append(machine_temp)
            self.action_space.append(machine_temp.machineID)
        #self.machines_reset = self.machines
        self.action_len = len(self.action_space)

    def observe(self,task_new):
     
        #current task is a part of observation,and DQN take an acti  on of current from the observation
        self.current_task = task_new
        observation = np.zeros((1, self.action_len * 2 + 1)) #dtype=tf.float32
        if task_new is None:
            observation[0, 0] = 0.0
        else:
            observation[0, 0] = task_new.CPURequest
            # print("task cpu request:", task_new.CPURequest)
        #print(observation[0,0])
        count = len(self.machines)
        Index = 1
        for i in range(count):
            observation[0, Index] = self.machines[i].availableCPU
            observation[0, Index+1] = self.machines[i].utilization
            Index += 2
        # observation[0, 19] = self.machines[9].availableCPU
        self.state_len = len(observation)
        return observation


    def step(self,action):
        power = 0
        job_latency = 0
        done = False
        detection = False
        workload_changed = False

        current_task = self.current_task
        #we should judge if the task queue is empty
        if current_task != None:
            self.machines[action].allocate_task(current_task)
        next_task = None
        if self.task_allocate_counter == self.task_number:
            done = True
        else:
            next_task, detection = self.get_new_task()

        self.unchange_task_number += 1  
        if self.unchange_task_number == 1:
            self.stage_total_task_cpu = next_task.CPURequest
        if current_task != None and next_task != None:
            if detection:
                # workload_changed = self.detecte_change(next_task.CPURequest)
                self.unchange_task_number -= 1
                if workload_changed:
                    self.stage_total_task_cpu = next_task.CPURequest
                    self.unchange_task_number = 1
             #      print('skip')
            else:
                self.stage_total_task_cpu += next_task.CPURequest
                self.average_task_cpu = self.stage_total_task_cpu / self.unchange_task_number

        reward = self.get_reward()
        self.reward.append(reward)

        for i in range(len(self.machines)):
            power += self.machines[i].idlePower + self.machines[i].activePower * self.machines[i].utilization
        self.total_power.append(power)
        for i in range(len(self.machines)):
            for j in range(len(self.machines[i].running_task)):
                job_latency += 1

            for k in range(len(self.machines[i].waiting_task)):
                job_latency += 1
        self.total_job_latency.append(job_latency)

        return self.observe(next_task), reward, done, workload_changed  

    def detecte_change(self, neww_task):
        change = False
        up_bound = self.average_task_cpu + self.average_task_cpu * 0.3   
        low_bound = self.average_task_cpu - self.average_task_cpu * 0.3

        if (neww_task > up_bound ) or (neww_task < low_bound):
            self.detection_number += 1 
        else:
            self.detection_number = 0  

        if self.detection_number >= 3: change = True

        return  change

    def get_new_task(self):
        detection = False
        delaytime = np.random.poisson(2, 1)
        self.delaytime_sum += delaytime[0]
        if self.delaytime_sum >= 50:
            detection = True
            self.delaytime_sum = 0
        delay = delaytime[0] / 10
        time.sleep(delay)
        task = None
        if self.task_allocate_counter < self.task_number:
            task = self.tasks[self.task_allocate_counter]
            self.task_allocate_counter += 1
            # print("task allocate counter:", self.task_allocate_counter)
            #self.tasks.remove(task)
        else:
            self.task_allocate_counter = 0
        return task, detection

        reward = 0
        total_power = 0
        job_latency = 0
        current_job_waiting_time = 0
    
        for i in range(len(self.machines)):
            total_power += self.machines[i].idlePower + self.machines[i].activePower * self.machines[i].utilization
       
        for i in range(len(self.machines)):
            if self.current_task in self.machines[i].running_task :
                for j in range(len(self.machines[i].running_task)):
                    if self.machines[i].running_task[j] is not None:
                        time_remaining = (self.machines[i].running_task[j].executeTime - (self.machines[i].time_horizon - self.machines[i].running_task[j].startTime))
                    else:
                        time_remaining = 0
                    current_job_waiting_time += time_remaining
            if self.current_task in self.machines[i].waiting_task :
                for j in range(len(self.machines[i].running_task)):
                    if self.machines[i].running_task[j] is not None:
                        time_remaining = (self.machines[i].running_task[j].executeTime - (self.machines[i].time_horizon - self.machines[i].running_task[j].startTime))
                    else:
                        time_remaining = 0
                    current_job_waiting_time += time_remaining
                for k in range(len(self.machines[i].waiting_task)):
                    if self.machines[i].waiting_task[k] is not None:
                        time_remaining = self.machines[i].waiting_task[k].executeTime
                    else:
                        time_remaining = 0
                    current_job_waiting_time += time_remaining

        #reward = -self.theta * total_power - (1 - self.theta) * current_job_waiting_time
        reward += -current_job_waiting_time
        return reward

    def reset(self):
        self.task_allocate_counter = 0
        self.tasks = self.task_reset
        self.machines = self.machines_reset
        for i in range(len(self.machines)):
            self.machines[i].running_task = []
            self.machines[i].waiting_task = []
            self.machines[i].utilization = 0.0
            self.machines[i].availableCPU = self.machines[i].CPUCapacity

    def plot_power(self):
        plt.plot(np.arange(len(self.total_power)), self.total_power)
        plt.title('Power')
        plt.ylabel('Power')
        plt.xlabel('number of tasks')
        plt.show()

    def plot_reward(self):
        plt.plot(np.arange(len(self.reward)), self.reward)
        plt.title('reward')
        plt.ylabel('Reward')
        plt.xlabel('number of tasks')
        plt.show()

    def plot_job_latency(self):
        plt.plot(np.arange(len(self.total_job_latency)), self.total_job_latency)
        plt.title('job_latency')
        plt.ylabel('job_latency')
        plt.xlabel('number of tasks')
        plt.show()
