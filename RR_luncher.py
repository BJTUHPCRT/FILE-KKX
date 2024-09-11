"""
    this part is the beginning
"""

from comparsionAlgo.random import randomAgent
from comparsionAlgo.RR import RoundRobinAgent
from entities import environment_RR
import matplotlib.pyplot as plt

import threading
import copy
import numpy as np

def launcher(environment):
    output = open('record.log','w')
    def running_machine():
        for i in range(len(environment.machines)):
            environment.machines[i].check_if_have_task_finished()
        global timer
        timer = threading.Timer(0.1, running_machine) 
        timer.start()
    running_machine()  

    step = 0
    current_step = 0
    current_episode = 0
    IsGamma = True
    task_temp = []
    task_substitution = None
    gamma_list = []
    environment_list = []
    en_reward_list = []
    en_power_list = []
    en_latecy_list = []
    en_waitingTime_list = []
    random_counter = 1101
    gamma_tmp = 0.9
    gamma_changed = 0
    task_number = []

    for episode in range(1):
        # initial observation
        global observation
        environment.reset()
        environment_list = []

        # DQN.reset()
        #time.sleep(60)
        observation = environment.observe(None)

        for w in range(environment.task_number):
            done, workload_task_list, workload_change = environment.create_taskWave()
            #RL choose action based on observation
            print(environment.change_task_number)
            for i in range (len(workload_task_list)):
                task_number.append(len(workload_task_list[i]))
                task_tmp_list = workload_task_list[i]
                tmp = len(task_tmp_list)
                end = task_tmp_list[tmp - 1]
                if end == 0:
                    observation, step = DQNrl(tmp, task_tmp_list, step, environment, observation, gamma_list, gamma_tmp=0.9)
                else:
 
                    observation, step = DQNrl(tmp, task_tmp_list, step, environment, observation, gamma_list, gamma_tmp=0.3)
                # break while loop when end of this episode
            if done:
                break
                # print('step:', step)

        print('------------------------------loop------------------------------------------------------------------------', episode)


    TT = 0
    for i in range(len(task_number)):
        TT += task_number[i]
    print('total task number',TT)


    print(standerd_rewards(environment.total_power), standerd_rewards(environment.total_job_latency),
          standerd_rewards(environment.total_job_waitingTime))

    en_plot_job_latency(environment.total_job_latency)
    en_plot_power(environment.total_power)
    en_plot_job_waitingTime(environment.total_job_waitingTime)
    # end of game
    print('loop over')

def  DQNrl(task_temp_length,task_temp, step, environment, observation, gamma_list, gamma_tmp):
    task_count = 0
    current_step = step
    task_temp_length = task_temp_length - 1

    while (task_count < task_temp_length):
        action = RR_agent.choose_action()

 
        observation_, reward = environment.temp_step(action, task_temp[task_count])
        task_count += 1

 
        observation = observation_
        step += 1
    return observation, step

def en_plot_power(total_power):
    plt.clf()
    import pandas
    # save results into file
    dataframe = pandas.DataFrame(data=list(total_power), index=None, columns=None)
    dataframe.to_csv('power.csv', index=False, sep=',')
    plt.plot(np.arange(len(total_power)), total_power)
    plt.title('Power_enum')
    plt.ylabel('Power')
    plt.xlabel('number of tasks')

def en_plot_job_latency(total_job_latency):
    plt.clf()
    import pandas
    # save results into file
    dataframe = pandas.DataFrame(data=list(total_job_latency), index=None, columns=None)
    dataframe.to_csv('jobLatency.csv', index=False, sep=',')
    plt.plot(np.arange(len(total_job_latency)),total_job_latency)
    plt.title('job_latency_enum')
    plt.ylabel('job_latency')
    plt.xlabel('number of tasks')

def en_plot_job_waitingTime(total_job_waitingTime):
    plt.clf()
    import pandas
    # save results into file
    dataframe = pandas.DataFrame(data=list(total_job_waitingTime), index=None, columns=None)
    dataframe.to_csv('waiting_time.csv', index=False, sep=',')

    plt.plot(np.arange(len(total_job_waitingTime)),total_job_waitingTime)
    plt.title('total_job_waitingTime')
    plt.ylabel('total_job_waitingTime')
    plt.xlabel('steps')

def standerd_rewards(reward):
    average_r = 0
    for i in range(len(reward)):
        average_r += reward[i]
    average_r = average_r / len(reward)

    return average_r

if __name__ == "__main__":
    # maze game
    global environment
    environment = environment_RR.environment()
    random_agent = randomAgent(environment.action_len)
    RR_agent = RoundRobinAgent(environment.action_len)
    launcher(environment)
    #
    # environment.plot_job_waitingTime()
    # environment.plot_power()
    # environment.plot_job_latency()
