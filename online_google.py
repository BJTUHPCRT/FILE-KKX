"""
    this part is the beginning
"""

from networks.DQN_enum import DeepQNetwork
from comparsionAlgo.random import randomAgent
from entities import enviroment_google
from entities import fluctions
import matplotlib.pyplot as plt
import pandas

from model import en
from model import en1
from model import en2
from model import en3
from model import en4
from model import en5
from model import en6
from model import en7
from model import en8
from model import en9


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
    task_tmp_list = []
    task_number = []
    random_counter = 0
    gamma_tmp = 0.9
    gamma_changed = 0
    still_enum = True
    fluction_gamma_list = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,]
    # fluction_gamma_list = [0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9]

    for episode in range(1):
        # initial observation
        global observation
        environment.reset()

        environment_list = []
        is_update = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,]
        observation = environment.observe(None)

        for w in range(environment.task_number):

            # while True:
            done, workload_task_list, workload_change = environment.create_taskWave() 
            print(environment.change_task_number)
            for i in range (len(workload_task_list)):
                task_number.append(len(workload_task_list[i]))
                task_tmp_list = workload_task_list[i]
                tmp = len(task_tmp_list)
                end = task_tmp_list[tmp - 1]
                if end == 0:
                    observation, step = DQNrl(tmp, task_tmp_list, step, environment, observation, gamma_list, gamma_tmp = 0.9)
                else:
                    observation, step = DQNrl(tmp, task_tmp_list, step, environment, observation, gamma_list, gamma_tmp=0.9)
           
            if done:
                break
              
        print(standerd_rewards(environment.reward),standerd_rewards(environment.total_power),standerd_rewards(environment.total_job_latency),standerd_rewards(environment.total_job_waitingTime))
      
        print('------------------------------loop------------------------------------------------------------------------', episode)

    TT = 0
    for i in range(len(task_number)):
        TT += task_number[i]
    print('total task number',TT)

    en_plot_job_latency(environment.total_job_latency)
    en_plot_power(environment.total_power)
    en_plot_reward(environment.reward)
    en_plot_job_waitingTime(environment.total_job_waitingTime)

    # end of game
    print('loop over')

def  DQNrl(task_temp_length,task_temp, step, environment, observation, gamma_list, gamma_tmp):
    task_count = 0
    current_step = step
    task_temp_length = task_temp_length - 1

    while (task_count < task_temp_length):
        action = DQN.choose_action(observation)

        # RL take action and get next observation and reward
        observation_, reward = environment.temp_step(action, task_temp[task_count])
        task_count += 1
        DQN.store_transition(observation, action, reward, observation_, gamma_tmp)

        if (step > 360) and (step % 5 == 0):
            DQN.learn(step) 
            gamma_list.append(gamma_tmp)
      
        observation = observation_
        step += 1
    return observation, step

def  enumeration(observation, environment, task_temp, task_temp_length, step, start,episode):

    ENaction = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0]
    ENenvironment = copy.deepcopy(environment)
    reward_list = []
    reward_pick = []
    tmp_reward = 0
    theta = 0.5
    count = 30
    iterations = 80 

    reward_averge9 = en9.en_9(observation, ENenvironment, task_temp, task_temp_length, step,iterations, count, start, episode)
    tmp_reward = theta * standerd_rewards(reward_averge9) + (1-theta) * reward_averge9[iterations - 1]
    reward_pick.append(tmp_reward)
    reward_list += reward_averge9

    reward_averge8 = en8.en_8(observation, ENenvironment, task_temp, task_temp_length, step, iterations,count, start, episode)
    tmp_reward = theta * standerd_rewards(reward_averge8) + (1 - theta) * reward_averge8[iterations - 1]
    reward_pick.append(tmp_reward)
    reward_list += reward_averge8

    reward_averge7 = en7.en_7(observation, ENenvironment, task_temp, task_temp_length, step,  iterations, count, start, episode)
    tmp_reward = theta * standerd_rewards(reward_averge7) + (1 - theta) * reward_averge7[iterations - 1]
    reward_pick.append(tmp_reward)
    reward_list += reward_averge7

    reward_averge6 = en6.en_6(observation, ENenvironment, task_temp, task_temp_length, step, iterations, count, start, episode)
    tmp_reward = theta * standerd_rewards(reward_averge6) + (1 - theta) * reward_averge6[iterations - 1]
    reward_pick.append(tmp_reward)
    reward_list += reward_averge6

    reward_averge5 = en5.en_5(observation, ENenvironment, task_temp, task_temp_length, step,  iterations, count, start, episode)
    tmp_reward = theta * standerd_rewards(reward_averge5) + (1 - theta) * reward_averge5[iterations - 1]
    reward_pick.append(tmp_reward)
    reward_list += reward_averge5

    reward_averge4 = en4.en_4(observation, ENenvironment, task_temp, task_temp_length, step, iterations, count, start, episode)
    tmp_reward = theta * standerd_rewards(reward_averge4) + (1 - theta) * reward_averge4[iterations - 1]
    reward_pick.append(tmp_reward)
    reward_list += reward_averge4

    reward_averge3 = en3.en_3(observation, ENenvironment, task_temp, task_temp_length, step,  iterations,count, start, episode)
    tmp_reward = theta * standerd_rewards(reward_averge3) + (1 - theta) * reward_averge3[iterations - 1]
    reward_pick.append(tmp_reward)
    reward_list += reward_averge3

    reward_averge2 = en2.en_2(observation, ENenvironment, task_temp, task_temp_length, step,  iterations, count, start, episode)
    tmp_reward = theta * standerd_rewards(reward_averge2) + (1 - theta) * reward_averge2[iterations - 1]
    reward_pick.append(tmp_reward)
    reward_list += reward_averge2

    reward_averge1 = en1.en_1(observation, ENenvironment, task_temp, task_temp_length, step,iterations, count, start, episode)
    tmp_reward = theta * standerd_rewards(reward_averge1) + (1 - theta) * reward_averge1[iterations - 1]
    reward_pick.append(tmp_reward)
    reward_list += reward_averge1

    reward_averge0 = en.en(observation, ENenvironment, task_temp, task_temp_length, step, iterations,count, start, episode)
    tmp_reward = theta * standerd_rewards(reward_averge0) + (1-theta) * reward_averge0[iterations - 1]
    reward_pick.append(tmp_reward)
    reward_list += reward_averge0

    plot_reward(reward_averge9, reward_averge8, reward_averge7, reward_averge6, reward_averge5, reward_averge4,reward_averge3, reward_averge2, reward_averge1, reward_averge0, min(reward_list),max(reward_list), iterations,  start, episode)
    # plot_two_reward(reward_averge9,reward_averge7,reward_averge5,reward_averge3,reward_averge1,min(reward_list),max(reward_list),iterations, start,episode)
    m = min(reward_pick)
    index = reward_pick.index(m)
    # if flag: break


    print('----------------------------------------------------------------------')
    return ENaction[index]
    # return 0.5

def plot_gamma(gamma_list):
    import matplotlib.pyplot as plt
    import pandas
    # save results into file
    dataframe = pandas.DataFrame(data=list(gamma_list), index=None, columns=None)
    dataframe.to_csv('gamma.csv', index=False, sep=',')
    plt.plot(np.arange(len(gamma_list)), gamma_list)
    plt.title('gamma')
    plt.ylabel('gamma')
    plt.xlabel('training steps')
def plot_reward(reward_averge9,reward_averge8,reward_averge7,reward_averge6,reward_averge5,reward_averge4,reward_averge3,reward_averge2,reward_averge1,reward_averge0,min_reward,max_reward,iterations, start, episode):
    import matplotlib.pyplot as plt
    from matplotlib.pyplot import MultipleLocator
    import time
    plt.clf()
    import pandas
    
    import numpy as np
    import matplotlib.pyplot as plt
    internal = iterations 
    temp_inter = 0
    last_temp_inter = 0
    start = str(start)
    episode = str(episode)

    x = np.arange(internal) 
   

    plt.title('Compare reward different gamma-' + start + '-' + episode)  
   

    plt.plot(x, reward_averge9, color='grey', label='0.9', linestyle='-', )
    plt.plot(x, reward_averge8, 'grey', label='0.8', linestyle=':', )
    plt.plot(x, reward_averge7, color='blue', label='0.7', linestyle='-')
    plt.plot(x, reward_averge6, 'blue', label='0.6', linestyle=':')
    plt.plot(x, reward_averge5, color='#FF7400', label='0.5', linestyle='-')
    plt.plot(x, reward_averge4, '#FF7400', label='0.4', linestyle=':')
    plt.plot(x, reward_averge3, color='hotpink', label='0.3', linestyle='-')
    plt.plot(x, reward_averge2, 'hotpink', label='0.2', linestyle=':')
    plt.plot(x, reward_averge1, color='forestgreen', label='0.1', linestyle='-')
    plt.plot(x, reward_averge0, 'forestgreen', label='0', linestyle=':')

    plt.legend()
    plt.xlabel('episode')
    plt.ylabel('av_reward')
  
    mi = min_reward - 10
    ma = max_reward + 10
    plt.ylim(mi,ma)

    plot_file = episode + '-' + start + '-' + 'reward.png'

    import sys
    import os
    path = sys.path[0] + '\orewards\\'
    dataframe = pandas.DataFrame(data=list(reward_averge9), index=None, columns=None)
    x = dataframe.to_csv(os.path.join(path, episode + '-' + start + '9.csv'), sep='\t', header=None, index=None)

    dataframe = pandas.DataFrame(data=list(reward_averge8), index=None, columns=None)
    x = dataframe.to_csv(os.path.join(path, episode + '-' + start + '8.csv'), sep='\t', header=None, index=None)

    dataframe = pandas.DataFrame(data=list(reward_averge7), index=None, columns=None)
    x = dataframe.to_csv(os.path.join(path, episode + '-' + start + '7.csv'), sep='\t', header=None, index=None)

    dataframe = pandas.DataFrame(data=list(reward_averge6), index=None, columns=None)
    x = dataframe.to_csv(os.path.join(path, episode + '-' + start + '6.csv'), sep='\t', header=None, index=None)

    dataframe = pandas.DataFrame(data=list(reward_averge5), index=None, columns=None)
    x = dataframe.to_csv(os.path.join(path, episode + '-' + start + '5.csv'), sep='\t', header=None, index=None)

    dataframe = pandas.DataFrame(data=list(reward_averge4), index=None, columns=None)
    x = dataframe.to_csv(os.path.join(path, episode + '-' + start + '4.csv'), sep='\t', header=None, index=None)

    dataframe = pandas.DataFrame(data=list(reward_averge3), index=None, columns=None)
    x = dataframe.to_csv(os.path.join(path, episode + '-' + start + '3.csv'), sep='\t', header=None, index=None)

    dataframe = pandas.DataFrame(data=list(reward_averge2), index=None, columns=None)
    x = dataframe.to_csv(os.path.join(path, episode + '-' + start + '2.csv'), sep='\t', header=None, index=None)

    dataframe = pandas.DataFrame(data=list(reward_averge1), index=None, columns=None)
    x = dataframe.to_csv(os.path.join(path, episode + '-' + start + '1.csv'), sep='\t', header=None, index=None)

    dataframe = pandas.DataFrame(data=list(reward_averge0), index=None, columns=None)
    x = dataframe.to_csv(os.path.join(path, episode + '-' + start + '0.csv'), sep='\t', header=None, index=None)

def plot_two_reward(reward_averge9,reward_averge7,reward_averge5,reward_averge3,reward_averge1,min_reward,max_reward,iterations, start, episode):
    import matplotlib.pyplot as plt
    from matplotlib.pyplot import MultipleLocator
    import time
    plt.clf()
    import pandas
    import numpy as np
    import matplotlib.pyplot as plt
    internal = iterations 
    temp_inter = 0
    last_temp_inter = 0

    x = np.arange(internal)

    plt.title('Compare reward with same gamma-' + str(start) + '-' + str(episode)) 
   
    plt.plot(x, reward_averge9, color='red', label='0.9', linestyle='-', )
    plt.plot(x, reward_averge7, 'blue', label='0.7', linestyle='-', )
    plt.plot(x, reward_averge5, color='#FF7400', label='0.5', linestyle='-')
    plt.plot(x, reward_averge3, color='hotpink', label='0.3', linestyle='-')
    plt.plot(x, reward_averge1, color='forestgreen', label='0.1', linestyle='-')

    plt.legend()  
    plt.xlabel('episode')
    plt.ylabel('av_reward')
    # plt.axis([0, 110, -110, 1]) 
    mi = min_reward - 10
    ma = max_reward + 10
    plt.ylim(mi,ma)
    # plt.ylim(-1100,0)

    plot_file = time.strftime('%Y-%m-%d_%H-%M-%S') +  'reward.png'

def en_plot_reward(en_reward_list):
    plt.clf()
    dataframe = pandas.DataFrame(data=list(en_reward_list), index=None, columns=None)
    dataframe.to_csv('reward.csv', index=False, sep=',')
    plt.plot(np.arange(len(en_reward_list)), en_reward_list)
    plt.title('reward_enum_total')
    plt.ylabel('Reward')
    plt.xlabel('number of tasks')


def en_plot_power(total_power):
    plt.clf()
    dataframe = pandas.DataFrame(data=list(total_power), index=None, columns=None)
    dataframe.to_csv('power.csv', index=False, sep=',')
    plt.plot(np.arange(len(total_power)), total_power)
    plt.title('Power_enum')
    plt.ylabel('Power')
    plt.xlabel('number of tasks')


def en_plot_job_latency(total_job_latency):
    plt.clf()
    dataframe = pandas.DataFrame(data=list(total_job_latency), index=None, columns=None)
    dataframe.to_csv('jobLatency.csv', index=False, sep=',')
    plt.plot(np.arange(len(total_job_latency)),total_job_latency)
    plt.title('job_latency_enum')
    plt.ylabel('job_latency')
    plt.xlabel('number of tasks')


def en_plot_job_waitingTime(total_job_waitingTime):
    plt.clf()
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
    environment = enviroment_google.environment()
    DQN = DeepQNetwork(environment.action_len,
                       environment.action_len * 2 + 1,
                      learning_rate=0.01,
                      # reward_decay=0.9,
                      e_greedy=0.8,
                      replace_target_iter=200,
                      memory_size=500,
                       batch_size=32,
                      # output_graph=True
                      )
    random_agent = randomAgent(environment.action_len)
    launcher(environment)
    DQN.plot_cost()
    DQN.plot_qtarget()