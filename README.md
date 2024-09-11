# BJTUHPCRT-Adaptive-DRL-based-Task-Scheduling-for-Energy-Efficient-Cloud-Computing
# Abstract
Intelligent task scheduling solutions are highly demanded in the operation of complex cloud data centers so that resources can be utilized in an energy-efficient way while still ensuring various requirements of users. However, the energy problem of task scheduling in cloud environment becomes more challenging with the ever-increasing number of users as well as the constant and unpredictable change of workloads.In this research, we propose an Adaptive Deep Reinforcement Learning-based (ADRL) task scheduling framework for energy-efficient cloud computing. We first present a Change Detection algorithm to detect whether the workload has changed greatly. On this basis, we built an Automatic Generation network to adjust the discount factor of Deep Reinforcement Learning (DRL) dynamically according to the changing workload, which enables faster and more accurate learning. We finally introduce the adaptive DRL to learn the optimal policy of dispatching arriving user requests with the reward aiming to minimize task response time and maximize resource utilization. Simulated experiments have confirmed that the proposed scheduling scheme performs well on accelerating learning convergence and promoting allocation accuracy, thus it is very effective in reducing the average response time of tasks and increasing the CPU utilization rate of resources, which eventually makes the cloud system more energy efficient.

# Citation
@article{kang2021adaptive,
  title={Adaptive DRL-based task scheduling for energy-efficient cloud computing},
  author={Kang, Kaixuan and Ding, Ding and Xie, Huamao and Yin, Qian and Zeng, Jing},
  journal={IEEE Transactions on Network and Service Management},
  volume={19},
  number={4},
  pages={4948--4961},
  year={2021},
  publisher={IEEE}
}
