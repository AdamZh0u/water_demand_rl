import numpy as np
import random
import wandb

class AgentGreedy:
    def __init__(self):
        self.record = np.zeros((2, 2)) # n_obs =2  mean = 2   
    
    def update_record(self,action,r):
        record = self.record
        new_r=(record[action,0] * record[action,1]+r) / (record[action,0]+1)
        record[action,0]+=1
        record[action,1]=new_r
        self.record=record
    
    def get_best_action(self):
        return np.argmax(self.record[:,1])
    
    def dump_record(self,data_path = 'data/train/eps_greedy_record.npy'):
        np.save(data_path,self.record)
        print('record dumped to',data_path)
    
    def load_record(self, data_path = 'data/train/eps_greedy_record.npy'):
        self.record = np.load(data_path)
        print('record loaded from',data_path)
    
    def update(self, eps=0.1):
        # 与观测无关，不需要输入观测
        # 贪心的选择最佳动作或者随机选择动作 
        if random.random() > eps: 
            choice=self.get_best_action()
        else:
            choice=random.choice([0, 1])

        return choice
