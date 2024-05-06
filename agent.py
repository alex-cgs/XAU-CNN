import numpy as np

def sigmoid(matrix):
    return 1 / (1 + np.exp(-matrix))
            
def softmax(matrix):
    exp_matrix = np.exp(matrix)
    sum_exp = np.sum(exp_matrix, axis=1, keepdims=True)
    softmax_matrix = exp_matrix / sum_exp
    
    return softmax_matrix

def relu(matrix):
    return np.maximum(0, matrix)

class Agent:
    
    def __init__(self):
        self.balance = 10000
        self.w1 = np.random.uniform(-1, 1, size=(200, 20))
        self.w2 = np.random.uniform(-1, 1, size=(20, 1))
        self.b1 = np.random.uniform(-1, 1, size=(20, 20))
        self.b2 = np.random.uniform(-1, 1, size=(20, 1))
    
    def propagate(self, inp):
        A1 = sigmoid(inp.dot(self.w1) + self.b1)
        A2 = relu(A1.dot(self.w2) + self.b2)
        return A2
    
    def long(self, unit):
        us = self.balance / unit[0]
        self.balance -= us * unit[0]
        self.balance += us * unit[1]
                
    def short(self, unit):
        us = self.balance / unit[0]
        self.balance += us * unit[0]
        self.balance -= us * unit[1]

def merge(agent1, agent2):

    merged_agent = Agent()

    merged_agent.w1 = (agent1.w1 + agent2.w1) / 2
    merged_agent.w2 = (agent1.w2 + agent2.w2) / 2

    merged_agent.b1 = (agent1.b1 + agent2.b1) / 2
    merged_agent.b2 = (agent1.b2 + agent2.b2) / 2
    
    return merged_agent