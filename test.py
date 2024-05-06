import json
from agent import Agent
import numpy as np
from sklearn.preprocessing import minmax_scale
from train import take_idx, serialize

def form(time):
    days = time // 1440
    hours = (time - (days * 1440)) // 60
    mins = (time - (days * 1440)) - (hours * 60)
    
    return f"{days} day, {hours} hours, {mins} minutes"

def best_agent():
    f = open('best_relu.json') # Change it with whatever profile you like!
    data = json.load(f)
    
    agent = Agent()
    
    agent.w1 = np.array(data['w1'])
    agent.w2 = np.array(data['w2'])
    agent.b1 = np.array(data['b1'])
    agent.b2 = np.array(data['b2'])
    
    return agent

def test():
    best = best_agent()
    
    f = open('data.json')
    data = json.load(f)
    length = len(data['quotes'])
    
    for i in range(length - 120):
        
        price = take_idx(i, 100)
        ser_price = np.array(minmax_scale((serialize(price[0]))))
        
        actions = np.round(best.propagate(ser_price))
        
        for j in range(len(actions)):
            if actions[j][0] == 0:
                best.short(price[1][j])
            else:
                best.long(price[1][j])
        print(f"Current balance: ${round(best.balance, 2)} ({form(i)})")

test()