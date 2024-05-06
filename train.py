import json
import random as rd
from agent import *
from sklearn.preprocessing import minmax_scale

def take_to_date_random(nb):
    f = open('data.json')
    data = json.load(f)
    length = len(data['quotes'])
    ind = rd.randint(0, length - nb - 20)
    
    prices_read = [(data['quotes'][i]['open'], data['quotes'][i]['close']) for i in range(ind, ind + nb)]
    prices_act = [(data['quotes'][i]['open'], data['quotes'][i]['close']) for i in range(ind + nb, ind + nb + 20)]
    return (prices_read, prices_act)

def take_idx(idx, nb):
    f = open('data.json')
    data = json.load(f)
    prices_read = [(data['quotes'][i]['open'], data['quotes'][i]['close']) for i in range(idx, idx + nb)]
    prices_act = [(data['quotes'][i]['open'], data['quotes'][i]['close']) for i in range(idx + nb, idx + nb + 20)]
    return (prices_read, prices_act)

def serialize(inp):
    newarr = []
    for elt in inp:
        newarr.append(elt[0])
        newarr.append(elt[1])
    return newarr

# print(take_to_date_random(100))

def write_best(agent):
    w1 = agent.w1
    w2 = agent.w2
    b1 = agent.b1
    b2 = agent.b2
    agent_data = {'w1': w1.tolist(), 'w2': w2.tolist(), 'b1': b1.tolist(), 'b2': b2.tolist()}
    
    #Make sure that no presaved profile was stored here!
    with open('best.json', 'w') as f: 
        json.dump(agent_data, f)

def train():
    epoch = 0
    
    env = [Agent() for _ in range(20)]    
    
    while epoch <= 1000000:
        print(f'Epoch: {epoch}')
        for j in range(200):
            start = rd.randint(0, 1800)
            situation = take_idx(start, 100)
            ser_train = np.array(minmax_scale((serialize(situation[0]))))
            
            # print(env[0].propagate(ser_train))
            
            for agent in env:
                actions = np.round(agent.propagate(ser_train))
                for i in range(len(actions)):
                    
                    if actions[i][0] == 0:
                        agent.short(situation[1][i])
                    else:
                        agent.long(situation[1][i])
                    
        env.sort(key=lambda agent: agent.balance, reverse=True)
        
        print([agent.balance for agent in env])
        
        best = env[0]
        
        for i in range(10):
            env[i] = merge(best, env[i])
        for i in range(10, 20):
            env[i] = Agent()

        epoch += 1
        if epoch % 10 == 0:
            write_best(env[0])

# train()