# Hello world of the reinforcement learning
# update from Lucas&Thomas Amunategui
# shaohong352@outlook.com

import numpy as np
import pylab as plt

# create a map for each direction to take. starting point 0 . goal point 7
points = [(0,1), (1,5), (5,6), (5,4), (1,2), (2,3), (2,7)]
GOAL = 7
import networkx as nx
G = nx.Graph()
G.add_edges_from(points)
pos = nx.spring_layout(G)
nx.draw_networkx_nodes(G,pos)
nx.draw_networkx_edges(G,pos)
nx.draw_networkx_labels(G,pos)
plt.show()

# create a reward matrix
# how many points in graph

MATRIX_SIZE = 8

# initialize a reward matrix to -1
reward = np.ones((MATRIX_SIZE, MATRIX_SIZE))*(-1)
# 0 for viable path, 100 for goal reaching point
for connect in points:
    if connect[1] == GOAL:
        reward[connect] = 100
    else:
        reward[connect] = 0

    if connect[0] == GOAL:
        reward[connect[::-1]] = 100
    else:
        reward[connect[::-1]] = 0
# round trip on goal
reward[GOAL,GOAL] = 100
print(reward)

#  Getting AI smarter with Q-learning: a simple first step in Python.
Q = np.zeros([MATRIX_SIZE, MATRIX_SIZE])
# leaning parameter
gamma = 0.8

initial_state = 1

def available_actions(state):
    current_state_row = reward[state,]
    av_act = np.where(current_state_row >= 0)[0]
    return av_act

available_act = available_actions(initial_state)

def sample_next_action(available_action_range):
    next_action = int(np.random.choice(available_action_range, 1))
    return next_action

action = sample_next_action(available_act)

def update(current_state, action, gamma):
    max_index = np.where(Q[action,:] == np.max(Q[action,:]))[0]

    if max_index.shape[0] > 1:
        max_index = int(np.random.choice(max_index, size=1))
    else:
        max_index = int(max_index)
    max_value = Q[action, max_index]

    Q[current_state, action] = reward[current_state, action] + gamma*max_value
    print('max_value', reward[current_state, action] + gamma*max_value)
    if np.max(Q) > 0:
        return np.sum(Q/np.max(Q)*100)
    else:
        return 0

update(initial_state, action, gamma)

# Training
scores = []
for i in range(700):
    current_state = np.random.randint(0, int(Q.shape[0]))
    available_act = available_actions(current_state)
    action = sample_next_action(available_act)
    score = update(current_state, action, gamma)
    scores.append(score)
    print('Score:', str(score))
print('Trainned Q matrix:')
print(Q/np.max(Q)*100)
plt.matshow(Q)
plt.show()

# Testing
current_state = 0
steps = [current_state]
while current_state != GOAL:
    next_step_index = np.where(Q[current_state,] == np.max(Q[current_state,]))[0]

    if next_step_index.shape[0] > 1:
        next_step_index = int(np.random.choice(next_step_index, size=1))
    else:
        next_step_index = int(next_step_index)
    steps.append(next_step_index)
    current_state = next_step_index
print('most efficient path')
print(steps)

plt.plot(scores)
plt.show()