import numpy as np
from datetime import datetime

now1 = datetime.now()
formatter = "%Y%m%d-%H%M%S"
nowstrf = now1.strftime(formatter)

#10x10-es rács definiálása
grid_size = 10
start = (0,0)
goal = (9,9)
grid = np.zeros((grid_size, grid_size))

#meghatározott útvonal: például jobb-jobb-le-le-le-jobb-jobb-fel-fel-jobb-jobb
path = [(0,0), (0,1), (0,2), (1,2), (2,2), (3,2), (3,3), (3,4), (2,4), (1,4), (1,5), (1,6),
        (2,6), (3,6), (4,6), (5,6), (6,6), (7,6), (8,6), (9,6), (9,7,), (9,8), (9,9)]

#Funkció a szomszédos lépések megtalálására
def get_neighbors(position):
    neighbors = []
    x, y = position
    if x > 0:
        neighbors.append((x-1, y))
    if x < grid_size - 1:
        neighbors.append((x+1, y))
    if y > 0:
        neighbors.append((x, y-1))
    if y < grid_size - 1:
        neighbors.append((x, y+1))
    return neighbors

#Q-learning paraméter
alpha = 0.1 #tanulási ráta
gamma = 0.9 #diszkont faktor
epsilon = 0.1 #felfedezés/exploitáció arány

#Q-tábla inicializálása
Q = np.zeros((grid_size, grid_size, 4))

#Lépések definiálása
actions = ['left', 'right', 'up', 'down']
action_to_index = {action: idx for idx, action in enumerate(actions)}

#Visszaadja az új pozíciót egy adott akció után
def move(position, action):
    x, y = position
    if action == 'left':
        y -= 1
    elif action == 'right':
        y += 1
    elif action == 'up':
        x -= 1
    elif action == 'down':
        x += 1
    return x, y

#cselekvés választása epsilon-greedy stratégia szerint
def choose_action(state):
    if np.random.rand() < epsilon:
        return np.random.choice(actions)
    else:
        x, y = state
        return actions[np.argmax(Q[x, y])]

#Q-learning algoritmus futtatása
def q_learning():
    for episode in range(1000): #epizódok száma
        state = start
        total_reward = 0
        step_counter = 0
        states = []
        while state != goal:
            action = choose_action(state)
            new_state = move(state, action)
            if new_state not in path:
                new_state = start #visszatérés a start mezőre
                reward = -100 #büntetés a helytelen lépésért
            else:
                reward = -1 #kis büntetés minden lépésért
            total_reward += reward
            x, y = state
            nx, ny = new_state
            Q[x, y, action_to_index[action]] = (1 - alpha) * Q[x, y, action_to_index[action]] + alpha * (reward + gamma * np.max(Q[nx, ny]))
            state = new_state
            states.append(new_state)
            print(state)
            #with open(f'episode_log{episode}.txt', "a") as log:
            #    log.write(f'{new_state}\n')
            step_counter += 1

                
        print(f"Episode {episode} completed in {step_counter} steps with total reward: {total_reward}")
        if episode == 0 or episode % 9 == 0:
            with open(f'{nowstrf}-episodes_and_steps_log.txt', "a") as log:
                rownumber = 0
                for row in states:
                    rownumber += 1
                    log.write(f"Step{rownumber}/{step_counter} - {row}, Ep{episode}\n")
                log.write(f'Episode {episode} completed in {step_counter} steps with total reward: {total_reward}\n')
        
        with open(f'{nowstrf}-episode_rewards_log.txt', "a") as alog: 
            alog.write(f'Episode {episode} completed in {step_counter} steps with total reward: {total_reward}\n')

q_learning()