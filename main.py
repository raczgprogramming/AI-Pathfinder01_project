import numpy as np
from datetime import datetime
import AIs
import os

nameversion = "Pathfinder1_V1.1.0.0"
print(f"Name and version: {nameversion}")

now1 = datetime.now()
formatter = "%Y%m%d_%H%M%S"
nowstrf = now1.strftime(formatter)

print("Választd ki az AI-t az alábbiak közül:")
ailist = AIs.ai_list()
for ai in ailist:
    print(ai)
selected_ai = AIs.ai_set()

#10x10-es rács definiálása
grid_size = 10
start = (0,0)
goal = (9,9)
grid = np.zeros((grid_size, grid_size))

#meghatározott útvonal: például jobb-jobb-le-le-le-jobb-jobb-fel-fel-jobb-jobb
path = [(0,0), (0,1), (0,2), (1,2), (2,2), (3,2), (3,3), (3,4), (2,4), (1,4), (1,5), (1,6),
        (2,6), (3,6), (4,6), (5,6), (6,6), (7,6), (8,6), (9,6), (9,7,), (9,8), (9,9)]

#Funkció a szomszédos lépések megtalálására (Jelenleg nem használt)
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

def file_location(filename):
    script_dir = os.path.dirname(__file__)
    log_file_path = os.path.join(script_dir, filename)
    return log_file_path

#Q-learning paraméter
alpha = selected_ai.alpha #tanulási ráta
gamma = selected_ai.gamma #diszkont faktor
epsilon = selected_ai.epsilon #kezdeti felfedezés/exploitáció arány
epsilon_decay = selected_ai.epsilon_decay #epsilon csökkentési arány minden epizód után
epsilon_min = selected_ai.epsilon_min #minimum epsilon érték

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
    global epsilon
    for episode in range(1000): #epizódok száma
        state = start
        total_reward = 0
        step_counter = 0
        states = []
        while state != goal and step_counter <=1000:
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
            if step_counter == 1001:
                reward =-5000

        print(f"Episode {episode} completed in {step_counter} steps with total reward: {total_reward}")
        if episode == 0 or episode % 9 == 0:
            ep_and_steps = file_location(f'{nameversion}-{nowstrf}-episodes_and_steps_log.txt')
            with open(ep_and_steps, "a") as log:
                rownumber = 0
                log.write(f'Q-Learning paramether in EP{episode}: Alpha: {alpha}, Gamma: {gamma}, Epsilon: {epsilon}\n')
                for row in states:
                    rownumber += 1
                    log.write(f"Step{rownumber}/{step_counter} - {row}, Ep{episode}\n")
                log.write(f'Episode {episode} completed in {step_counter} steps with total reward: {total_reward}\n\n')
        
        ep_rewards = file_location(f'{nameversion}-{nowstrf}-episode_rewards_log.txt')
        with open(ep_rewards, "a") as alog: 
            alog.write(f'Episode {episode} completed in {step_counter} steps with total reward: {total_reward}\n')
            alog.write(f'Q-Learning paramether: Alpha: {alpha}, Gamma: {gamma}, Epsilon: {epsilon}\n\n')

        if epsilon > epsilon_min:
            epsilon *=epsilon_decay

q_learning()