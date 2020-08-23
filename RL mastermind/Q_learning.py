import random
from environment import env
from agent import Agent
import pickle

def train(agent, n_episodes):
    for _ in range(n_episodes):
        input_code = env._number_from_index(random.randint(0, 6**4 - 1))
        envi = env(input_code)
        agent.reset_possible_states()
        action = agent.random_action()  # init action
        
        if action == input_code: # if init guess is correct skip this episode
            continue
            
        run = True
        while run:
            feedback = env.get_feedback(action)
            reward   = env.reward(action)
            agent.learn_from_move(action, feedback, reward)
            if action == input_code:
                break  # correct guess stop episode
            else:
                action = agent.random_action()  # else next guess

q_agent = Agent()
train(q_agent, 2000)
with open('learned_q_agent.pkl','wb') as f:
    pickle.dump(q_agent, f)