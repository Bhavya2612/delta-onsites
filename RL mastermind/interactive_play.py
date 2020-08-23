import pickle
from environment import env
from agent import Agent
with open('learned_q_agent.pkl', 'rb') as X:
    q_agent = pickle.load(X)

def interactive_play(agent):
    input_code = input('Please enter a input code code any pattern between 0000 - 5555')
    agent.reset_possible_states()
    guess = agent.get_best_action()
    envi = env(input_code)
    print(f"initial guess = {guess}")
    u = input('Press enter to let q-learning agent make the next guess')
    while guess!= input_code:
        feedback = env.score(input_code, guess)
        agent.restrict_possible_states(guess, feedback)
        guess = agent.get_best_action()
        print(f"Next guess = {guess}")
        u = input()
    if guess == input_code:
        print("mastermind level maxx, guess is right!")

interactive_play(q_agent)