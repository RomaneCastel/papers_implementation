# test this out in the penulum environment
from ddpg_tf import Agent
import gym
import numpy as np
from utils import plotLearning

if __name__ == "__main__":
    env = gym.make('Pendulum-v0')
    agent = Agent(alpha=0.0001, beta=0.001, input_dims=[3], tau=0.001, env=env, batch_size=64, layer1_size=400, layer2_size=300, n_actions=1)
    # keep track of the scores
    score_history = []
    # for replicability
    np.random.seed(0)
    # play 1000 games
    for i in range(1000):
        obs = env.reset()
        done = False
        score = 0
        while not done:
            act = agent.choose_action(obs)
            new_state, reward, done, info = env.step(act)
            agent.remember(obs, act, reward, new_state, done)
            agent.learn()
            score += reward
            obs = new_state
        # at the end of each episod, append that score in the score_history
        score_history.append(score)
        print('episod {} score {} 100 games average'.format(i, score, np.mean(score_history[-100:])))
        
    filename = 'pemdulum.png'
    plotLearning(score_history, filename, window=100)
