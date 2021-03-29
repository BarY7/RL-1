import gym
import random
import numpy as np 
import matplotlib.pyplot as plt

def calculate_episode(env,w):
    observation = env.reset()
    total_reward = 0
    for t in range(200):
      # env.render()
      #print(observation)
      inner_prod = np.inner(w, observation)
      action = 1 if inner_prod>=0 else 0
      observation, reward, done, info = env.step(action)
      total_reward = total_reward + reward
      if done:
          print("Episode finished after {} timesteps, Total score: {}".format(t+1,total_reward))
          return total_reward
    

def random_search(env,times):
  env = gym.make('CartPole-v0')
  best_reward = 0
  best_weights = [0 for i in range(4)]
  tries = 0
  for i_episode in range(times):
      w = [random.uniform(-1,1) for x in range(4)]
      cur_reward = calculate_episode(env,w)
      if(cur_reward > best_reward):
        best_reward = cur_reward
        best_weights = w
        if(cur_reward == 200):
          tries = i_episode
          break
  return (tries,best_weights)

    
env = gym.make('CartPole-v0')
total_tries = 1000
agents_array = np.array([random_search(env,10000)[0] for i in range(total_tries)])
# times_to_max_array = np.array([0 for i in range(total_tries)])
avg = np.average(agents_array)
print("average number of episodes: {}".format(avg))
plt.xlabel("Episodes")
plt.ylabel("Num. Predictors")
plt.hist(agents_array)
plt.show()

  
env.close()
