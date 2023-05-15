import gym
import numpy as np

# Frozen Lake environment setup
epsilon = 0.1
env = gym.make('MountainCar-v0')
total_episodes = 10000
max_steps = 100
alpha = 0.85
gamma = 0.95

# Initialize Q-table
num_states = env.observation_space.shape[0]
num_actions = env.action_space.n
Q = np.zeros((num_states, num_actions))

# SARSA algorithm
for episode in range(total_episodes):
    state = env.reset()
    action = env.action_space.sample()
    sum=0
    for step in range(max_steps):
        # Take an action and observe the next state and reward
        next_state, reward, done, _ = env.step(action)
        next_state=np.argmax(next_state)
        sum+=reward
        # Choose the next action using epsilon-greedy policy
        next_action = env.action_space.sample() if np.random.rand() < epsilon else np.argmax(Q[next_state])
        state=np.argmax(state)
        # Update Q-value using SARSA update rule
        Q[state, action] = Q[state, action] + alpha * (reward + gamma * Q[next_state, next_action] - Q[state, action])

        state = next_state
        action = next_action

        if done:
            break
        reward+=1
    print('episode : '+str(episode)+' reward: '+str(sum))
# Print the optimal Q-values
print("Optimal Q-values:")
print(Q)
