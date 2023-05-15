import gym
import numpy as np

# Frozen Lake environment setup
epsilon = 0.9
env = gym.make('FrozenLake-v1')
total_episodes = 10000
max_steps = 100
alpha = 0.85
gamma = 0.95

# Initialize Q-table
num_states = env.observation_space.n
num_actions = env.action_space.n
Q = np.zeros((num_states, num_actions))

# SARSA algorithm
for episode in range(total_episodes):
    state = env.reset()
    action = env.action_space.sample()

    for step in range(max_steps):
        # Take an action and observe the next state and reward
        next_state, reward, done, _ = env.step(action)

        # Choose the next action using epsilon-greedy policy
        next_action = env.action_space.sample() if np.random.rand() < epsilon else np.argmax(Q[next_state])

        # Update Q-value using SARSA update rule
        Q[state, action] = Q[state, action] + alpha * (reward + gamma * Q[next_state, next_action] - Q[state, action])

        state = next_state
        action = next_action

        if done:
            break

# Print the optimal Q-values
print("Optimal Q-values:")
print(Q)
