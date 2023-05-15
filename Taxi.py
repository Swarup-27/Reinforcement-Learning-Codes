import gym
import numpy as np

# Taxi environment setup
env = gym.make('Taxi-v3')
total_episodes = 10000
max_steps = 100
alpha = 0.1
gamma = 0.6
epsilon = 0.1

# Initialize Q-table
num_states = env.observation_space.n
num_actions = env.action_space.n
Q = np.zeros((num_states, num_actions))

# Q-learning algorithm
for episode in range(total_episodes):
    state = env.reset()

    for step in range(max_steps):
        # Choose an action using epsilon-greedy policy
        if np.random.uniform(0, 1) < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(Q[state])

        # Take the chosen action and observe the next state, reward, and done flag
        next_state, reward, done, _ = env.step(action)

        # Update Q-value using the Q-learning update rule
        Q[state, action] = Q[state, action] + alpha * (reward + gamma * np.max(Q[next_state]) - Q[state, action])

        state = next_state

        if done:
            break

# Evaluate the learned policy
total_rewards = 0
num_eval_episodes = 10
for _ in range(num_eval_episodes):
    state = env.reset()

    for _ in range(max_steps):
        action = np.argmax(Q[state])
        next_state, reward, done, _ = env.step(action)
        total_rewards += reward
        state = next_state
        if done:
            break

average_reward = total_rewards / num_eval_episodes

# Print the average reward
print("Average reward:", average_reward)
