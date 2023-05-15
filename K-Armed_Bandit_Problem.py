import numpy as np
import matplotlib.pyplot as plt

def epsilon_greedy(epsilon, q_values):
    if np.random.random() < epsilon:
        # Exploration: Choose a random action
        action = np.random.randint(len(q_values))
        print('action in epsilon_greedy (exploration): ',action)
    else:
        # Exploitation: Choose the action with the highest Q-value
        action = np.argmax(q_values)
        print('action in epsilon_greedy (exploitation): ',action)
    return action

def run_bandit(epsilon):
    num_arms = 10
    num_episodes = 1000
    rewards = np.zeros(num_episodes)
    q_values = np.zeros(num_arms)
    action_counts = np.zeros(num_arms)

    for episode in range(num_episodes):
        action = epsilon_greedy(epsilon, q_values)
        reward = np.random.normal(0, 1)  # Reward from a normal distribution with mean 0
        rewards[episode] = reward

        # Update action value estimates
        action_counts[action] += 1
        q_values[action] += (reward - q_values[action]) / action_counts[action]
        '''print("episode = ",episode)
        print('action: ',action)
        print('reward: ',reward)
        print('rewards: ',rewards)
        print('action_counts: ',action_counts)
        print('q_values: ',q_values)
        print('----------------------------------------------------------------------------------------------')'''

    return rewards

# Run bandit problem for different epsilon values
epsilon_values = [0, 0.1, 0.01]
results = []

for epsilon in epsilon_values:
    rewards = run_bandit(epsilon)
    results.append(rewards)

# Plotting
plt.figure(figsize=(10, 6))
for i, epsilon in enumerate(epsilon_values):
    plt.plot(range(len(results[i])), results[i], label=f'epsilon={epsilon}')

plt.xlabel('Episode')
plt.ylabel('Average Reward')
plt.title('Average Reward Comparison')
plt.legend()
plt.show()
