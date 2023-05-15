import numpy as np
import matplotlib.pyplot as plt

# Define the number of arms and the number of episodes
num_arms = 10
num_episodes = 1000

# Define the epsilon values to test
epsilons = [0, 0.1, 0.01]

# Define the true reward distribution for each arm
reward_means = np.random.normal(loc=0, scale=1, size=num_arms)

# Initialize the estimated reward distribution for each arm
estimated_means = np.zeros(num_arms)

# Initialize the number of times each arm has been pulled
num_pulls = np.zeros(num_arms)

# Define the epsilon-greedy action selection function
def epsilon_greedy(epsilon):
    if np.random.uniform() < epsilon:
        # Choose a random arm
        action = np.random.choice(num_arms)
    else:
        # Choose the arm with the highest estimated mean reward
        action = np.argmax(estimated_means)
    return action

# Initialize arrays to store the rewards and average rewards for each episode
rewards = np.zeros((len(epsilons), num_episodes))
avg_rewards = np.zeros((len(epsilons), num_episodes))

# Loop over the episodes
for i in range(num_episodes):
    # Loop over the epsilon values
    for j, epsilon in enumerate(epsilons):
        # Choose an action using the epsilon-greedy method
        action = epsilon_greedy(epsilon)

        # Pull the arm and observe the reward
        reward = np.random.normal(loc=reward_means[action], scale=1)

        # Update the estimated mean reward for the chosen arm
        num_pulls[action] += 1
        estimated_means[action] += (reward - estimated_means[action]) / num_pulls[action]

        # Store the reward and average reward
        rewards[j, i] = reward
        avg_rewards[j, i] = np.mean(rewards[j, :i+1])

# Plot the average rewards for each epsilon value
for j, epsilon in enumerate(epsilons):
    plt.plot(avg_rewards[j, :], label='epsilon = ' + str(epsilon))
plt.xlabel('Episode')
plt.ylabel('Average Reward')
plt.legend()
plt.show()
