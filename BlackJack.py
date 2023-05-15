# Using MonteCarlo approach
import gym
import numpy as np

# Blackjack environment setup
env = gym.make('Blackjack-v1')

# Action mapping
actions_mapping = {
    0: 'Stick',
    1: 'Hit'
}

# Initialize Q-value and C-value arrays
num_states = 31  # player's hand sum
num_actions = 2  # 0: Stick, 1: Hit
Q = np.zeros((num_states, num_actions))
C = np.zeros((num_states, num_actions))

# Parameters
total_episodes = 100000
epsilon = 0.1
gamma = 1.0

# Generate episode using the given policy
def generate_episode(policy):
    episode = []
    state = env.reset()

    done = False
    while not done:
        action = np.random.choice(np.arange(num_actions), p=policy[state[0]])
        next_state, reward, done, _ = env.step(action)
        episode.append((state, action, reward))
        state = next_state

    return episode

# Monte Carlo Policy Control with Importance Sampling
for episode in range(total_episodes):
    # Initialize weights
    W = 1.0

    # Generate episode using current policy
    policy = np.ones((num_states, num_actions)) * (epsilon / num_actions)
    policy[np.arange(num_states), np.argmax(Q, axis=1)] = 1 - epsilon + (epsilon / num_actions)
    episode = generate_episode(policy)

    # Update Q-values and C-values
    G = 0  # Return
    for t in range(len(episode) - 1, -1, -1):
        state, action, reward = episode[t]
        G = gamma * G + reward

        # Update C-value
        C[state, action] += W

        # Update Q-value
        Q[state, action] += (W / C[state, action]) * (G - Q[state, action])

        # Update policy
        best_action = np.argmax(Q[state[0]])
        for a in range(num_actions):
            if a == best_action:
                policy[state, a] = 1 - epsilon + (epsilon / num_actions)
            else:
                policy[state, a] = epsilon / num_actions

        if action != best_action:
            break

        W = W / policy[state, action]

# Print the optimal policy
optimal_policy = np.argmax(Q, axis=1)
print("Optimal policy:")
for state in range(num_states):
    print(f"Player sum: {state}, Optimal action: {actions_mapping[optimal_policy[state]]}")
