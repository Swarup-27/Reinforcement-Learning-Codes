# Using Deep Q-Learning
import gym
import tensorflow as tf
from tensorflow.keras import layers

# Atari Breakout environment setup
env = gym.make('Breakout-v4')

# Deep Q-Learning model
class DQN(tf.keras.Model):
    def __init__(self, num_actions):
        super(DQN, self).__init__()
        self.conv1 = layers.Conv2D(32, (8, 8), strides=(4, 4), activation='relu')
        self.conv2 = layers.Conv2D(64, (4, 4), strides=(2, 2), activation='relu')
        self.conv3 = layers.Conv2D(64, (3, 3), strides=(1, 1), activation='relu')
        self.flatten = layers.Flatten()
        self.fc1 = layers.Dense(512, activation='relu')
        self.fc2 = layers.Dense(num_actions)

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.flatten(x)
        x = self.fc1(x)
        q_values = self.fc2(x)
        return q_values

# Create the Deep Q-Network
num_actions = env.action_space.n
model = DQN(num_actions)

# Loss function and optimizer
loss_fn = tf.keras.losses.Huber()
optimizer = tf.keras.optimizers.Adam(learning_rate=0.00025, clipnorm=1.0)

# Replay buffer
replay_buffer = []

# Training parameters
num_episodes = 10000
epsilon = 1.0
epsilon_decay = 0.995
min_epsilon = 0.01
batch_size = 32
target_update_freq = 1000
update_freq = 4

# Main training loop
for episode in range(num_episodes):
    state = env.reset()
    episode_reward = 0

    while True:
        # Choose an action using epsilon-greedy policy
        if np.random.rand() < epsilon:
            action = env.action_space.sample()
        else:
            q_values = model(tf.expand_dims(state, 0))
            action = np.argmax(q_values[0])

        # Take the chosen action and observe the next state, reward, and done flag
        next_state, reward, done, _ = env.step(action)
        episode_reward += reward

        # Store the transition in the replay buffer
        replay_buffer.append((state, action, reward, next_state, done))

        state = next_state

        # Perform training if enough samples are available in the replay buffer
        if len(replay_buffer) >= batch_size and episode % update_freq == 0:
            # Sample a minibatch from the replay buffer
            minibatch = random.sample(replay_buffer, batch_size)

            # Prepare the input and target tensors
            states = np.array([transition[0] for transition in minibatch])
            actions = np.array([transition[1] for transition in minibatch])
            rewards = np.array([transition[2] for transition in minibatch])
            next_states = np.array([transition[3] for transition in minibatch])
            dones = np.array([transition[4] for transition in minibatch])

            # Compute the target Q-values
            q_values = model(states)
            next_q_values = model(next_states)
            target_q_values = q_values.numpy()
            max_next_q_values = np.amax(next_q_values.numpy(), axis=1)
            target_q_values[np.arange(batch_size), actions] = rewards + (1 - dones) * gamma * max_next_q_values
                    # Train the model on the minibatch
        with tf.GradientTape() as tape:
            q_values = model(states)
            loss = loss_fn(target_q_values, q_values)
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

    if done:
        break

# Update the epsilon value
epsilon = max(epsilon * epsilon_decay, min_epsilon)

# Update the target network weights
if episode % target_update_freq == 0:
    model_target.set_weights(model.get_weights())

# Print the episode statistics
print('Episode:', episode, 'Reward:', episode_reward)

