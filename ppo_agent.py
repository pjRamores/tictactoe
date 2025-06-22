import os

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

class PPOAgent:
    def __init__(self, load_models=False, policy_path=None, value_path=None):
        self.policy_model = self.build_policy_model()
        self.value_model = self.build_value_model()
        self.gamma = 0.99
        self.lambda_ = 0.95
        self.clip_epsilon = 0.2
        self.policy_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0003)
        self.value_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0003)
        if load_models and policy_path and value_path:
            self.load_models(policy_path, value_path)

    def build_policy_model(self):
        model = models.Sequential([
            layers.Dense(9, activation='relu', input_shape=(9,)),
            layers.Dense(9, activation='relu'),
            layers.Dense(9, activation='softmax')  # 9 actions
        ])
        return model

    def build_value_model(self):
        model = models.Sequential([
            layers.Dense(9, activation='relu', input_shape=(9,)),
            layers.Dense(9, activation='relu'),
            layers.Dense(1)  # State value
        ])
        return model

    def save_models(self, policy_path, value_path):
        # Save policy and value models to files
        self.policy_model.save(policy_path)
        self.value_model.save(value_path)
        print(f"Models saved to {policy_path} and {value_path}")

    def load_models(self, policy_path, value_path):
        # Load policy and value models from files
        if os.path.exists(policy_path) and os.path.exists(value_path):
            self.policy_model = tf.keras.models.load_model(policy_path)
            self.value_model = tf.keras.models.load_model(value_path)
            print(f"Models loaded from {policy_path} and {value_path}")
        else:
            raise FileNotFoundError("Model files not found")

    def get_action(self, state, deterministic=False):
        state = np.array(state).reshape(1, 9)
        probs = self.policy_model(state).numpy()[0]
        valid_moves = [i for i in range(9) if state[0, i] == 0]

        if not valid_moves:
            # No valid moves (shouldn't happen in Tic-Tac-Toe mid-game)
            return None, 0.0

        # Clip probabilities to avoid numerical instability
        probs = np.clip(probs, 1e-10, 1.0)
        valid_probs = probs[valid_moves]
        prob_sum = np.sum(valid_probs)

        if prob_sum <= 0 or np.isnan(prob_sum):
            # Fallback: uniform distribution over valid moves
            valid_probs = np.ones(len(valid_moves)) / len(valid_moves)
        else:
            valid_probs = valid_probs / prob_sum
            # Ensure probabilities sum to 1 to avoid numerical errors
            valid_probs = valid_probs / np.sum(valid_probs)

        # Verify probabilities sum to 1 within tolerance
        if not np.isclose(np.sum(valid_probs), 1.0, rtol=1e-5, atol=1e-8):
            print(f"Warning: Probabilities sum to {np.sum(valid_probs)}, adjusting to uniform")
            valid_probs = np.ones(len(valid_moves)) / len(valid_moves)

        if deterministic:
            action = valid_moves[np.argmax(valid_probs)]  # Best action
        else:
            # action = np.random.choice(valid_moves, p=valid_probs)  # Sample action
            action = self.get_index_with_max_value(probs, valid_moves)

        log_prob = np.log(max(probs[action], 1e-10))  # Ensure log_prob is valid
        return action, log_prob

    def compute_gae(self, rewards, values, next_value, dones):
        advantages = []
        gae = 0
        returns = []
        values = values + [next_value]
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + self.gamma * values[t + 1] * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * self.lambda_ * (1 - dones[t]) * gae
            advantages.insert(0, gae)
            returns.insert(0, gae + values[t])
        advantages = np.array(advantages)
        advantages = (advantages - np.mean(advantages)) / (np.std(advantages) + 1e-8)
        return advantages, returns

    def train_step(self, states, actions, old_log_probs, advantages, returns):
        states = np.array(states)
        actions = np.array(actions)
        old_log_probs = np.array(old_log_probs)
        advantages = np.array(advantages)
        returns = np.array(returns)

        # Policy update
        with tf.GradientTape() as tape:
            probs = self.policy_model(states)
            log_probs = tf.math.log(tf.reduce_sum(probs * tf.one_hot(actions, 9), axis=1) + 1e-10)
            ratio = tf.exp(log_probs - old_log_probs)
            surr1 = ratio * advantages
            surr2 = tf.clip_by_value(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages
            policy_loss = -tf.reduce_mean(tf.minimum(surr1, surr2))
            entropy = -tf.reduce_mean(tf.reduce_sum(probs * tf.math.log(probs + 1e-10), axis=1))
            policy_loss -= 0.01 * entropy

        policy_grads = tape.gradient(policy_loss, self.policy_model.trainable_variables)
        self.policy_optimizer.apply_gradients(zip(policy_grads, self.policy_model.trainable_variables))

        # Value update
        with tf.GradientTape() as tape:
            values = self.value_model(states)[:, 0]
            value_loss = tf.reduce_mean(tf.square(values - returns))

        value_grads = tape.gradient(value_loss, self.value_model.trainable_variables)
        self.value_optimizer.apply_gradients(zip(value_grads, self.value_model.trainable_variables))

    @staticmethod
    def get_index_with_max_value(arr, whitelist):
        if not whitelist or not arr.size:  # Check for empty inputs
            return None
        valid_indices = np.array(whitelist)[np.array(whitelist) < len(arr)]
        if not valid_indices.size:
            return None
        return valid_indices[np.argmax(arr[valid_indices])]
