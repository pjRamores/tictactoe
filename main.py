import pandas as pd
import numpy as np
np.set_printoptions(legacy="1.25")

from bot_player import BotPlayer
from ppo_agent import PPOAgent
from tictactoe import TicTacToe


def train_ppo():
    ask_human_input = False
    track_training = False
    num_episodes = 1000

    global opponent_action
    policy_path = "tic_tac_toe_policy_model.h5"
    value_path = "tic_tac_toe_value_model.h5"
    env = TicTacToe()
    # agent = PPOAgent()
    agent = PPOAgent(load_models=True, policy_path=policy_path, value_path=value_path) if input("Press y to load the model: ") == 'y' else PPOAgent()
    bot_player = BotPlayer()
    max_steps = 9
    batch_size = 64
    epochs = 10
    lost_count = 0
    training_tracker = pd.read_csv('training_tracker.csv') if track_training and input("Press y to load the training tracker: ") == 'y' else pd.DataFrame([])
    action_tracker = {}
    for _, row in training_tracker.iterrows():
        if row["action"] in action_tracker:
            action_tracker[row["action"]] += 1
        else:
            action_tracker[row["action"]] = 1



    for episode in range(training_tracker.shape[0], num_episodes + training_tracker.shape[0]):
        state = env.reset()
        trajectory = []
        episode_reward = 0

        # Collect trajectory against random opponent
        for _ in range(max_steps):
            action, log_prob = agent.get_action(state)
            if action is None:
                break
            next_state, reward, done, _ = env.step(action)
            if not done:
                # opponent_action = random.choice(env.get_valid_moves())
                # opponent_action = get_human_action(env) if ask_human_input else random.choice(env.get_valid_moves())
                best_move = bot_player.get_best_move(env.board, 1, -1)
                opponent_action = best_move[0]*3+best_move[1]
                next_state, reward, done, _ = env.step(opponent_action, player=-1)
            value = agent.value_model(np.array(state).reshape(1, 9)).numpy()[0, 0]
            trajectory.append((state, action, opponent_action, reward, log_prob, done, value))
            state = next_state
            episode_reward += reward
            if done:
                break

        # Compute advantages and returns
        states, actions, opponent_actions, rewards, log_probs, dones, values = zip(*trajectory)
        next_value = agent.value_model(np.array(state).reshape(1, 9)).numpy()[0, 0] if not done else 0
        advantages, returns = agent.compute_gae(rewards, list(values), next_value, dones)

        # Update policy and value networks
        for _ in range(epochs):
            indices = np.random.permutation(len(trajectory))
            for start in range(0, len(trajectory), batch_size):
                batch_indices = indices[start:start + batch_size]
                batch_states = [states[i] for i in batch_indices]
                batch_actions = [actions[i] for i in batch_indices]
                batch_log_probs = [log_probs[i] for i in batch_indices]
                batch_advantages = [advantages[i] for i in batch_indices]
                batch_returns = [returns[i] for i in batch_indices]
                # print(f"batch_states: {batch_states}, batch_actions: {batch_actions}, batch_log_probs: {batch_log_probs}, batch_advantages: {batch_advantages}, batch_returns: {batch_returns}")
                agent.train_step(batch_states, batch_actions, batch_log_probs, batch_advantages, batch_returns)

        if track_training:
            if actions in action_tracker:
                action_tracker[actions] += 1
            else:
                action_tracker[actions] = 1
            print(f"Episode: {episode}, Reward: {episode_reward}, State: {state}, Action: {actions} = {action_tracker[actions]}")
            training_record = {
                "episode": episode,
                "reward": episode_reward,
                "state": state,
                "actions": actions,
                "action_count": action_tracker[actions],
                "opponent_actions": opponent_actions,
                "log_probs": log_probs,
                "values": values
            }
            new_training = pd.DataFrame([training_record])
            training_tracker = pd.concat([training_tracker, new_training], ignore_index=True)

            if not continue_to_next_round():
                break
        else:
            if episode % 100 == 0:
                print(f"Episode {episode}, Reward: {episode_reward}")
            if episode_reward < 0:
                lost_count += 1
                print(f"Episode: {episode}, Lost: {lost_count}, Reward: {episode_reward}, State: {state}, Action: {actions}")

    # Save models after training
    if input("Press y to save the model: ") == 'y':
        agent.save_models(policy_path, value_path)

    if track_training and input("Press y to save the training tracker: ") == 'y':
        training_tracker.to_csv('training_tracker.csv', index=False)

    return policy_path, value_path

# Interactive Play Function
def play_game(policy_path, value_path):
    env = TicTacToe()
    agent = PPOAgent(load_models=True, policy_path=policy_path, value_path=value_path)
    state = env.reset()

    print("Welcome to Tic-Tac-Toe! You are 'O', the agent is 'X'.")
    print("Enter a position (0-8) to make your move:")
    print(" 0 | 1 | 2 ")
    print("---+---+---")
    print(" 3 | 4 | 5 ")
    print("---+---+---")
    print(" 6 | 7 | 8 ")

    while not env.done:
        # Agent's turn (X)
        env.render()
        action, _ = agent.get_action(state, deterministic=True)  # Use best action
        if action is None:
            print("No valid moves available for agent.")
            break
        state, reward, done, info = env.step(action)
        if info:
            print(info)
            break
        if done:
            env.render()
            if env.winner == 1:
                print("Agent (X) wins!")
            elif env.winner == -1:
                print("You (O) win!")
            else:
                print("It's a draw!")
            break

        # Human's turn (O)
        env.render()
        while True:
            try:
                human_action = int(input("Your move (0-8): "))
                if human_action in env.get_valid_moves():
                    break
                print("Invalid move, try again.")
            except ValueError:
                print("Please enter a number between 0 and 8.")
        state, reward, done, info = env.step(human_action, player=-1)
        if info:
            print(info)
            break
        if done:
            env.render()
            if env.winner == 1:
                print("Agent (X) wins!")
            elif env.winner == -1:
                print("You (O) win!")
            else:
                print("It's a draw!")
            break

def get_human_action(env):
    env.render()
    human_action = 0
    while True:
        try:
            human_action = int(input("Your move (0-8): "))
            if human_action in env.get_valid_moves():
                break
            print("Invalid move, try again.")
        except ValueError:
            print("Please enter a number between 0 and 8.")
    return human_action

def continue_to_next_round():
    return input("Press y to continue: ") == 'y'

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # print("Training PPO agent...")
    policy_path, value_path = train_ppo()

    # policy_path = "tic_tac_toe_policy_model.h5"
    # value_path = "tic_tac_toe_value_model.h5"
    # play_game(policy_path, value_path)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
