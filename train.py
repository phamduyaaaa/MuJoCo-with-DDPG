import csv
from tqdm import tqdm
import gymnasium as gym
import torch
from ddpg import Actor, Critic, init_weights
from colorama import Fore
from utils import *
from memory import ReplayMemory
import os

if __name__ == '__main__':
    checkpoint_dir = "checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)
    NUM_EPISODES = 8000
    BATCH_SIZE = 32
    UPDATE = 100
    GAMMA = 0.99
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(Fore.MAGENTA + f"Using: {DEVICE}")

    env = gym.make('Ant-v5')
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    critic = Critic(state_dim, action_dim).to(DEVICE)
    critic.load_state_dict(torch.load("critic_episode_1000.pth"))
    critic.to(DEVICE)

    actor = Actor(state_dim, action_dim).to(DEVICE)
    actor.load_state_dict(torch.load("actor_episode_1000.pth"))
    actor.to(DEVICE)

    memory = ReplayMemory(capacity=50000)

    optimizer = torch.optim.Adam(list(actor.parameters()) + list(critic.parameters()), lr=1e-4)
    loss_fn = torch.nn.MSELoss()

    max_reward = float('-inf')
    total_step = 0

    with open("reward_data.csv", "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Episode", "Total Reward", "Loss", "Epsilon"])

        progress_bar = tqdm(total=NUM_EPISODES, desc="Training Progress", ncols=100, unit="episode", colour="green", dynamic_ncols=True)

        for episode in range(NUM_EPISODES):
            observation = env.reset()[0]
            total_reward = 0
            done = False
            current_epsilon = exponential_epsilon_decay(step_idx=total_step)
            step_per_episode = 0

            while not done:
                state = torch.tensor(observation, dtype=torch.float32).unsqueeze(0).to(DEVICE)

                action = actor(state).detach().cpu().numpy().flatten()

                next_observation, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated

                memory.push(observation, action, reward, next_observation, done)

                observation = next_observation
                total_reward += reward
                step_per_episode += 1
                total_step += 1

                if len(memory) > BATCH_SIZE:
                    minibatch = memory.sample(BATCH_SIZE)
                    states, actions, rewards, next_states, dones = zip(*minibatch)

                    states = torch.tensor(np.array(states), dtype=torch.float32).to(DEVICE)
                    actions = torch.tensor(np.array(actions), dtype=torch.float32).to(DEVICE)
                    rewards = torch.tensor(np.array(rewards), dtype=torch.float32).unsqueeze(1).to(DEVICE)
                    next_states = torch.tensor(np.array(next_states), dtype=torch.float32).to(DEVICE)
                    dones = torch.tensor(np.array(dones), dtype=torch.float32).unsqueeze(1).to(DEVICE)

                    q_values = critic(states, actions)
                    with torch.no_grad():
                        next_actions = actor(next_states)
                        next_q_values = critic(next_states, next_actions)
                        target_values = rewards + GAMMA * next_q_values * (1 - dones)

                    loss = loss_fn(q_values, target_values)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    torch.cuda.empty_cache()

            if total_reward > max_reward:
                max_reward = total_reward

            progress_bar.set_postfix({
                "Reward": total_reward,
                "Max Reward": max_reward,
                "Memory": len(memory),
                "Loss": loss.item() if 'loss' in locals() else None,
                "Epsilon": current_epsilon
            })
            progress_bar.update(1)

            if episode % UPDATE == 0:
                torch.save(actor.state_dict(), f'checkpoints/actor_episode_{episode}.pth')
                torch.save(critic.state_dict(), f'checkpoints/critic_episode_{episode}.pth')
                torch.cuda.empty_cache()

            writer.writerow([episode, total_reward, loss.item() if 'loss' in locals() else None, current_epsilon])

        progress_bar.close()
        env.close()
