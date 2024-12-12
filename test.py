import torch
import gymnasium as gym
from ddpg import Actor


def run_test_trained_model(actor_path, num_episodes=10, render=False):
    env = gym.make('Ant-v5', render_mode="human" if render else None)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    actor = Actor(state_dim, action_dim)
    actor.load_state_dict(torch.load(actor_path))
    actor.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    actor.to(device)

    print(f"Testing trained model for {num_episodes} episodes...")
    total_rewards = []

    for episode in range(num_episodes):
        observation = env.reset()[0]
        total_reward = 0
        done = False

        while not done:
            state = torch.tensor(observation, dtype=torch.float32).unsqueeze(0).to(device)
            with torch.no_grad():
                action = actor(state).cpu().numpy().flatten()

            next_observation, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            observation = next_observation
            total_reward += reward

        total_rewards.append(total_reward)
        print(f"Episode {episode + 1}/{num_episodes}: Total Reward = {total_reward:.2f}")

    env.close()

    avg_reward = sum(total_rewards) / len(total_rewards)
    print(f"Average Reward over {num_episodes} episodes: {avg_reward:.2f}")


if __name__ == "__main__":
    actor_checkpoint_path = "checkpoints/actor_episode_7000.pth"
    run_test_trained_model(actor_checkpoint_path, num_episodes=5, render=True)
