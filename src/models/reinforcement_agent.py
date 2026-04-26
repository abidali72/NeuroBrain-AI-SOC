"""
🧠 Neuro Brain — Reinforcement Learning Agent
Deep Q-Network (DQN) that learns optimal actions through trial and error.

How RL differs from supervised learning:
─────────────────────────────────────────────────────────────
  SUPERVISED:  "Here are 1000 labeled examples. Learn the pattern."
  RL:          "Here's an environment. Try things. I'll tell you
               what's good (+reward) and bad (-reward). Figure it out."
─────────────────────────────────────────────────────────────

Key Concepts:
  • State:    What the agent currently observes
  • Action:   What the agent decides to do
  • Reward:   Score received after taking an action
  • Policy:   The agent's strategy (state → action mapping)
  • Q-Value:  Expected future reward for taking action A in state S

Usage:
    python -m src.models.reinforcement_agent
"""

import numpy as np
import random
from collections import deque

import torch
import torch.nn as nn
import torch.optim as optim


class DQNetwork(nn.Module):
    """
    Deep Q-Network — a neural network that estimates Q-values.

    Architecture:
        State (observations) → Dense(128)+ReLU → Dense(128)+ReLU → Q-values (one per action)

    Q-value = "how good is it to take action A in state S?"
    The agent picks the action with the highest Q-value.
    """

    def __init__(self, state_size, action_size):
        super().__init__()
        self.network = nn.Sequential(
            # Input: current state (e.g., cart position, pole angle)
            nn.Linear(state_size, 128),
            nn.ReLU(),                    # ReLU: learns which features matter

            nn.Linear(128, 128),
            nn.ReLU(),                    # ReLU: deeper pattern extraction

            # Output: one Q-value per possible action
            # NO activation — Q-values can be any real number
            nn.Linear(128, action_size),
        )

    def forward(self, state):
        return self.network(state)


class ReplayMemory:
    """
    Experience Replay Buffer — stores past experiences for learning.

    WHY replay memory?
    Without it, the agent learns from consecutive experiences, which are
    highly correlated (step 1 looks like step 2). This makes training unstable.
    By storing experiences and sampling RANDOMLY, we break this correlation
    and get much more stable learning.
    """

    def __init__(self, capacity=10000):
        self.memory = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        """Store one experience."""
        self.memory.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        """Randomly sample a batch of experiences."""
        batch = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards, dtype=np.float32),
            np.array(next_states),
            np.array(dones, dtype=np.float32),
        )

    def __len__(self):
        return len(self.memory)


class RLAgent:
    """
    Deep Q-Learning Agent.

    TRAINING PARAMETERS:
    ─────────────────────────────────────────────────────────
    • gamma (0.99) — Discount factor
      How much the agent values future rewards vs immediate ones.
      0.99 = "future rewards are almost as valuable as current"
      0.5  = "only care about the next few steps"

    • epsilon (1.0 → 0.01) — Exploration rate
      Probability of taking a RANDOM action instead of the "best" one.
      Starts at 1.0 (100% random) and decays to 0.01 (1% random).
      WHY? Early on, the agent knows nothing — random exploration
      helps it discover good strategies. Later, it exploits what it learned.

    • learning_rate (0.001) — How fast the network updates
      Same as supervised learning.

    • batch_size (64) — Experiences sampled per training step
      Sampled randomly from replay memory.

    • target_update (10) — Steps between target network sync
      We use TWO networks: one for choosing actions, one for
      computing target Q-values. This prevents instability.
    ─────────────────────────────────────────────────────────
    """

    def __init__(self, state_size, action_size, lr=0.001, gamma=0.99,
                 epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma                # Discount factor for future rewards
        self.epsilon = epsilon            # Exploration rate
        self.epsilon_min = epsilon_min    # Minimum exploration
        self.epsilon_decay = epsilon_decay  # Decay per episode

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Two networks: policy (decides actions) + target (computes stable Q-targets)
        self.policy_net = DQNetwork(state_size, action_size).to(self.device)
        self.target_net = DQNetwork(state_size, action_size).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.criterion = nn.MSELoss()
        self.memory = ReplayMemory(capacity=10000)
        self.batch_size = 64

    def choose_action(self, state):
        """
        Epsilon-greedy action selection.

        With probability epsilon: take RANDOM action (explore)
        With probability 1-epsilon: take BEST action (exploit)
        """
        if random.random() < self.epsilon:
            # EXPLORE: random action to discover new strategies
            return random.randrange(self.action_size)
        else:
            # EXPLOIT: use the network to pick the best action
            state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            with torch.no_grad():
                q_values = self.policy_net(state_t)
            return q_values.argmax(dim=1).item()

    def learn(self):
        """
        Train the policy network using a batch from replay memory.

        The Q-learning update rule:
        Q(s, a) = reward + gamma × max_a' Q(s', a')
        "The value of an action = immediate reward + discounted future value"
        """
        if len(self.memory) < self.batch_size:
            return 0.0

        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)

        # Convert to tensors
        states_t = torch.FloatTensor(states).to(self.device)
        actions_t = torch.LongTensor(actions).to(self.device)
        rewards_t = torch.FloatTensor(rewards).to(self.device)
        next_states_t = torch.FloatTensor(next_states).to(self.device)
        dones_t = torch.FloatTensor(dones).to(self.device)

        # Current Q-values: Q(s, a) from policy network
        current_q = self.policy_net(states_t).gather(1, actions_t.unsqueeze(1)).squeeze()

        # Target Q-values: r + gamma * max_a' Q_target(s', a')
        with torch.no_grad():
            next_q = self.target_net(next_states_t).max(1)[0]
            target_q = rewards_t + (1 - dones_t) * self.gamma * next_q

        # Loss: how far off are our Q-value estimates?
        loss = self.criterion(current_q, target_q)

        # Update policy network
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def update_target_network(self):
        """Sync target network with policy network."""
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def decay_epsilon(self):
        """Reduce exploration rate over time."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save(self, path="data/models/rl_agent.pth"):
        """Save the trained agent."""
        torch.save({
            'policy_net': self.policy_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'epsilon': self.epsilon,
        }, path)
        print(f"💾 RL Agent saved to {path}")

    def load(self, path="data/models/rl_agent.pth"):
        """Load a trained agent."""
        checkpoint = torch.load(path, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint['policy_net'])
        self.target_net.load_state_dict(checkpoint['target_net'])
        self.epsilon = checkpoint['epsilon']
        print(f"📂 RL Agent loaded from {path}")


def train_cartpole():
    """
    Demo: Train an RL agent to balance a pole on a cart.

    CartPole Environment:
    ─────────────────────────────────────────────────────────
    State:  [cart_position, cart_velocity, pole_angle, pole_velocity]
    Actions: 0 = push left, 1 = push right
    Reward:  +1 for each step the pole stays upright
    Done:    pole falls past 15° or cart goes off-screen
    Goal:    keep the pole balanced for 500 steps
    ─────────────────────────────────────────────────────────
    """
    try:
        import gymnasium as gym
    except ImportError:
        print("⚠️  Install gymnasium: pip install gymnasium")
        print("   Then run: python -m src.models.reinforcement_agent")
        return

    print("=" * 55)
    print("  🎮 Reinforcement Learning — CartPole Demo")
    print("=" * 55)

    env = gym.make('CartPole-v1')
    state_size = env.observation_space.shape[0]  # 4
    action_size = env.action_space.n              # 2

    agent = RLAgent(state_size, action_size)
    episodes = 300
    target_update_freq = 10
    scores = []

    print(f"\n   State size:  {state_size} (position, velocity, angle, angular vel)")
    print(f"   Actions:     {action_size} (left, right)")
    print(f"   Episodes:    {episodes}")
    print(f"   Epsilon:     {agent.epsilon:.2f} → {agent.epsilon_min}")
    print()

    for episode in range(episodes):
        state, _ = env.reset()
        total_reward = 0
        done = False

        while not done:
            action = agent.choose_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            agent.memory.push(state, action, reward, next_state, done)
            agent.learn()

            state = next_state
            total_reward += reward

        agent.decay_epsilon()
        scores.append(total_reward)

        if (episode + 1) % target_update_freq == 0:
            agent.update_target_network()

        if (episode + 1) % 20 == 0:
            avg = np.mean(scores[-20:])
            print(f"   Episode {episode+1:>3} │ Score: {total_reward:>5.0f} │ "
                  f"Avg(20): {avg:>6.1f} │ ε: {agent.epsilon:.3f}")

    env.close()

    final_avg = np.mean(scores[-20:])
    print(f"\n   ✅ Training complete! Final avg score: {final_avg:.1f}")
    if final_avg >= 400:
        print(f"   🎉 Agent SOLVED CartPole! (avg ≥ 400)")
    agent.save()


if __name__ == "__main__":
    train_cartpole()
