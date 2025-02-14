from torch.distributions import Categorical
import gym
import numpy as np
if not hasattr(np, 'bool8'):
    np.bool8 = np.bool_
import torch
import torch.nn as nn
import torch.optim as optim

gamma = 0.99

class Pi(nn.Module):
  def __init__(self, in_dim, out_dim):
    super(Pi, self).__init__()
    layers = [
        nn.Linear(in_dim, 64),
        nn.ReLU(),
        nn.Linear(64, out_dim),
        ]
    self.model = nn.Sequential(*layers)
    self.onpolicy_reset()
    self.train() # set training mode
  
  def onpolicy_reset(self):
    self.log_probs = []
    self.rewards = []
  
  def forward(self, x):
    pdparam = self.model(x)
    return pdparam
    
  def act(self, state):
    x = torch.from_numpy(state.astype(np.float32)) # to tensor
    pdparam = self.forward(x) # forward pass
    pd = Categorical(logits=pdparam) # probability distribution
    action = pd.sample() # pi(a|s) in action via pd
    log_prob = pd.log_prob(action) # log_prob of pi(a|s)
    self.log_probs.append(log_prob) # store for training
    return action.item()
  
def train(pi, optimizer):
  # Inner gradient-ascent loop of REINFORCE algorithm
  T = len(pi.rewards)
  rets = np.empty(T, dtype=np.float32) # the returns
  future_ret = 0.0
  # compute the returns efficiently
  for t in reversed(range(T)):
    future_ret = pi.rewards[t] + gamma * future_ret
    rets[t] = future_ret

  rets = torch.tensor(rets)
  log_probs = torch.stack(pi.log_probs)
  loss = - log_probs * rets # gradient term; Negative for maximizing
  loss = torch.sum(loss)
  optimizer.zero_grad()
  loss.backward() # backpropagate, compute gradients
  optimizer.step() # gradient-ascent, update the weights
  return loss

def main():
    env = gym.make('CartPole-v0', render_mode="human")  # 최신 버전에서는 render_mode 지정 필요
    in_dim = env.observation_space.shape[0]  # 4
    out_dim = env.action_space.n  # 2
    pi = Pi(in_dim, out_dim)  # policy pi_theta for REINFORCE
    optimizer = optim.Adam(pi.parameters(), lr=0.01)

    for epi in range(1000):
        state = env.reset()
        if isinstance(state, tuple):  # 최신 버전에서는 (state, info) 반환 가능
            state = state[0]

        for t in range(200):  # cartpole max timestep is 200
            action = pi.act(state)
            next_state, reward, done, _, _ = env.step(action)  # 최신 버전에서는 반환값이 5개일 수도 있음
            pi.rewards.append(reward)
            state = next_state
            env.render()
            if done:
                break
        loss = train(pi, optimizer)  # train per episode
        total_reward = sum(pi.rewards)
        solved = total_reward > 195.0
        pi.onpolicy_reset()  # on-policy: clear memory after training

        print(f'Episode {epi}, loss: {loss:.4f}, total_reward: {total_reward}, solved: {solved}')

    
if __name__ == '__main__':
  main()