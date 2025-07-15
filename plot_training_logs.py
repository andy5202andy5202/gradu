import pickle
import matplotlib.pyplot as plt

LOG_PATH = 'checkpoints/high_level_training_logs.pkl'

with open(LOG_PATH, 'rb') as f:
    logs = pickle.load(f)

rewards = logs['rewards']
losses = logs['losses']

plt.figure()
plt.plot(rewards)
plt.title('Training Rewards per Episode')
plt.xlabel('Episode')
plt.ylabel('Average Reward')
plt.savefig('checkpoints/reward_curve.png')
plt.close()

plt.figure()
plt.plot(losses)
plt.title('Training Loss')
plt.xlabel('Episode')
plt.ylabel('Loss')
plt.savefig('checkpoints/loss_curve.png')
plt.close()

print("已儲存 reward_curve.png 和 loss_curve.png 至 checkpoints/")
