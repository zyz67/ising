import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
import os

# 1. Metropolis Monte Carlo for 2D Ising Model
def initialize_spin_config(L):
    return np.random.choice([-1, 1], size=(L, L))

def mcmc_step(config, T, L):
    for _ in range(L * L):
        x, y = np.random.randint(0, L), np.random.randint(0, L)
        S = config[x, y]
        nb_S = config[(x+1)%L, y] + config[(x-1)%L, y] + config[x, (y+1)%L] + config[x, (y-1)%L]
        dE = 2 * S * nb_S
        if dE <= 0 or np.random.rand() < np.exp(-dE / T):
            config[x, y] *= -1
    return config

def generate_data(L, temperatures, samples_per_T, eq_steps=1000, mc_steps=10):
    data = []
    labels = []
    Tc_exact = 2.269
    for T in temperatures:
        print(f"Generating data for T = {T:.2f}")
        config = initialize_spin_config(L)
        for _ in range(eq_steps):
            mcmc_step(config, T, L)
        
        for _ in range(samples_per_T):
            for _ in range(mc_steps):
                mcmc_step(config, T, L)
            data.append(config.copy())
            labels.append(1 if T < Tc_exact else 0) # 1 for ordered, 0 for disordered
    return np.array(data), np.array(labels)

L = 16
temperatures = np.linspace(1.0, 3.5, 26)
samples_per_T = 200

# Generate or load data
data_file = 'ising_data.npz'
if os.path.exists(data_file):
    loaded = np.load(data_file)
    X = loaded['X']
    y = loaded['y']
else:
    X, y = generate_data(L, temperatures, samples_per_T, eq_steps=500, mc_steps=5)
    np.savez(data_file, X=X, y=y)

X_tensor = torch.tensor(X, dtype=torch.float32).unsqueeze(1) # [N, 1, L, L] for CNN
y_tensor = torch.tensor(y, dtype=torch.long)

dataset = TensorDataset(X_tensor, y_tensor)
# Train-test split (we want to test on all T to see accuracy vs T, so we can train on a subset or all)
# Let's train on a random 80% split
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(dataset, batch_size=64, shuffle=False) # Test on all to get acc vs T

# 2. Classifier Network
class CNN(nn.Module):
    def __init__(self, L):
        super(CNN, self).__init__()
        # To learn "order parameter" (magnetization), a large filter or global average pooling helps.
        # But let's use a simple CNN
        self.conv1 = nn.Conv2d(1, 4, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(4 * L * L, 64)
        self.fc2 = nn.Linear(64, 2)
        
    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

model = CNN(L)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

print("Training Network...")
epochs = 10
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_loader):.4f}")

# 3. Evaluation: Confusion Matrix and Acc vs T
model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for batch_X, batch_y in test_loader:
        outputs = model(batch_X)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.numpy())
        all_labels.extend(batch_y.numpy())

all_preds = np.array(all_preds)
all_labels = np.array(all_labels)

# Confusion Matrix
cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Disordered', 'Ordered'], yticklabels=['Disordered', 'Ordered'])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.savefig('confusion_matrix.png')
plt.close()

# Accuracy vs Temperature
acc_vs_T = []
prob_ordered_vs_T = [] # Average predicted probability of ordered phase

# We need to map back preds to temperatures.
# Since dataset is ordered sequentially before split, and test_loader uses `dataset` (not split) sequentially:
y_probs = []
with torch.no_grad():
    for batch_X, _ in DataLoader(dataset, batch_size=64, shuffle=False):
        outputs = model(batch_X)
        probs = torch.softmax(outputs, dim=1)[:, 1] # prob of ordered class (label 1)
        y_probs.extend(probs.numpy())
y_probs = np.array(y_probs)

for i, T in enumerate(temperatures):
    start_idx = i * samples_per_T
    end_idx = (i + 1) * samples_per_T
    T_preds = all_preds[start_idx:end_idx]
    T_labels = all_labels[start_idx:end_idx]
    T_probs = y_probs[start_idx:end_idx]
    
    acc = np.mean(T_preds == T_labels)
    acc_vs_T.append(acc)
    prob_ordered_vs_T.append(np.mean(T_probs))

plt.figure(figsize=(8, 5))
plt.plot(temperatures, acc_vs_T, marker='o', label='Accuracy')
plt.plot(temperatures, prob_ordered_vs_T, marker='x', label='P(Ordered)')
plt.axvline(x=2.269, color='r', linestyle='--', label='Analytical Tc $\\approx 2.269$')
plt.xlabel('Temperature (T)')
plt.ylabel('Metric')
plt.title('Accuracy and Predicted Probability vs Temperature')
plt.legend()
plt.grid(True)
plt.savefig('accuracy_vs_T.png')
plt.close()

# Find estimated Tc where P(Ordered) crosses 0.5
estimated_Tc = temperatures[np.argmin(np.abs(np.array(prob_ordered_vs_T) - 0.5))]
print(f"Analytical Tc: 2.269")
print(f"Estimated Tc from network: {estimated_Tc:.3f}")

# 4. Visualize learned features (Conv1 weights)
weights = model.conv1.weight.detach().numpy()
plt.figure(figsize=(10, 3))
for i in range(4):
    plt.subplot(1, 4, i+1)
    plt.imshow(weights[i, 0], cmap='coolwarm')
    plt.title(f'Filter {i+1}')
    plt.axis('off')
plt.suptitle('Learned Convolutional Filters')
plt.tight_layout()
plt.savefig('learned_features.png')
plt.close()

print("Done. Saved confusion_matrix.png, accuracy_vs_T.png, and learned_features.png")
