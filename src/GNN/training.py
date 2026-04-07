import torch
from torch_geometric.loader import DataLoader
from torch_geometric.datasets import TUDataset
from GNN.Classifier import Classifier
import torch.nn.functional as Function

#================= CONSTANTS ============================================
DATASET_NAME = "MUTAG"
BATCH_SIZE = 10
TRAIN_PERCENTAGE = 0.7
VALIDATION_PERCANTAGE = 0.15
LEARNING_RATE = 0.001
HIDDEN_DIMENSION = 128
EPOCHS = 20

#================= LOAD DATASET and divide into categories ==============
dataset = TUDataset(root="./data", name=DATASET_NAME)
dataset = dataset.shuffle()

train_size = int(TRAIN_PERCENTAGE * len(dataset))
val_size = int(VALIDATION_PERCANTAGE * len(dataset))

train_dataset = dataset[:train_size]
val_dataset = dataset[train_size:train_size + val_size]
test_dataset = dataset[train_size + val_size:]

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

#================= EVALUATE MODEL =======================================
def evaluate(model, loader):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in loader:
            logits = model(batch)
            pred = logits.argmax(dim=1)
            matches = pred == batch.y
            matches = matches.int()     
            num_correct_in_batch = matches.sum().item()  
            correct += num_correct_in_batch
            total += batch.y.size(0)

    return correct / total

#================= SETUP MODEL ===========================================
model = Classifier(input_dim=dataset.num_node_features, hidden_dim=HIDDEN_DIMENSION, output_dim=dataset.num_classes)

opt = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

#================= TRAINING LOOP =========================================
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0

    for batch in train_loader:
        logits = model(batch)
        loss = Function.cross_entropy(logits, batch.y)

        opt.zero_grad()
        loss.backward()
        opt.step()

        total_loss += loss.item()

    train_acc = evaluate(model, train_loader)
    val_acc = evaluate(model, val_loader)

    print(
        f"Epoch {epoch:02d} | "
        f"Loss: {total_loss / len(train_loader):.4f} | "
        f"Train Acc: {train_acc:.4f} | "
        f"Val Acc: {val_acc:.4f}"
    )

#================= FINAL MEASUREMENTS ====================================
test_acc = evaluate(model, test_loader)
print(f"Final Test Accuracy: {test_acc:.4f}")

torch.save(model.state_dict(), f"{DATASET_NAME}_model.pth")