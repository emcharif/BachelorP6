import os
import torch
import torch.nn.functional as Function
from TemporalClassifier import TemporalClassifier
from load_datasets import load_datasets

NUM_TIMESTEPS = 20

INPUT_DIM = 11
HIDDEN_DIM = 64
OUTPUT_DIM = 3

LEARNING_RATE = 1e-4

DIMENSION = 0

graphs = load_datasets()

def identity_collate(batch):                                                                       #normalt merger PyGs Dataloader graferne sammen, her siger vi altså bare at den skal returnere listen som den er 
    return batch  

dataloader = torch.utils.data.DataLoader(
    graphs,
    batch_size=32,
    shuffle=True,
    drop_last=False,
    collate_fn=identity_collate
)
#laver modelen, det er egenligt her den laver weights fra layer til layer, alle random fra start
model = TemporalClassifier(input_dim=INPUT_DIM, hidden_dim=HIDDEN_DIM, output_dim=OUTPUT_DIM, num_timesteps=NUM_TIMESTEPS) 
# hvis der allerede er en model, så load den og fortsæt træningen, ellers start forfra
if os.path.exists("model_temporal.pth"):
    model.load_state_dict(torch.load("model_temporal.pth"))
    print("Loaded existing model, continuing training...")
else:
    print("No existing model found, training from scratch...")

# class_weights bruges til at håndtere ubalancerede klasser, så modellen ikke bare lærer at gætte den mest almindelige klasse
class_weights = torch.tensor([756/371, 756/184, 756/201]) 
# = tensor([2.04, 4.11, 3.76])
opt = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE) #bruges til at update weights

for epoch in range(50):
    total_loss = 0 #akkumulerer gennemsnitslig loss over alle batches i én epoch
    correct = 0
    total = 0
    for batch in dataloader:
        all_logits = [] #akkumulerer logits og labels for alle grafer i batchen, så vi kan beregne loss samlet for hele batchen
        all_labels = [] 
        for graph in batch:
            logits = model(graph) # kører hver graf gennem modelen og får logits, som er modelens rå output før softmax
            all_logits.append(logits) # akkumulerer logits for alle grafer i batchen
            all_labels.append(graph.y) # akkumulerer de sande labels for alle grafer i batchen

        logits = torch.cat(all_logits, dim=0) #samler alle logits i én tensor, form: [batch_size, num_classes]
        labels = torch.cat(all_labels, dim=0) #samler alle labels i én tensor, form: [batch_size]
        # beregner cross-entropy loss for hele batchen, med class_weights for at håndtere ubalancerede klasser
        loss = Function.cross_entropy(logits, labels, weight=class_weights) 

        opt.zero_grad() # nulstiller gradients før backpropagation, så de ikke akkumuleres over batches
        loss.backward() # beregner gradients for alle modelens parametre baseret på den beregnede loss
        opt.step() # opdaterer modelens parametre baseret på de beregnede gradients og den valgte learning rate
        total_loss += loss.item() # akkumulerer loss for hele batchen, så vi kan beregne gennemsnitslig loss for epochen

        preds = logits.argmax(dim=1) # konverterer logits til klasseprediktioner ved at tage indekset med højeste værdi for hver graf, form: [batch_size]
        correct += (preds == labels).sum().item() 
        total += labels.size(0) 

    print(f"Epoch {epoch:02d} | Loss: {total_loss/len(dataloader):.4f} | Accuracy: {correct/total*100:.1f}%")

torch.save(model.state_dict(), "model_temporal.pth")
print("Model saved.")
