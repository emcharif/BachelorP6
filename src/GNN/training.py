import torch
import torch.nn.functional as Function
from TemporalClassifier import TemporalClassifier
from load_datasets import load_datasets

NUM_TIMESTEPS = 20

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

model = TemporalClassifier(input_dim=11, hidden_dim=64, output_dim=3, num_timesteps=NUM_TIMESTEPS) #laver modelen, det er egenligt her den laver weights fra layer til layer, alle random fra start
update_weights = torch.optim.Adam(model.parameters(), lr=1e-3)                                     #bruges til at update weights

for epoch in range(30):
    total_loss = 0                                                                                 #akkumulerer average loss over alle batches i én epoch
    for batch in dataloader:                                                                       #itererer igennem en batch
        all_logits = []                                                                            #gemmer outputs for at regne total loss ud senere
        all_labels = []                                                                            #gemmer outputs for at regne total loss ud senere
        for graph in batch:                                                                        #itererer igennem én graf i batchen
            logits = model(graph)                                                                  #kører GNN for den enkelte graf og returner [1, 3], altså hvilken klasse grafen er i
            all_logits.append(logits)                                                              #appender disse to for at kigge om det den regnede ud matcher med det aktuelle label
            all_labels.append(graph.y)                                                             #appender disse to for at kigge om det den regnede ud matcher med det aktuelle label

        logits = torch.cat(all_logits, dim=0)                                                      #concatinates 32 logits individuelle [1,3] -> [32,3]
        labels = torch.cat(all_labels, dim=0)                                                      #concatinates 32 labels individuelle [1] -> [32]
        loss = Function.cross_entropy(logits, labels)                                              #måler loss for at hvor forkert den har været, den bruger softmax (kan kigges mere i dybden hvis man ønsker)

        update_weights.zero_grad()                                                                 #skal clear dem inden hver backward pass
        loss.backward()                                                                            #computes gradients, går gennem hvert lag, starter med classifier -> GRU -> GCN layers
        update_weights.step()                                                                      #først her udfører den ændringen af weights
        total_loss += loss.item()                                                                  #genererer total loss af denne epoch

    print(f"Epoch {epoch:02d} | Loss: {total_loss/len(dataloader):.4f}")

torch.save(model.state_dict(), "model_temporal.pth")