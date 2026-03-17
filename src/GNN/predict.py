import glob
import os
import torch
import pandas as pd
from TemporalClassifier import TemporalClassifier

NUM_TIMESTEPS = 20

INPUT_DIM = 11
HIDDEN_DIM = 64
OUTPUT_DIM = 3

model = TemporalClassifier(input_dim = INPUT_DIM, hidden_dim = HIDDEN_DIM, output_dim = OUTPUT_DIM, num_timesteps=NUM_TIMESTEPS)
model.load_state_dict(torch.load("model_temporal.pth"))
model.eval()

labels = pd.read_csv("labels_density_3class.csv")
labels['filename'] = labels['path'].apply(lambda x: os.path.basename(x))

files = glob.glob("data/predict_dataset/*.pt")
print(f"Found {len(files)} graphs to predict\n")

correct = 0
total = 0
not_found = 0

for file in files:
    graph = torch.load(file, weights_only=False)

    with torch.no_grad():
        logits = model(graph)
        predicted_class = logits.argmax(dim=1).item()

    filename = os.path.basename(file)
    match = labels[labels['filename'] == filename]

    if len(match) == 0:
        print(f"{filename} → Predicted: {predicted_class}, True: not found in csv")
        not_found += 1
        continue

    true_label = match.iloc[0]['label']
    is_correct = predicted_class == true_label
    correct += int(is_correct)
    total += 1

    print(f"{filename} → Predicted: {predicted_class}, True: {true_label} {'✓' if is_correct else '✗'}")

print("\n=== Results ===")
print(f"Correct:  {correct} / {total}")
if total > 0:
    print(f"Accuracy: {correct/total*100:.1f}%")
if not_found > 0:
    print(f"Skipped:  {not_found} (not found in CSV)")