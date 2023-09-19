import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import BertTokenizer, BertForSequenceClassification, AdamW

# Prepare a dataset for text classification
texts = ["Good", "Average", "Poor"]
labels = [2, 1, 0]  # Positive: 2, Neutral: 1, Negative: 0

# Load the BERT tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# Tokenize the text and convert it to the model's input format
tokenized_texts = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")

# Convert labels to tensors
labels = torch.tensor(labels)

# Create a dataset
dataset = TensorDataset(tokenized_texts.input_ids, tokenized_texts.attention_mask, labels)

# Create a DataLoader
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

# Load the BERT model
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=3)

# Set up an optimizer
optimizer = AdamW(model.parameters(), lr=1e-5)

# Training loop
num_epochs = 5
for epoch in range(num_epochs):
    for batch in dataloader:
        input_ids, attention_mask, labels = batch
        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()

# Model evaluation
new_text = "Good"
tokenized_new_text = tokenizer(new_text, padding=True, truncation=True, return_tensors="pt")
with torch.no_grad():
    outputs = model(tokenized_new_text.input_ids, attention_mask=tokenized_new_text.attention_mask)
    predicted_label = torch.argmax(outputs.logits, dim=1).item()

print(new_text)
print(f"Predicted label: {predicted_label}")
