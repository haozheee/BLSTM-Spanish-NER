import os

import torch
from torch import nn

from data_loader import LocalDataLoader
from model import Net

local_loader = LocalDataLoader()
local_loader.load_data()
train_data = local_loader.pad()

max_steps = 100000
embedding_dim = 20
print(len(local_loader.dict_ner))
print(local_loader.dict_ner)


model = Net(vocab_size=len(local_loader.dict_vocab) + 1, embedding_dim=20,
            hidden_dim=100, output_dim=len(local_loader.dict_ner))
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("GPU")
else:
    device = torch.device("cpu")
    print("CPU")

if os.path.exists('./model.pth'):
    model = torch.load("./model.pth")

model = model.to(device)

optimizer = torch.optim.SGD(model.parameters(), lr=0.003)
optimizer.zero_grad()
loss_fn = nn.CrossEntropyLoss()

for i in range(max_steps):
    total_loss = 0
    if i % 10 == 0:
        torch.save(model, "./model.pth")
    for batch_s, batch_l in train_data:
        batch_labels = batch_l.to(device)
        batch_sentences = batch_s.to(device)
        batch_labels = batch_labels.view(-1)  # (batch_size * sequence_len)
        batch_predict = model(batch_sentences).view(-1, len(local_loader.dict_ner))   # (batch_size * sequence_len) * feature
        idx = batch_labels > 0
        batch_labels = batch_labels[idx]
        batch_predict = batch_predict[idx]
        loss = loss_fn(batch_predict, batch_labels)
        total_loss = total_loss + loss
        loss.backward()
        optimizer.step()
    print("Iteration " + str(i) + ", Loss: " + str(total_loss))
