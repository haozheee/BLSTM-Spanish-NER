import os

import torch

from data_loader import LocalDataLoader
from model import Net

loader = LocalDataLoader()
loader.load_test_data()
test_data = loader.pad()

embedding_dim = 20
model = Net(vocab_size=len(loader.dict_vocab) + 1, embedding_dim=20,
            hidden_dim=100, output_dim=len(loader.dict_ner))

if os.path.exists('./model.pth'):
    model = torch.load("./model.pth", map_location=torch.device('cpu'))
    model.eval()

total_accuracy = 0
k = 0
confusion_matrix = torch.zeros(9, 9)
size = 0
for batch_sentences, batch_labels in test_data:
    batch_labels = batch_labels.view(-1)  # (batch_size * sequence_len)
    batch_predict = model(batch_sentences).view(-1, len(loader.dict_ner))  # (batch_size * sequence_len) * feature
    _, batch_predict = torch.max(batch_predict, 1)  # (batch_size * sequence_len)
    total = 0
    correct = 0
    for i in range(batch_predict.shape[0]):
        if batch_labels[i] != 0:
            total = total + 1
            size = size + 1
            confusion_matrix[batch_labels[i]-1, batch_predict[i]-1] = confusion_matrix[batch_labels[i]-1, batch_predict[i]-1] + 1
            if batch_predict[i] == batch_labels[i]:
                correct = correct + 1
    accuracy = correct / total
    print("batch accuracy: " + str(accuracy))
    total_accuracy = total_accuracy + accuracy
    k = k + 1
total_accuracy = total_accuracy / k
print("total accuracy: " + str(total_accuracy))

All_TP = confusion_matrix.diag()
F1 = torch.ones(9)
total_F1 = 0
for tag in range(9):
    idx = torch.ones(9).byte()
    idx[tag] = 0
    TP = All_TP[tag]
    TN = confusion_matrix[idx.nonzero()[:, None], idx.nonzero()].sum()
    FP = confusion_matrix[tag, idx].sum()
    FN = confusion_matrix[idx, tag].sum()
    F1[tag] = (2 * TP) / (2 * TP + FP + FN)
    print("T F1 Is: " + str(F1[tag]))
    print("T Weight Is: " + str(((TP+FN)/size)))
    # F1[tag] = ((TP+FN)/size) * F1[tag]
    F1[tag] = (1/9) * F1[tag]
print("Testing (macro) F1 Is: " + str(F1.sum()))

