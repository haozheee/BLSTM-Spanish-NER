# loading text corpus
import numpy as np
import torch
from torch.autograd import Variable
from torch.utils import data
from torch.utils.data import DataLoader


class LocalDataLoader:
    all_vocab = []
    dict_vocab = {}

    all_ner = []
    dict_ner = {}

    train_sentence = []
    train_labels = []

    train_path = "./esp.train"
    test_path = "./esp.testb"

    max_sentence_len = 0

    debug = True

    def debug_print(self, text):
        if self.debug:
            print(text)

    def load_data(self):
        with open(self.train_path) as f:
            sent_x = []
            sent_y = []
            for i, l in enumerate(f.read().splitlines()):
                if len(l) < 1:
                    if len(sent_x) > self.max_sentence_len:
                        self.max_sentence_len = len(sent_x)
                    self.train_sentence.append(sent_x)
                    self.train_labels.append(sent_y)
                    sent_x = []
                    sent_y = []
                    length = 0

                else:
                    word = l.split(" ")[0]
                    lb = " ".join(l.split(" ")[2:])
                    if word not in self.all_vocab:
                        self.all_vocab.append(word)
                    if lb not in self.all_ner:
                        self.all_ner.append(lb)
                    sent_x.append(word)
                    sent_y.append(lb)
            # self.dict_vocab["EOS"] = 1
            self.dict_vocab = {token: index + 1 for index, token in set(enumerate(self.all_vocab))}
            self.dict_vocab["PAD"] = 0
            self.dict_ner = {token: index + 1 for index, token in set(enumerate(self.all_ner))}
            self.dict_ner["PAD"] = 0

    def load_test_data(self):
        self.load_data()
        with open(self.test_path) as f:
            sent_x = []
            sent_y = []
            self.train_sentence = []
            self.train_labels = []
            for i, l in enumerate(f.read().splitlines()):
                if len(l) < 1:
                    self.train_sentence.append(sent_x)
                    self.train_labels.append(sent_y)
                    sent_x = []
                    sent_y = []
                    length = 0

                else:
                    word = l.split(" ")[0]
                    lb = " ".join(l.split(" ")[2:])
                    sent_x.append(word)
                    sent_y.append(lb)
            # self.dict_vocab["EOS"] = 1

    def pad(self):
        pad_sentence = self.dict_vocab['PAD'] * np.ones((len(self.train_sentence), self.max_sentence_len))
        pad_labels = -1 * np.ones((len(self.train_sentence), self.max_sentence_len))

        # copy the data to the numpy array
        for j in range(len(self.train_sentence)):
            cur_len = len(self.train_sentence[j])
            # print(self.train_sentence[j])
            # print(self.sent2id(self.train_sentence[j]))
            if len(self.sent2id(self.train_sentence[j])) > self.max_sentence_len:
                continue
            pad_sentence[j] = self.sent2id(self.train_sentence[j])
            pad_labels[j] = self.label2id(self.train_labels[j])
        # since all data are indices, we convert them to torch LongTensors
        pad_sentence, pad_labels = torch.LongTensor(pad_sentence), torch.LongTensor(pad_labels)
        # convert Tensors to Variables
        sent, label = Variable(pad_sentence), Variable(pad_labels)
        train_data = Dataset(sent, label)
        loader = DataLoader(train_data, batch_size=32, shuffle=True, num_workers=8)
        return loader

    def sent2id(self, sent):
        id_sent = []
        for word in sent:
            if word in self.dict_vocab:
                id_sent.append(self.dict_vocab[word])
            else:
                id_sent.append(self.dict_vocab["PAD"])
        pad = [0] * self.max_sentence_len
        pad[:len(id_sent)] = id_sent
        return torch.LongTensor(pad)

    def label2id(self, sent):
        id_sent = []
        for ner in sent:
            id_sent.append(self.dict_ner[ner])
        pad = [0] * self.max_sentence_len
        pad[:len(id_sent)] = id_sent
        return pad


class Dataset(data.Dataset):
    'Characterizes a dataset for PyTorch'
    def __init__(self, x, y):
        'Initialization'
        self.x = x
        self.y = y

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.x)

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        x = self.x[index]
        # Load data and get label
        y = self.y[index]

        return x, y
