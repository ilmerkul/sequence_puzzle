import pandas as pd
import torch
from sklearn.model_selection import train_test_split

import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from collections import Counter

import matplotlib.pyplot as plt


class SeqDataset(Dataset):
    def __init__(self, data):
        super(SeqDataset, self).__init__()

        data = list(map(lambda x: list(map(int, x[0].split())) + [x[1]], data.values))

        self.data = torch.tensor(data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item]


class Model(nn.Module):
    def __init__(self, n_vocab, emb_size, hid_size):
        super(Model, self).__init__()

        self.hid_size = hid_size
        self.emb_size = emb_size

        self.emb = nn.Embedding(n_vocab, emb_size)
        self.rnn = nn.GRU(emb_size, hid_size, num_layers=1, bias=True, batch_first=True)
        self.fnn1 = nn.Linear(hid_size, hid_size // 2, bias=True)
        self.fnn2 = nn.Linear(hid_size // 2, 1, bias=True)
        self.act = nn.Sigmoid()

    def forward(self, x):
        x = self.emb(x)
        seq, x = self.rnn(x)
        x = self.fnn1(x.squeeze())
        x = self.act(x)
        x = self.fnn2(x)
        x = self.act(x)

        return x.squeeze()


if __name__ == '__main__':
    data = pd.read_csv('sequence_puzzle.csv')
    data_task = pd.read_csv('dataset_537384_10.txt')

    chs = Counter([ch for seq in data['sequence'].apply(list).values for ch in seq])
    print(chs.keys())
    vocab = {ch: i for i, ch in enumerate(set([ch for seq in data['sequence'].apply(list).values for ch in seq]))}

    print(data['sequence'].apply(lambda x: not x.find(':') % 2) == data['class'])
    data['tokenized_sequence'] = data['sequence'].apply(
        lambda x: ' '.join(map(lambda y: str(vocab[y]), list(x)))).astype(dtype=str)
    data_task['tokenized_sequence'] = data_task['sequence'].apply(
        lambda x: ' '.join(map(lambda y: str(vocab[y]), list(x)))).astype(dtype=str)

    X_train, X_test = train_test_split(data[['tokenized_sequence', 'class']], test_size=0.2)
    dataset_train = SeqDataset(X_train)
    dataset_test = SeqDataset(X_test)
    dataloader_train = DataLoader(dataset_train, batch_size=64, shuffle=True)
    dataloader_test = DataLoader(dataset_test, batch_size=64, shuffle=False)

    model = Model(len(vocab), 64, 128)
    loss = nn.BCELoss()
    opt = Adam(model.parameters(), lr=1e-4)
    history = []

    prepare_data_task = list(map(lambda x: list(map(int, x.split())), data_task['tokenized_sequence'].values))
    sd = torch.load('model.pth')
    model.load_state_dict(sd)

    for epoch in range(2):
        for i, batch in enumerate(dataloader_train):
            model.train()

            opt.zero_grad()
            x = model(batch[:, :-1])
            loss_t = loss(x, batch[:, -1].to(dtype=torch.float))

            loss_t.backward()

            opt.step()

            if i % 20 == 0:
                model.eval()

                history.append(loss_t.item())
                with torch.no_grad():
                    loss_test = 0
                    accuracy = 0
                    for b in dataloader_test:
                        x = model(b[:, :-1])
                        loss_t = loss(x, b[:, -1].to(dtype=torch.float))

                        loss_test += loss_t.item()
                        accuracy += torch.sum(b[:, -1] == (x > 0.5)).item()
                    loss_test /= len(dataloader_test)
                    accuracy /= (len(dataloader_test) * 64)

                print(f'{i}/{len(dataloader_train)}', loss_t.item(), loss_test, accuracy)

    pd.Series((model(torch.tensor(prepare_data_task)) > 0.5).to(dtype=torch.int)).to_csv(
        'sequence_puzzle_predictions.csv', index=False, header=False)

    torch.save(model.state_dict(), 'model.pth')
    plt.plot(range(len(history)), history)
    plt.show()
