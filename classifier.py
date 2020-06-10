import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm

class EmbedDataset(torch.utils.data.Dataset):
    def __init__(self, embed_img, label):
        self.embed_img = embed_img
        self.label = label
        
    def __len__(self):
        return self.label.shape[0]

    def __getitem__(self, ix):
        return self.embed_img[ix], self.label[ix]


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.fc1 = nn.Linear(128, 512)
        self.fc2 = nn.Linear(512, 1024)
        self.fc3 = nn.Linear(1024, 1000)

    def forward(self, x, label=None):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        out = self.fc3(x)
        if label is None:
            return F.softmax(out)
        loss = F.cross_entropy(out, label)
        return out, loss


class Trainer:
    def __init__(self, model, optimizer, train_dl, val_dl, scheduler=None, device=None):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_dl = train_dl
        self.val_dl = val_dl
        self.checkpoint_dir = 'checkpoint'
        if not os.path.isdir(self.checkpoint_dir):
            os.mkdir(self.checkpoint_dir)
        if device is None:
            self.device = torch.device('cpu')
        else:
            self.device = device

    def run_epoch(self, dataloader, is_training=True):
        if is_training:
            self.model.train()
        else:
            self.model.eval()
        
        total_loss = 0
        total_item = 0
        total_predict = []
        total_label = []

        desc = "loss=%.4f | acc=%.4f | f1=%.4f"
        with tqdm(total=len(dataloader)) as pbar:
            for x,y in dataloader:
                x = x.to(self.device)
                y = y.to(self.device)
                
                self.optimizer.zero_grad()
                out, loss = self.model(x, y)

                if is_training:
                    loss.backward()
                    self.optimizer.step()
                    if self.scheduler is not None:
                        self.scheduler.step()

                total_loss += loss.item()
                total_item += x.shape[0]

                cur_predict = out.argmax(dim=-1).detach().cpu().numpy().tolist()
                cur_label = y.detach().cpu().numpy().tolist()
                total_predict.extend(cur_predict)
                total_label.extend(cur_label)

                acc = accuracy_score(total_label, total_predict)
                f1 = f1_score(total_label, total_predict, average='macro')

                pbar.update(1)
                pbar.set_description(desc%(total_loss/total_item, acc, f1))
        return total_loss/total_item

    def train(self, num_epoch):
        for epoch in range(1, num_epoch+1):
            print("Epoch %d"%epoch, flush=True)
            self.run_epoch(self.train_dl)
            self.run_epoch(self.val_dl, is_training=False)
            ckpt_path = os.path.join(self.checkpoint_dir, 'model_%02d.pth'%epoch)
            torch.save(self.model.state_dict(), ckpt_path)