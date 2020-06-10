import os
import argparse
import numpy as np
from sklearn.model_selection import train_test_split
import torch
from classifier import Model, EmbedDataset, Trainer

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='embed_data')
parser.add_argument('--batch_size',type=int, default=32)
parser.add_argument('--num_epoch', type=int, default=100)


if __name__ == "__main__":
    args = parser.parse_args()
    embed_img = np.load(os.path.join(args.data_dir, 'embed_img.npy'))
    label = np.load(os.path.join(args.data_dir, 'label.npy'))
    x_train, x_test, y_train, y_test = train_test_split(embed_img, label, test_size=0.2, random_state=42)
    train_ds = EmbedDataset(x_train, y_train)
    test_ds = EmbedDataset(x_test, y_test)
    train_dl = torch.utils.data.DataLoader(train_ds, args.batch_size, True)
    test_dl = torch.utils.data.DataLoader(test_ds, args.batch_size)

    model = Model()
    optimizer = torch.optim.Adam(model.parameters())
    trainer = Trainer(model, optimizer, train_dl, test_dl)
    trainer.train(args.num_epoch)