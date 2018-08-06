import torch
from torch.utils.data import DataLoader
import argparse

from models import general_FFTNet
from dataset import CMU_Dataset
from hparams import hparams

parser = argparse.ArgumentParser(description='FFTNet vocoder training.')
parser.add_argument('--dir', type=str, default='training_data')


def main():
    print('==> Loading Dataset..')
    training_dataset = CMU_Dataset("training_data")
    training_loader = DataLoader(training_dataset, batch_size=hparams.batch_size, num_workers=4, shuffle=True)

    test_dataset = CMU_Dataset("training_data", False)
    test_loader = DataLoader(test_dataset, batch_size=1, num_workers=4, shuffle=False)

    print('==> Building model..')
    net = general_FFTNet(radixs=hparams.radixs, aux_channels=hparams.n_mfcc + 1, channels=hparams.fft_channels,
                         classes=hparams.quantization_channels).cuda()

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=hparams.learning_rate)

    print("Start Training.")
    global_step = 0
    while global_step < hparams.training_steps:
        for batch_idx, (inputs, targets, features) in enumerate(training_loader):
            inputs, targets, features = inputs.cuda(), targets.cuda(), features.cuda()

            if hparams.noise_injecting:
                inputs += torch.zeros_like(inputs) / hparams.quantization_channels

            optimizer.zero_grad()

            logits = net(inputs, features)[:, :, 1:]
            loss = criterion(logits.unsqueeze(-1), targets.unsqueeze(-1))
            loss.backward()
            optimizer.step()

            print(global_step, "{:.4f}".format(loss.item()))
            global_step += 1
            if global_step > hparams.training_steps:
                break

    """
    print("Evaluating...")
    net.eval()
    with torch.no_grad():
        for batch_idx, (targets, features) in enumerate(test_loader):
            targets, features = targets.cuda(), features.cuda()

            logits = net.fast_generate(h=features, c=hparams.c)
            loss = criterion(logits.unsqueeze(-1), targets.unsqueeze(-1))
    """


if __name__ == '__main__':
    main()
