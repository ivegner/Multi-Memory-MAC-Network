import sys
import os
import pickle
from collections import Counter

import numpy as np
import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import click

from dataset import CLEVR, collate_data, transform
from model import MACNetwork

batch_size = 64
dim = 512

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def accumulate(model1, model2, decay=0.999):
    par1 = dict(model1.named_parameters())
    par2 = dict(model2.named_parameters())

    for k in par1.keys():
        par1[k].data.mul_(decay).add_(1 - decay, par2[k].data)


def train(net, accum_net, optimizer, criterion, clevr_dir, epoch):
    clevr = CLEVR(clevr_dir, transform=transform)
    train_set = DataLoader(clevr, batch_size=batch_size, num_workers=4, collate_fn=collate_data)

    dataset = iter(train_set)
    pbar = tqdm(dataset)
    moving_loss = 0

    net.train(True)
    for image, question, q_len, answer, _ in pbar:
        image, question, answer = (image.to(device), question.to(device), answer.to(device))

        net.zero_grad()
        output = net(image, question, q_len)
        loss = criterion(output, answer)
        loss.backward()
        optimizer.step()
        correct = output.detach().argmax(1) == answer
        correct = torch.tensor(correct, dtype=torch.float32).sum() / batch_size

        if moving_loss == 0:
            moving_loss = correct

        else:
            moving_loss = moving_loss * 0.99 + correct * 0.01

        pbar.set_description(
            "Epoch: {}; Loss: {:.5f}; Acc: {:.5f}".format(epoch + 1, loss.item(), moving_loss)
        )

        accumulate(accum_net, net)

    clevr.close()


def valid(accum_net, clevr_dir, epoch):
    clevr = CLEVR(clevr_dir, "val", transform=None)
    valid_set = DataLoader(clevr, batch_size=batch_size, num_workers=4, collate_fn=collate_data)
    dataset = iter(valid_set)

    accum_net.train(False)
    family_correct = Counter()
    family_total = Counter()
    with torch.no_grad():
        for image, question, q_len, answer, family in tqdm(dataset):
            image, question = image.to(device), question.to(device)

            output = accum_net(image, question, q_len)
            correct = output.detach().argmax(1) == answer.to(device)
            for c, fam in zip(correct, family):
                if c:
                    family_correct[fam] += 1
                family_total[fam] += 1

    with open("log/log_{}.txt".format(str(epoch + 1).zfill(2)), "w") as w:
        for k, v in family_total.items():
            w.write("{}: {:.5f}\n".format(k, family_correct[k] / v))

    print("Avg Acc: {:.5f}".format(sum(family_correct.values()) / sum(family_total.values())))

    clevr.close()


@click.command()
@click.argument("clevr_dir")
@click.option("-l", "--load", "load_filename", type=str, help="load a model")
@click.option("-e", "--n-epochs", default=20, show_default=True, help="Number of epochs")
@click.option("-m", "--n-memories", default=3, show_default=True, help="Number of memories for the network")
@click.option(
    "-t",
    "--only-test",
    default=False,
    is_flag=True,
    show_default=True,
    help="Do not train. Only run tests and export results for visualization.",
)
# @click.option(
#     "--strict-load/--no-strict-load",
#     default=True,
#     show_default=True,
#     help="Whether to load the model (from --load) strictly or loosely (loosely = ignore missing params in load file)",
# )
def main(clevr_dir, load_filename=None, n_epochs=20, n_memories=3, only_test=False):
    with open(os.path.join(clevr_dir, "preprocessed/dic.pkl"), "rb") as f:
        dic = pickle.load(f)

    n_words = len(dic["word_dic"]) + 1
    n_answers = len(dic["answer_dic"])

    net = MACNetwork(n_words, dim, n_memories=n_memories)
    accum_net = MACNetwork(n_words, dim, n_memories=n_memories)
    net = net.to(device)
    accum_net = accum_net.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=1e-4)
    start_epoch = 0
    if load_filename:
        checkpoint = torch.load(load_filename)
        if checkpoint.get("model_state_dict", None) is None:
            # old format - just the net, not a dict of stuff
            print("Loading old-format checkpoint...")
            net.load_state_dict(checkpoint)
        else:
            # new format
            net.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            start_epoch = checkpoint["epoch"]
            print(f"Starting at epoch {start_epoch}")

    if device.type == "cuda" and torch.cuda.device_count() > 1:
        print("Using", torch.cuda.device_count(), "GPUs!")
        net = nn.DataParallel(net)
        accum_net = nn.DataParallel(accum_net)

    accumulate(accum_net, net, 0)  # copy net's parameters to accum_net

    if not only_test:
        # do training and validation
        for epoch in range(start_epoch, n_epochs):
            train(net, accum_net, optimizer, criterion, clevr_dir, epoch)
            valid(accum_net, clevr_dir, epoch)

            with open("checkpoint/checkpoint_{}.model".format(str(epoch + 1).zfill(2)), "wb") as f:
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": accum_net.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                    },
                    f,
                )
    else:
        # predict on the test set and make visualization data
        pass


if __name__ == "__main__":
    main()
