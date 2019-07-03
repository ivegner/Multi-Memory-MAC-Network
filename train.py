import sys
import os
import pickle
from collections import Counter

import numpy as np
import torch

torch.manual_seed(0)
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import click
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from dataset import CLEVR, collate_data, transform
from model import MACNetwork
from visualize import plot_grad_flow, visualize


batch_size = 128

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
    for i, (image, question, q_len, answer, _, _) in enumerate(pbar):
        image, question, answer = (image.to(device), question.to(device), answer.to(device))

        m = net.module if isinstance(net, nn.DataParallel) else net

        net.zero_grad()
        output = net(image, question, q_len)
        loss = criterion(output, answer)

        # def get_batch_mean_pdist(mat):  # [b, n_cells, dim]
        #     return torch.stack([torch.pdist(b, p=2) for b in mat], 0).mean(0)

        # # get similarity of states between cells

        # control_similarity = get_batch_mean_pdist(
        #     torch.stack(saved_states["mac"]["control"], 0).view(
        #         (-1, m.n_cells, m.state_dim)
        #     )
        # )
        # memory_similarity = get_batch_mean_pdist(
        #     torch.stack(saved_states["mac"]["memory"], 0).view((-1, m.n_cells, m.state_dim))
        # )

        # # print(control_similarity, memory_similarity)

        # state_redundancy = (control_similarity + memory_similarity).sum()

        # loss += state_redundancy

        loss.backward()

        # if wrapped in a DataParallel, the actual net is at DataParallel.module
        # torch.nn.utils.clip_grad_norm_(m.mac.read.parameters(), 1)
        # torch.nn.utils.clip_grad_value_(net.parameters(), 0.05)

        # if i % 300 == 0:
        #     plot_grad_flow(net.named_parameters())

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
        for image, question, q_len, answer, family, _ in tqdm(dataset):
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

    avg_acc = sum(family_correct.values()) / sum(family_total.values())
    print(f"Avg Acc: {avg_acc:.5f}")

    clevr.close()
    return avg_acc


# def test(accum_net, clevr_dir):
#     print("Starting tests!")
#     print(accum_net)
#     clevr = CLEVR(clevr_dir, "val", transform=None)
#     test_set = DataLoader(clevr, batch_size=batch_size, num_workers=4, collate_fn=collate_data)
#     dataset = iter(test_set)

#     accum_net.train(False)
#     family_correct = Counter()
#     family_total = Counter()
#     with torch.no_grad():
#         for image, question, q_len, answer, family, _ in tqdm(dataset):
#             image, question = image.to(device), question.to(device)

#             output = accum_net(image, question, q_len)

#             # if wrapped in a DataParallel, the actual net is at DataParallel.module
#             m = accum_net.module if isinstance(accum_net, nn.DataParallel) else accum_net
#             # [{read, write}, n_steps, batch_size, {??????, n_memories}]
#             attentions = m.saved_attns
#             for i, step in enumerate(attentions):
#                 print(f"Step {i}")
#                 print("Read attn shape:", torch.tensor(step["read"][0]).shape)
#                 print(image.shape)

#             sys.exit()
#             correct = output.detach().argmax(1) == answer.to(device)
#             for c, fam in zip(correct, family):
#                 if c:
#                     family_correct[fam] += 1
#                 family_total[fam] += 1

#     with open("log/test_log.txt", "w") as w:
#         for k, v in family_total.items():
#             w.write("{}: {:.5f}\n".format(k, family_correct[k] / v))

#     print("Avg Acc: {:.5f}".format(sum(family_correct.values()) / sum(family_total.values())))

#     clevr.close()



@click.command()
@click.argument("clevr_dir")
@click.option("-t", "--model-tag", required=True, help="identifier name for this model")
@click.option("-l", "--load", "load_filename", type=str, help="load a model")
@click.option("-e", "--n-epochs", default=20, show_default=True, help="Number of epochs")
@click.option(
    "-n", "--n-cells", default=3, show_default=True, help="Number of cells for the network"
)
@click.option("-d", "--state-dim", default=512, show_default=True, help="cell state dimensions")
@click.option(
    "--only-test",
    default=False,
    is_flag=True,
    show_default=True,
    help="Do not train. Only run tests and export results for visualization.",
)
@click.option(
    "-vis",
    "--only-vis",
    default=False,
    is_flag=True,
    show_default=True,
    help="Do not train. Only visualize.",
)
@click.option(
    "--strict-load/--no-strict-load",
    default=True,
    show_default=True,
    help="Whether to load the model (from --load) strictly or loosely (loosely = ignore missing params in load file)",
)
@click.option(
    "-c",
    "--checkpoint-dir",
    type=str,
    default="checkpoint",
    show_default=True,
    help="Directory path to save the checkpoints in",
)
def main(
    clevr_dir,
    model_tag="",
    load_filename=None,
    n_epochs=20,
    n_cells=3,
    state_dim=512,
    only_test=False,
    only_vis=False,
    strict_load=True,
    checkpoint_dir="checkpoint",
):
    with open(os.path.join(clevr_dir, "preprocessed", "dic.pkl"), "rb") as f:
        dic = pickle.load(f)

    n_words = len(dic["word_dic"]) + 1
    n_answers = len(dic["answer_dic"])

    memory_dim = state_dim
    control_dim = 128

    net, accum_net = [
        MACNetwork(n_words, n_cells, control_dim, memory_dim, memory_gate=True) for _ in range(2)
    ]
    net = net.to(device)
    accum_net = accum_net.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=1e-4)
    start_epoch = 0

    print(net)
    if load_filename:
        checkpoint = torch.load(load_filename)
        net.load_state_dict(checkpoint["model_state_dict"], strict=strict_load)
        if strict_load:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        print(f"Starting at epoch {start_epoch+1}")

    if device.type == "cuda" and not (only_test or only_vis):
        print("Using", torch.cuda.device_count(), "GPUs!")
        net = nn.DataParallel(net)
        accum_net = nn.DataParallel(accum_net)

    accumulate(accum_net, net, 0)  # copy net's parameters to accum_net

    prev_accuracy = 0.0
    prev_filename = None
    if not (only_test or only_vis):
        # do training and validation
        for epoch in range(start_epoch, n_epochs):
            train(net, accum_net, optimizer, criterion, clevr_dir, epoch)
            avg_accuracy = valid(accum_net, clevr_dir, epoch)

            if avg_accuracy >= prev_accuracy:
                if not os.path.isdir(checkpoint_dir):
                    os.makedirs(checkpoint_dir)

                filename = os.path.join(checkpoint_dir, f"checkpoint_{n_cells}n_{model_tag}_{avg_accuracy}%.model")
                with open(filename, "wb") as f:
                    torch.save(
                        {
                            "epoch": epoch,
                            "model_state_dict": (
                                net.module if isinstance(net, nn.DataParallel) else net
                            ).state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(),
                        },
                        f,
                    )
                if prev_filename is not None and prev_filename != filename:
                    # delete old only after successfully saving new
                    os.remove(prev_filename)
                prev_filename = filename
    elif only_test:
        avg_accuracy = valid(accum_net, clevr_dir, epoch)
    else:
        visualize(accum_net, clevr_dir, dic)



if __name__ == "__main__":
    main()
