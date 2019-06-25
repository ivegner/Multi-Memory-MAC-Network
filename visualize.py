import os
import pickle

import torch
import numpy as np
import matplotlib as plt
from PIL import Image
from matplotlib.lines import Line2D
from dataset import CLEVR
from torchvision.transforms import Resize
from image_feature import CLEVR as ImageDataset

N_VIS = 10

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def decode_token(token, dic):
    for (word, idx) in dic.items():
        if idx == token:
            return word


def visualize(net, clevr_dir, dics):
    net.eval()
    net.save_states = True
    word_dic, answer_dic = dics["word_dic"], dics["answer_dic"]

    clevr = iter(CLEVR(clevr_dir, "val", transform=None))
    images = iter(ImageDataset(clevr_dir, "val"))

    transform = Resize([224, 224])
    for i in range(N_VIS):
        prep_image, question, q_len, answer, family, imgfile = next(clevr)

        img = os.path.join(clevr_dir, "images", "val", imgfile)

        img = Image.open(img).convert("RGB")
        img = transform(img)
        img.show()

        decoded_question = []
        for token_idx in question:
            decoded_question.append(decode_token(token_idx, word_dic))
        print("Question", " ".join(decoded_question))

        prep_image, question, q_len = (
            prep_image.unsqueeze(0).to(device),
            torch.tensor(question).unsqueeze(0).to(device),
            torch.tensor(q_len).unsqueeze(0).to(device),
        )
        output, saved_states = net(prep_image, question, q_len)
        output = output.detach().argmax(1).item()
        output = decode_token(output, answer_dic)
        print(output)

        image_attns = saved_states["submodules"]["image_attn"]["attn"]
        text_attns = saved_states["submodules"]["text_attn"]["attn"]
        for (image_attn, text_attn) in zip(image_attns, text_attns):
            image_attn = image_attn.squeeze().view(14, 14).detach().cpu().numpy()
            image_attn = np.expand_dims(transform(Image.fromarray(image_attn)), -1)

            # print(image_attn)
            numpy_img = np.array(img)
            image_attn = numpy_img * image_attn
            Image.fromarray(np.uint8(image_attn)).show()

            text_attn = text_attn.squeeze()
            print(text_attn)
            input()


def plot_grad_flow(named_parameters):
    """Plots the gradients flowing through different layers in the net during training.
    Can be used for checking for possible gradient vanishing / exploding problems.

    Usage: Plug this function in Trainer class after loss.backwards() as
    "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow"""
    ave_grads = []
    max_grads = []
    layers = []
    for n, p in named_parameters:
        if (p.requires_grad) and ("bias" not in n):
            layers.append(n)
            ave_grads.append(p.grad.abs().mean())
            max_grads.append(p.grad.abs().max())
    plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.5, lw=1, color="c")
    plt.bar(np.arange(len(max_grads)), ave_grads, alpha=0.5, lw=1, color="b")
    plt.hlines(0, 0, len(ave_grads) + 1, lw=2, color="k")
    plt.xticks(range(0, len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(left=0, right=len(ave_grads))
    plt.ylim(
        bottom=-0.001, top=np.quantile(torch.tensor(max_grads).cpu(), 0.75)
    )  # zoom in on the lower gradient regions
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)
    plt.legend(
        [
            Line2D([0], [0], color="c", lw=4),
            Line2D([0], [0], color="b", lw=4),
            Line2D([0], [0], color="k", lw=4),
        ],
        ["max-gradient", "mean-gradient", "zero-gradient"],
    )
    plt.show()
