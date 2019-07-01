import os
import pickle

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import pyplot
from matplotlib.colors import ListedColormap
from PIL import Image
from matplotlib.lines import Line2D
from dataset import CLEVR
from torchvision.transforms import Resize
from image_feature import CLEVR as ImageDataset
import networkx as nx

N_VIS = 10

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def decode_token(token, dic):
    for (word, idx) in dic.items():
        if idx == token:
            return word


def round_tensor(t, n_digits=3):
    return torch.round(t * 10 ** n_digits) / (10 ** n_digits)


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
        print("Question:", " ".join(decoded_question))

        prep_image, question, q_len = (
            prep_image.unsqueeze(0).to(device),
            torch.tensor(question).unsqueeze(0).to(device),
            torch.tensor(q_len).unsqueeze(0).to(device),
        )
        output, saved_states = net(prep_image, question, q_len)
        output = output.detach().argmax(1).item()
        output = decode_token(output, answer_dic)
        print("Answer:", output)

        image_attns = saved_states["submodules"]["image_attn"]["attn"]
        text_attns = saved_states["submodules"]["text_attn"]["attn"]
        mac_control_attns = saved_states["mac"]["attn"]["control"]
        mac_memory_attns = saved_states["mac"]["attn"]["memory"]

        for (image_attn, text_attn, mac_ctrl_attn, mac_mem_attn) in zip(
            image_attns, text_attns, mac_control_attns, mac_memory_attns
        ):

            f, (img_attn_ax, mem_attn_ax, ctrl_attn_ax) = plt.subplots(1, 3, figsize=(15, 15))

            image_attn = image_attn.squeeze().view(14, 14).detach().cpu().numpy()
            image_attn = np.expand_dims(transform(Image.fromarray(image_attn)), -1)

            print("Max attention:", np.max(image_attn))
            image_attn = image_attn * (1 / np.max(image_attn))
            numpy_img = np.array(img)
            image_attn = (numpy_img * image_attn).astype("uint8")
            image = img_attn_ax.imshow(image_attn)
            img_attn_ax.set_title("Image Attentions")

            print("--- QUESTION WEIGHTS  ---")
            text_attn = text_attn.squeeze()
            print([f"{word}:{weight:.2f}" for (word, weight) in zip(decoded_question, text_attn)])

            print("--- MAC CONTROL ATTNS ---")
            print(round_tensor(mac_ctrl_attn))
            plot_graph_with_labels(mac_ctrl_attn[0], None, ctrl_attn_ax)
            ctrl_attn_ax.set_title("Control Attentions")
            print("--- MAC MEMORY ATTNS  ---")
            print(round_tensor(mac_mem_attn))
            plot_graph_with_labels(mac_mem_attn[0], None, mem_attn_ax)
            mem_attn_ax.set_title("Memory Attentions")
            print("-" * 24)

            plt.subplots_adjust(left=0.04, bottom=0.0, right=1.0, top=0.9, wspace=0.0, hspace=0.0)
            mng = plt.get_current_fig_manager()
            mng.resize(*mng.window.maxsize())

            plt.show()


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


def plot_graph_with_labels(adjacency_matrix: torch.Tensor, mylabels: list, axes=None):

    # weight = color alpha
    CMAP_N = 256
    reds = np.array([[1, 0, 0, 1]], dtype="float").repeat(CMAP_N, axis=0)
    reds[:, -1] = np.linspace(0.0, 1.0, CMAP_N)
    alpha_red_cmap = ListedColormap(reds, N=CMAP_N)

    adj = adjacency_matrix.detach().cpu().numpy().T

    gr = nx.from_numpy_matrix(adj, create_using=nx.DiGraph)
    pos = nx.circular_layout(gr)
    # nx.draw_networkx_nodes(gr, pos=pos, node_size=500, ax=axes)
    # nx.draw_networkx_labels(gr, pos=pos, ax=axes)

    # for (u, v) in gr.edges:
    #     # draw one by one to allow setting edge transparency
    #     weight = gr.get_edge_data(u, v)["weight"]
    #     nx.draw_networkx_edges(
    #         gr,
    #         pos=pos,
    #         ax=axes,
    #         edgelist=[(u, v)],
    #         # alpha=weight,
    #         edge_cmap=alpha_red_cmap,
    #         edge_color=[weight],
    #         edge_vmin=0,
    #         edge_vmax=1,
    #     )
    nx.draw(
        gr,
        pos=pos,
        ax=axes,
        node_size=500,
        with_labels=True,
        edge_cmap=alpha_red_cmap,
        edge_color=[gr.get_edge_data(u, v)["weight"] for (u, v) in gr.edges],
        edge_vmin=0,
        edge_vmax=1,
        connectionstyle="arc3,rad=0.2",
    )
