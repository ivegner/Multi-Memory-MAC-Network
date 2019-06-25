import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.init import kaiming_uniform_, xavier_uniform_


def linear(in_dim, out_dim, bias=True):
    lin = nn.Linear(in_dim, out_dim, bias=bias)
    xavier_uniform_(lin.weight)
    if bias:
        lin.bias.data.zero_()

    return lin


class MACModule(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def setup(self, batch_size: int):
        """Set inputs for module for the current forward run"""
        self.batch_size = batch_size

    def forward(self, control: torch.Tensor, memory: torch.Tensor):
        """ Perform one step of reasoning"""
        raise NotImplementedError()


class ImageAttnModule(MACModule):
    def __init__(self, state_dim, image_feature_dim=512):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1024, image_feature_dim, 3, padding=1),
            nn.ELU(),
            nn.Conv2d(image_feature_dim, image_feature_dim, 3, padding=1),
            nn.ELU(),
        )
        kaiming_uniform_(self.conv[0].weight)
        self.conv[0].bias.data.zero_()
        kaiming_uniform_(self.conv[2].weight)
        self.conv[2].bias.data.zero_()
        self.memory = linear(state_dim, image_feature_dim)
        self.concat = linear(image_feature_dim * 2, image_feature_dim)
        self.attn = linear(image_feature_dim, 1)
        self.out = linear(image_feature_dim, state_dim)

        self.image_feature_dim = image_feature_dim

        # set dynamically
        self.state_dim = state_dim
        self.input = None

    def setup(self, batch_size, image):
        super().setup(batch_size)

        image = self.conv(image)
        image = image.view(batch_size, self.image_feature_dim, -1)
        self.input = image

    def forward(self, control, memory):
        image = self.input
        # transform input from neuron into query (control+memory in MAC)
        mem = self.memory(memory).unsqueeze(2)
        # combine query with the image, and just the image as a bonus
        # permute to (batch, h*w, image_feature_dim)
        # this step may not be necessary
        concat = self.concat(torch.cat([mem * image, image], 1).permute(0, 2, 1))

        attn = concat * control.unsqueeze(1)
        attn = self.attn(attn).squeeze(2)  # generate featurewise attn
        attn = F.softmax(attn, 1).unsqueeze(1)  # softmax featurewise attns

        # attn shape is (b, 1, h*w)

        # save attentions from this step for visualization
        # self.saved_attns.append(attn)

        # sum over pixels to give (b, image_feature_dim)
        out = self.out((attn * image).sum(2))

        # slice into control and memory
        return torch.zeros_like(out), out, attn


class TextAttnModule(MACModule):
    def __init__(self, state_dim, n_vocab, max_step=12, embed_hidden=300, text_feature_dim=512):
        super().__init__()

        self.position_aware = nn.ModuleList()
        for i in range(max_step):
            self.position_aware.append(linear(state_dim * 2, state_dim))

        self.concat = linear(text_feature_dim * 2, text_feature_dim)
        self.attn = linear(text_feature_dim, 1)
        self.out = linear(text_feature_dim, state_dim)

        self.embed = nn.Embedding(n_vocab, embed_hidden)
        self.embed.weight.data.uniform_(0, 1)
        self.lstm = nn.LSTM(embed_hidden, text_feature_dim, batch_first=True, bidirectional=True)
        self.lstm_proj = nn.Linear(text_feature_dim * 2, text_feature_dim)

        # lstm_out, hidden_state
        self.state_dim = state_dim
        self.input = (None, None)

    def setup(self, batch_size, question, question_len):
        super().setup(batch_size)
        embed = self.embed(question)
        embed = nn.utils.rnn.pack_padded_sequence(embed, question_len, batch_first=True)
        lstm_out, (hidden_state, _) = self.lstm(embed)
        lstm_out, _ = nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=True)
        lstm_out = self.lstm_proj(lstm_out)
        hidden_state = hidden_state.permute(1, 0, 2).contiguous().view(batch_size, -1)
        self.input = (lstm_out, hidden_state)

    def forward(self, control, memory, step):
        context, question = self.input
        question = self.position_aware[step](question)

        query_question = torch.cat([control, question], 1)
        query_question = self.concat(query_question).unsqueeze(1)

        context_prod = query_question * context
        attn_weight = self.attn(context_prod)

        attn = F.softmax(attn_weight, 1)

        out = self.out((attn * context).sum(1))

        # slice into control and memory
        return out, torch.zeros_like(out), attn
