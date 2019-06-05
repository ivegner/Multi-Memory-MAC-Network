import torch
from torch.autograd import Variable
from torch import nn
from torch.nn.init import kaiming_uniform_, xavier_uniform_, normal
import torch.nn.functional as F


def linear(in_dim, out_dim, bias=True):
    lin = nn.Linear(in_dim, out_dim, bias=bias)
    xavier_uniform_(lin.weight)
    if bias:
        lin.bias.data.zero_()

    return lin


class ControlUnit(nn.Module):
    def __init__(self, dim, max_step):
        super().__init__()

        self.position_aware = nn.ModuleList()
        for i in range(max_step):
            self.position_aware.append(linear(dim * 2, dim))

        self.control_question = linear(dim * 2, dim)
        self.attn = linear(dim, 1)

        self.dim = dim

    def forward(self, step, context, question, control):
        position_aware = self.position_aware[step](question)

        control_question = torch.cat([control, position_aware], 1)
        control_question = self.control_question(control_question)
        control_question = control_question.unsqueeze(1)

        context_prod = control_question * context
        attn_weight = self.attn(context_prod)

        attn = F.softmax(attn_weight, 1)

        next_control = (attn * context).sum(1)

        return next_control


class ReadUnit(nn.Module):
    def __init__(self, dim, n_memories):
        super().__init__()
        self.mem = nn.ModuleList([linear(dim, dim) for _ in range(n_memories)])
        self.concat = nn.ModuleList([linear(dim * 2, dim) for _ in range(n_memories)])
        self.attn = nn.ModuleList([linear(dim, 1) for _ in range(n_memories)])
        self.n_memories = n_memories

    def forward(self, memory, know, control):
        read = []
        for i in range(self.n_memories):
            mem = self.mem[i](memory[-1][i]).unsqueeze(2)
            concat = self.concat[i](torch.cat([mem * know, know], 1).permute(0, 2, 1))
            attn = concat * control[-1].unsqueeze(1)
            attn = self.attn[i](attn).squeeze(2)
            attn = F.softmax(attn, 1).unsqueeze(1)

            read.append((attn * know).sum(2))

        return read


class WriteUnit(nn.Module):
    def __init__(self, dim, n_memories, self_attention=False, memory_gate=False):
        super().__init__()

        self.concat = nn.ModuleList([linear(dim * 2, dim) for _ in range(n_memories)])

        # attns to other memories
        self.cross_attns = nn.ModuleList([linear(dim, n_memories * dim) for _ in range(n_memories)])
        self.mem_update = nn.ModuleList([linear(dim, dim) for _ in range(n_memories)])

        if self_attention:
            raise Exception(
                "Self-attention with multi-memory needs to be fixed and I haven't done it yet"
            )
            self.attn = nn.ModuleList([linear(dim, 1) for _ in range(n_memories)])
            self.mem = nn.ModuleList([linear(dim, dim) for _ in range(n_memories)])

        if memory_gate:
            self.control = linear(dim, 1)

        self.self_attention = self_attention
        self.memory_gate = memory_gate
        self.n_memories = n_memories
        self.dim = dim

    def forward(self, memories, retrieved, controls):
        prev_mem = memories[-1]

        indep_mem = []
        for i in range(self.n_memories):
            concat = self.concat[i](torch.cat([retrieved[i], prev_mem[i]], 1))
            indep_mem.append(concat)

            if self.self_attention:
                controls_cat = torch.stack(controls[:-1], 2)
                attn = controls[-1].unsqueeze(2) * controls_cat
                attn = self.attn[i](attn.permute(0, 2, 1))
                attn = F.softmax(attn, 1).permute(0, 2, 1)

                # next line will fail - memories needs to be indexed with [i]
                memories_cat = torch.stack(memories, 2)
                attn_mem = (attn * memories_cat).sum(2)
                indep_mem = self.mem(attn_mem) + concat

            if self.memory_gate:
                control = self.control(controls[-1])
                gate = F.sigmoid(control)
                indep_mem[i] = gate * prev_mem[i] + (1 - gate) * indep_mem[i]

        #### NEW STUFF ###

        new_mem = []
        for i, mem_prime in enumerate(indep_mem):
            # compute attention over other memories conditioned on control
            cross_attn = self.cross_attns[i](controls[-1])
            cross_attn = cross_attn.view(-1, self.n_memories, self.dim)  # reshape from flat
            cross_attn = cross_attn.permute(1, 0, 2)  # n_memories, batch_size, dim
            cross_attn = F.softmax(cross_attn, 0)  # across memories

            # apply attn to other memories
            mem_stack = torch.stack(indep_mem, 0)
            combined_mem = (mem_stack * cross_attn).sum(0)
            # pass through update function
            combined_mem = self.mem_update[i](combined_mem)
            new_mem.append(combined_mem)

        return new_mem


class MACUnit(nn.Module):
    def __init__(
        self, dim, max_step=12, n_memories=3, self_attention=False, memory_gate=False, dropout=0.15
    ):
        super().__init__()

        self.control = ControlUnit(dim, max_step)
        self.read = ReadUnit(dim, n_memories)
        self.write = WriteUnit(dim, n_memories, self_attention, memory_gate)

        self.control_0 = nn.Parameter(torch.zeros(1, dim))
        self.mem_0 = nn.ParameterList(
            [nn.Parameter(torch.zeros(1, dim)) for _ in range(n_memories)]
        )

        self.dim = dim
        self.max_step = max_step
        self.dropout = dropout
        self.n_memories = n_memories

    def get_mask(self, x, dropout):
        mask = torch.empty_like(x).bernoulli_(1 - dropout)
        mask = mask / (1 - dropout)

        return mask

    def forward(self, context, question, knowledge):
        batch_size = question.size(0)

        control = self.control_0.expand(batch_size, self.dim)
        memory = [mem_0.expand(batch_size, self.dim) for mem_0 in self.mem_0]

        if self.training:
            control_mask = self.get_mask(control, self.dropout)
            control = control * control_mask

            memory_masks = []
            for i, mem_unit in enumerate(memory):
                memory_mask = self.get_mask(mem_unit, self.dropout)
                memory[i] = mem_unit * memory_mask
                memory_masks.append(memory_mask)

        controls = [control]
        memories = [memory]

        for i in range(self.max_step):
            control = self.control(i, context, question, control)
            if self.training:
                control = control * control_mask
            controls.append(control)

            read = self.read(memories, knowledge, controls)
            memory = self.write(memories, read, controls)
            if self.training:
                for i, (mask, mem) in enumerate(zip(memory_masks, memory)):
                    memory[i] = mem * mask
            memories.append(memory)

        return memory


class MACNetwork(nn.Module):
    def __init__(
        self,
        n_vocab,
        dim,
        batch_size=64,
        embed_hidden=300,
        max_step=12,
        self_attention=False,
        memory_gate=False,
        classes=28,
        dropout=0.15,
        n_memories=3,
    ):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(1024, dim, 3, padding=1),
            nn.ELU(),
            nn.Conv2d(dim, dim, 3, padding=1),
            nn.ELU(),
        )

        self.embed = nn.Embedding(n_vocab, embed_hidden)
        self.lstm = nn.LSTM(embed_hidden, dim, batch_first=True, bidirectional=True)
        self.lstm_proj = nn.Linear(dim * 2, dim)

        self.mac = MACUnit(
            dim,
            max_step=max_step,
            n_memories=n_memories,
            self_attention=self_attention,
            memory_gate=memory_gate,
            dropout=dropout,
        )

        self.classifier = nn.Sequential(
            linear(dim * (n_memories + 2), dim), nn.ELU(), linear(dim, classes)
        )

        self.max_step = max_step
        self.dim = dim
        self.batch_size = batch_size

        self.reset()

    def reset(self):
        self.embed.weight.data.uniform_(0, 1)

        kaiming_uniform_(self.conv[0].weight)
        self.conv[0].bias.data.zero_()
        kaiming_uniform_(self.conv[2].weight)
        self.conv[2].bias.data.zero_()

        kaiming_uniform_(self.classifier[0].weight)

    def forward(self, image, question, question_len, dropout=0.15):
        batch_size = question.size(0)

        img = self.conv(image)
        img = img.view(batch_size, self.dim, -1)

        embed = self.embed(question)
        embed = nn.utils.rnn.pack_padded_sequence(embed, question_len, batch_first=True)
        lstm_out, (h, _) = self.lstm(embed)
        lstm_out, _ = nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=True)
        lstm_out = self.lstm_proj(lstm_out)
        h = h.permute(1, 0, 2).contiguous().view(batch_size, -1)

        # Run MAC classifier
        memory = self.mac(lstm_out, h, img)  # list of mems
        memory = torch.cat(memory, 1)

        # Read out output
        out = torch.cat([memory, h], 1)
        out = self.classifier(out)

        return out
