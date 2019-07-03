import torch
from torch.autograd import Variable
from torch import nn
from torch.nn.init import kaiming_uniform_, xavier_uniform_
import torch.nn.functional as F
import numpy as np
from modules import ImageAttnModule, TextAttnModule, linear, MACModule


class ControlUnit(nn.Module):
    def __init__(self, n_cells, cell_types, control_dim, memory_dim):
        super().__init__()
        cell_type_dim = cell_types.size(-1)
        self.attn = linear(control_dim + cell_type_dim, n_cells)
        self.n_cells = n_cells
        self.memory_dim = memory_dim
        self.control_dim = control_dim

        self.cell_types = cell_types

    def forward(self, control):
        batch_size = control.size(0)
        concat = torch.cat([self.cell_types.expand((batch_size, -1, -1)), control], dim=-1)
        # attentions across other controls
        attn = self.attn(concat)  # [n_cells, n_cells]
        attn = F.softmax(attn, 2)
        new_controls = attn @ control  # [n_cells, dim]
        return new_controls, attn


class MemoryUnit(nn.Module):
    def __init__(self, n_cells, cell_types, control_dim, memory_dim):
        super().__init__()
        cell_type_dim = cell_types.size(-1)
        self.attn = linear(cell_type_dim + control_dim, n_cells)
        self.n_cells = n_cells
        self.memory_dim = memory_dim
        self.control_dim = control_dim

        self.cell_types = cell_types

    def forward(self, control, memory):
        batch_size = control.size(0)
        concat = torch.cat([self.cell_types.expand((batch_size, -1, -1)), control], dim=-1)
        # attentions across memories
        attn = self.attn(concat)  # [n_cells, n_cells]
        attn = F.softmax(attn, 2)

        new_memories = attn @ memory  # [n_cells, dim]
        return new_memories, attn


class TransformUnit(nn.Module):
    def __init__(self, n_cells, cell_types, control_dim, memory_dim, n_filter_nn_layers=1):
        super().__init__()
        cell_type_dim = cell_types.size(-1)
        out_flat_dim = memory_dim ** 2
        # "smooth" interpolation of neurons
        # hidden_layer_units = np.linspace(
        #     start=cell_type_dim + memory_dim, stop=out_flat_dim, num=n_filter_nn_layers + 2, dtype=int
        # ).tolist()
        hidden_layer_units = [
            cell_type_dim + control_dim,
            cell_type_dim + control_dim,
            out_flat_dim,
        ]

        # assemble layers
        layers = []
        for i, layer_units in enumerate(hidden_layer_units):
            if i < len(hidden_layer_units) - 1:
                layers.append(linear(layer_units, hidden_layer_units[i + 1]))
                layers.append(nn.SELU())

        self.filter_gen_fc = nn.Sequential(*layers)

        self.n_cells = n_cells
        self.memory_dim = memory_dim
        self.control_dim = control_dim
        self.cell_type_dim = cell_type_dim

        self.cell_types = cell_types

    def forward(self, control, memory):

        batch_size = control.size(0)
        concat = torch.cat([self.cell_types.expand((batch_size, -1, -1)), control], dim=-1)
        concat_flat = concat.view(-1, (self.cell_type_dim + self.control_dim))
        transformations = self.filter_gen_fc(concat_flat)
        transformations = transformations.view(
            [batch_size, self.n_cells, self.memory_dim, self.memory_dim]
        )

        # (b, n_cells, dim, dim) * (b, n_cells, dim, 1)
        new_memories = transformations @ memory.unsqueeze(-1)
        new_memories = new_memories.squeeze(-1)
        return new_memories


class WriteUnit(nn.Module):
    def __init__(self, n_cells, control_dim, memory_dim, self_attention=False, memory_gate=False):
        super().__init__()

        if self_attention:
            raise NotImplementedError("Haven't done self-attention yet")
            self.attn = linear(dim, 1)
            self.mem = linear(dim, dim)

        if memory_gate:
            self.control = linear(control_dim, 1)

        self.self_attention = self_attention
        self.memory_gate = memory_gate

        self.control_update = nn.GRU(control_dim, control_dim)
        self.memory_update = nn.GRU(memory_dim, memory_dim)

        self.n_cells = n_cells
        self.control_dim = control_dim
        self.memory_dim = memory_dim

    def forward(self, controls, memories, new_controls, new_memories):
        # process new memories and controls?
        prev_memory = memories[-1]
        prev_control = controls[-1]
        # combined = new_memories * new_controls

        # if self.self_attention:
        #     controls_cat = torch.stack(controls[:-1], 2)
        #     attn = controls[-1].unsqueeze(2) * controls_cat
        #     attn = self.attn(attn.permute(0, 2, 1))
        #     attn = F.softmax(attn, 1).permute(0, 2, 1)

        #     memories_cat = torch.stack(memories, 2)
        #     attn_mem = (attn * memories_cat).sum(2)
        #     combined = self.mem(attn_mem) + combined

        batch_size = new_memories.size(0)

        def flatten_for_gru(p, dim):
            # flatten into single-batch units for GRU update
            if not p.is_contiguous():
                p = p.contiguous()
            return p.view([1, -1, dim])

        prev_memory_flat, prev_control_flat, new_memories_flat, new_controls_flat = map(
            lambda x: flatten_for_gru(*x),
            (
                (prev_memory, self.memory_dim),
                (prev_control, self.control_dim),
                (new_memories, self.memory_dim),
                (new_controls, self.control_dim),
            ),
        )

        next_control = self.control_update(new_controls_flat, prev_control_flat)[1]
        next_memory = self.memory_update(new_memories_flat, prev_memory_flat)[1]

        next_control = next_control.view([batch_size, self.n_cells, self.control_dim])
        next_memory = next_memory.view([batch_size, self.n_cells, self.memory_dim])

        if self.memory_gate:
            control = self.control(prev_control_flat.squeeze(0))
            gate = F.sigmoid(control).view([batch_size, self.n_cells, 1])
            next_memory = gate * prev_memory + (1 - gate) * next_memory
            next_control = gate * prev_control + (1 - gate) * next_control

        return next_control, next_memory


class MACUnit(MACModule):
    def __init__(
        self,
        n_cells,
        control_dim,
        memory_dim,
        cell_type_dim=32,
        self_attention=False,
        memory_gate=False,
        dropout=0.15,
    ):
        super().__init__()
        self.cell_types = nn.Parameter(torch.rand((1, n_cells, cell_type_dim)))
        self.control_unit = ControlUnit(n_cells, self.cell_types, control_dim, memory_dim)
        self.memory_unit = MemoryUnit(n_cells, self.cell_types, control_dim, memory_dim)
        self.transform_unit = TransformUnit(n_cells, self.cell_types, control_dim, memory_dim)
        self.write_unit = WriteUnit(n_cells, control_dim, memory_dim, self_attention, memory_gate)

        self.mem_0 = nn.Parameter(torch.zeros(n_cells, memory_dim))
        self.control_0 = nn.Parameter(torch.zeros(n_cells, control_dim))

        self.control_dim = control_dim
        self.memory_dim = memory_dim

        self.n_cells = n_cells
        self.dropout = dropout

    def get_mask(self, x, dropout):
        mask = torch.empty_like(x).bernoulli_(1 - dropout)
        mask = mask / (1 - dropout)

        return mask

    def setup(self, batch_size):
        super().setup(batch_size)
        control = self.control_0.expand([batch_size, -1, -1])
        memory = self.mem_0.expand([batch_size, -1, -1])

        if self.training:
            self.control_mask = self.get_mask(control, self.dropout)
            self.memory_mask = self.get_mask(memory, self.dropout)
            control = control * self.control_mask
            memory = memory * self.memory_mask

        self.controls = [control]
        self.memories = [memory]
        return control, memory

    def forward(self, control_inputs, memory_inputs):
        """ Perform one step of reasoning

        control_inputs, memory_inputs - a tensor of inputs to the network. They will be put directly
            into memory and control. Shape: `[n_inputs, batch_size, dim]`
        """
        n_inputs = control_inputs.size(0)
        control_inputs = control_inputs.permute(1, 0, 2)
        memory_inputs = memory_inputs.permute(1, 0, 2)
        control = self.controls[-1].clone()
        memory = self.memories[-1].clone()
        # insert inputs. First DIM of each are control, second are memory
        control[:, :n_inputs] = control_inputs
        memory[:, :n_inputs] = memory_inputs

        raw_control, control_attn = self.control_unit(control)
        raw_memory, memory_attn = self.memory_unit(control, memory)
        transformed_memory = self.transform_unit(control, raw_memory)
        transformed_memory = raw_memory
        control, memory = self.write_unit(
            self.controls, self.memories, raw_control, transformed_memory
        )

        if self.training:
            memory = memory * self.memory_mask
            control = control * self.control_mask

        self.controls.append(control)
        self.memories.append(memory)

        return control, memory, (control_attn, memory_attn)


class MACNetwork(nn.Module):
    def __init__(
        self,
        n_vocab,
        n_cells,
        control_dim,
        memory_dim,
        embed_hidden=300,
        max_step=12,
        classes=28,
        image_feature_dim=512,
        text_feature_dim=512,
        self_attention=False,
        memory_gate=False,
        dropout=0.15,
        save_states=False,
    ):
        super().__init__()

        self.submodules = nn.ModuleDict(
            [
                (
                    "image_attn",
                    ImageAttnModule(control_dim, memory_dim, image_feature_dim=image_feature_dim),
                ),
                (
                    "text_attn",
                    TextAttnModule(
                        control_dim,
                        memory_dim,
                        n_vocab,
                        max_step=max_step,
                        embed_hidden=embed_hidden,
                        text_feature_dim=text_feature_dim,
                    ),
                ),
            ]
        )

        self.mac = MACUnit(
            n_cells,
            control_dim,
            memory_dim,
            self_attention=self_attention,
            memory_gate=memory_gate,
            dropout=dropout,
        )

        self.classifier = nn.Sequential(
            linear(memory_dim, memory_dim), nn.ELU(), linear(memory_dim, classes)
        )
        for param in self.classifier.parameters():
            param.requires_grad = False

        kaiming_uniform_(self.classifier[0].weight)

        self.max_step = max_step
        self.control_dim = control_dim
        self.memory_dim = memory_dim
        self.n_cells = n_cells
        self.image_feature_dim = image_feature_dim
        self.text_feature_dim = text_feature_dim

        self.save_states = save_states

    def forward(self, image, question, question_len, dropout=0.15):
        batch_size = question.size(0)
        self.submodules["image_attn"].setup(batch_size, image)
        self.submodules["text_attn"].setup(batch_size, question, question_len)
        control, memory = self.mac.setup(batch_size)  # [b, n_cells, state_dim]

        if self.save_states:
            saved_states = {
                "submodules": dict(
                    [
                        (name, {"control": [], "memory": [], "attn": []})
                        for name in self.submodules.keys()
                    ]
                ),
                "mac": {"control": [], "memory": [], "attn": {"control": [], "memory": []}},
            }

        n_submodules = len(self.submodules)

        for step in range(self.max_step):
            submodule_ctrl = control[:, :n_submodules].permute((1, 0, 2))
            submodule_mem = memory[:, :n_submodules].permute((1, 0, 2))
            # shape: [n_submodules, b, dim]

            kwargs = {"image_attn": {}, "text_attn": {"step": step}}

            controls, memories = [], []
            for i, (name, _module) in enumerate(self.submodules.items()):
                (c, m, attn) = _module(submodule_ctrl[i], submodule_mem[i], **kwargs[name])
                controls.append(c)
                memories.append(m)

                if self.save_states:
                    saved_states["submodules"][name]["control"].append(c)
                    saved_states["submodules"][name]["memory"].append(m)
                    saved_states["submodules"][name]["attn"].append(attn)

            # Run MAC
            cm = (torch.stack(controls, 0), torch.stack(memories, 0))
            control, memory, (control_attn, memory_attn) = self.mac(*cm)
            if self.save_states:
                saved_states["mac"]["attn"]["control"].append(control_attn)
                saved_states["mac"]["attn"]["memory"].append(memory_attn)

        if self.save_states:
            saved_states["mac"]["control"] = self.mac.controls
            saved_states["mac"]["memory"] = self.mac.memories

        # Read out output
        text_hidden_state = self.submodules["text_attn"].input[1]
        cat = memory[:, -1]  # torch.cat([control[:, -1], memory[:, -1]], -1)

        out = cat  # torch.cat([cat, text_hidden_state], 1)
        out = self.classifier(out)

        if self.save_states:
            return out, saved_states
        else:
            return out
