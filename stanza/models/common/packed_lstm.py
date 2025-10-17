import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence, pack_sequence, PackedSequence

class PackedLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, bias=True, batch_first=False, dropout=0, bidirectional=False, pad=False, rec_dropout=0):
        super().__init__()

        self.batch_first = batch_first
        self.pad = pad
        if rec_dropout == 0:
            # use the fast, native LSTM implementation
            self.lstm = nn.LSTM(input_size, hidden_size, num_layers, bias=bias, batch_first=batch_first, dropout=dropout, bidirectional=bidirectional)
        else:
            self.lstm = LSTMwRecDropout(input_size, hidden_size, num_layers, bias=bias, batch_first=batch_first, dropout=dropout, bidirectional=bidirectional, rec_dropout=rec_dropout)

    def forward(self, input, lengths, hx=None):
        if not isinstance(input, PackedSequence):
            input = pack_padded_sequence(input, lengths, batch_first=self.batch_first)

        res = self.lstm(input, hx)
        if self.pad:
            res = (pad_packed_sequence(res[0], batch_first=self.batch_first)[0], res[1])
        return res

class LSTMwRecDropout(nn.Module):
    """ An LSTM implementation that supports recurrent dropout """
    def __init__(
        self,
        input_size,
        hidden_size,
        num_layers,
        bias=True,
        batch_first=False,
        dropout=0,
        bidirectional=False,
        pad=False,
        rec_dropout=0,
    ):
        super().__init__()
        self.batch_first = batch_first
        self.pad = pad
        self.num_layers = num_layers
        self.hidden_size = hidden_size

        self.dropout = dropout
        self.drop = nn.Dropout(dropout, inplace=True)
        self.rec_drop = nn.Dropout(rec_dropout, inplace=True)

        self.num_directions = 2 if bidirectional else 1

        self.cells = nn.ModuleList()
        for l in range(num_layers):
            in_size = input_size if l == 0 else self.num_directions * hidden_size
            for d in range(self.num_directions):
                self.cells.append(nn.LSTMCell(in_size, hidden_size, bias=bias))

    def forward(self, input, hx=None):
        def rnn_loop(x, batch_sizes, cell, inits, reverse=False):
            # RNN loop for one layer in one direction with recurrent dropout
            # Assumes input is PackedSequence, returns PackedSequence as well
            batch_size = batch_sizes[0].item()
            # states[0] and states[1] are lists of batch_size 1Tensors
            states = [list(init.split([1] * batch_size)) for init in inits]
            # Create and apply recurrent dropout mask only once per batch (not per timestep)
            h_drop_mask = x.new_ones(batch_size, self.hidden_size)
            h_drop_mask = self.rec_drop(h_drop_mask)
            resh = []
            
            if not reverse:
                st = 0
                for bs in batch_sizes:
                    h, c = torch.cat(states[0][:bs], 0), torch.cat(states[1][:bs], 0)
                    h = h * h_drop_mask[:bs]
                    out_h, out_c = cell(x[st:st+bs], (h, c))
                    resh.append(out_h)
                    # direct assignment over list slice is faster than per-elem replacement; avoids double unsqueeze
                    states[0][:bs] = out_h.split(1, 0)
                    states[1][:bs] = out_c.split(1, 0)
                    st += bs
            else:
                en = x.size(0)
                # Use reversed range for batch_sizes
                for i in reversed(range(batch_sizes.size(0))):
                    bs = batch_sizes[i]
                    h, c = torch.cat(states[0][:bs], 0), torch.cat(states[1][:bs], 0)
                    h = h * h_drop_mask[:bs]
                    out_h, out_c = cell(x[en-bs:en], (h, c))
                    resh.append(out_h)
                    states[0][:bs] = out_h.split(1, 0)
                    states[1][:bs] = out_c.split(1, 0)
                    en -= bs
                resh.reverse()

            # Concatenate outputs along time
            return torch.cat(resh, 0), (torch.cat(states[0], 0), torch.cat(states[1], 0))

        all_states = [[], []]
        inputdata, batch_sizes = input.data, input.batch_sizes

        # Preallocate storage for hx defaults (to avoid recreating zeros repeatedly)
        # Store .new_zeros template for initialization per forward
        hx_default = tuple(
            input.data.new_zeros(input.batch_sizes[0].item(), self.hidden_size, requires_grad=False)
            for _ in range(2)
        )

        for l in range(self.num_layers):
            new_input = []

            if self.dropout > 0 and l > 0:
                inputdata = self.drop(inputdata)
            for d in range(self.num_directions):
                idx = l * self.num_directions + d
                cell = self.cells[idx]
                # pull hx for layer-direction if provided, else reuse default
                if hx is not None:
                    inits = (hx[i][idx] for i in range(2))
                else:
                    # use preconstructed zero tensors (as generator)
                    inits = (hx_default[i] for i in range(2))
                out, states = rnn_loop(
                    inputdata,
                    batch_sizes,
                    cell,
                    inits,
                    reverse=(d == 1),
                )

                new_input.append(out)
                # .unsqueeze(0) is needed to preserve shape for cat at end
                all_states[0].append(states[0].unsqueeze(0))
                all_states[1].append(states[1].unsqueeze(0))

            if self.num_directions > 1:
                # concatenate both directions
                inputdata = torch.cat(new_input, 1)
            else:
                inputdata = new_input[0]

        input = PackedSequence(inputdata, batch_sizes)

        # Only do cat for each stack once, outside the for-loop
        return input, (torch.cat(all_states[0], 0), torch.cat(all_states[1], 0))
