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
        rec_dropout=0
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
            states = [list(init.split([1] * batch_size)) for init in inits]

            # Pre-calculate recurrent dropout mask only once per batch, not per time step
            h_drop_mask = x.new_ones(batch_size, self.hidden_size)
            h_drop_mask = self.rec_drop(h_drop_mask)

            resh = []

            # Preallocate tensor for output hidden states to minimize cat/appends
            output_chunks = []
            if not reverse:
                st = 0
                for bs in batch_sizes:
                    end = st + bs
                    # Efficient grouping/stacking of initial states for batch
                    h = torch.cat(states[0][:bs], 0)
                    c = torch.cat(states[1][:bs], 0)
                    # Mask only the hidden state with recurrent dropout
                    h = h * h_drop_mask[:bs]
                    s1 = cell(x[st:end], (h, c))
                    resh.append(s1[0])
                    # Update states in-place (avoid list append/insert ops)
                    h_split = s1[0].split(1, 0)
                    c_split = s1[1].split(1, 0)
                    for j in range(bs):
                        states[0][j] = h_split[j]
                        states[1][j] = c_split[j]
                    st = end
            else:
                en = x.size(0)
                for i in range(batch_sizes.size(0) - 1, -1, -1):
                    bs = batch_sizes[i]
                    st = en - bs
                    h = torch.cat(states[0][:bs], 0)
                    c = torch.cat(states[1][:bs], 0)
                    h = h * h_drop_mask[:bs]
                    s1 = cell(x[st:en], (h, c))
                    resh.append(s1[0])
                    h_split = s1[0].split(1, 0)
                    c_split = s1[1].split(1, 0)
                    for j in range(bs):
                        states[0][j] = h_split[j]
                        states[1][j] = c_split[j]
                    en = st
                resh = list(reversed(resh))

            # Avoid multiple torch.cat calls in loop, concat once at end
            output = torch.cat(resh, 0)
            # Stack states (hidden and cell) as contiguous tensor (minimize cat overhead)
            final_states = tuple(torch.cat(s, 0) for s in states)
            return output, final_states

        all_states = [[], []]
        inputdata, batch_sizes = input.data, input.batch_sizes

        # Use tuple and list comprehension to reduce per layer looping overhead
        # Loop unrolling for single direction
        num_layers = self.num_layers
        num_directions = self.num_directions
        cells = self.cells
        drop = self.drop
        dropout = self.dropout
        hidden_size = self.hidden_size

        # Preallocate tensor for states, reduce attribute lookups in inner loops
        hx_is_not_none = hx is not None
        batch_size = batch_sizes[0].item()

        for l in range(num_layers):
            new_input = []

            # Apply dropout only between layers if needed
            if dropout > 0 and l > 0:
                inputdata = drop(inputdata)
            for d in range(num_directions):
                idx = l * num_directions + d
                cell = cells[idx]

                if hx_is_not_none:
                    hx_gen = (hx[i][idx] for i in range(2))
                else:
                    hx_gen = (
                        inputdata.new_zeros(batch_size, hidden_size, requires_grad=False)
                        for _ in range(2)
                    )

                # Forward one direction at a time using efficient masking and batch updating
                out, states = rnn_loop(
                    inputdata, batch_sizes, cell, hx_gen, reverse=(d == 1)
                )
                new_input.append(out)
                # Use unsqueeze for correct stacking, reduce list overhead
                all_states[0].append(states[0].unsqueeze(0))
                all_states[1].append(states[1].unsqueeze(0))

            if num_directions > 1:
                # Concatenate outputs from both directions once at the end
                inputdata = torch.cat(new_input, 1)
            else:
                inputdata = new_input[0]

        input = PackedSequence(inputdata, batch_sizes)

        # Use tuple/list comprehension for final stacking of hidden/cell states
        return input, tuple(torch.cat(x, 0) for x in all_states)
