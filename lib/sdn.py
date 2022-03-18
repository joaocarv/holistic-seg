import math
import torch
import torch.nn as nn


class SDNCell(nn.Module):

    def __init__(self, state_size):
        super().__init__()
        # for previous states
        self.register_parameter('weight_ih', nn.Parameter(torch.randn(3 * state_size, 3 * state_size)))
        self.register_parameter('bias_ih', nn.Parameter(torch.randn(3 * state_size)))
        # for state prior
        self.register_parameter('weight_hh', nn.Parameter(torch.randn(3 * state_size, state_size)))
        self.register_parameter('bias_hh', nn.Parameter(torch.randn(3 * state_size)))
        # Initialization
        std = 1.0 / math.sqrt(state_size)
        for w in self.parameters():
            w.data.uniform_(-std, std)

    def forward(self, prev_states, state_prior):
        i_vals = torch.addmm(self.bias_ih, torch.cat(prev_states, dim=1), self.weight_ih.t())
        h_vals = torch.addmm(self.bias_hh, state_prior, self.weight_hh.t())
        r_i, z_i, n_i = i_vals.chunk(3, 1)
        r_h, z_h, n_h = h_vals.chunk(3, 1)
        r = torch.sigmoid(r_i + r_h)
        z = torch.sigmoid(z_i + z_h)
        n = torch.tanh(n_i + (r * n_h))
        return n * z + state_prior * (1-z)



class _SDNLayer(nn.Module):

    def __init__(self, state_size, dir=0):
        super().__init__()
        self.state_size = state_size
        self.cell = SDNCell(state_size)
        if dir == 0:
            self.forward = self.forward0
        elif dir == 1:
            self.forward = self.forward1
        elif dir == 2:
            self.forward = self.forward2
        else:
            self.forward = self.forward3

    def forward0(self, states):
        # zero state
        batch = states.shape[0]
        dim = states.shape[2]
        device = states.device
        states = states.contiguous(memory_format=torch.channels_last)

        # make a loop
        for d in range(1, dim):
            prev_states = torch.cat([
                torch.zeros((batch, self.state_size, 1), device=device),
                states[:, :, :, d - 1],
                torch.zeros((batch, self.state_size, 1), device=device)
            ], dim=2).transpose(1, 2)

            # compute states
            states[:, :, :, d] = self.cell(
                prev_states=[prev_states[:, :-2, :].reshape(-1, self.state_size),
                             prev_states[:, 1:-1, :].reshape(-1, self.state_size),
                             prev_states[:, 2:, :].reshape(-1, self.state_size)],
                state_prior=states[:, :, :, d].transpose(1, 2).reshape(-1, self.state_size).clone(memory_format=torch.preserve_format)
            ).reshape(batch, -1, self.state_size).transpose(1, 2)
        # return new states
        return states.contiguous(memory_format=torch.contiguous_format)

    def forward1(self, states):
        # zero state
        batch = states.shape[0]
        dim = states.shape[2]
        device = states.device
        states = states.contiguous(memory_format=torch.channels_last)
        # make a loop
        for d in range(dim - 2, -1, -1):
            prev_states = torch.cat([
                torch.zeros((batch, self.state_size, 1), device=device),
                states[:, :, :, d + 1],
                torch.zeros((batch, self.state_size, 1), device=device)
            ], dim=2).transpose(1, 2)
            # compute states
            states[:, :, :, d] = self.cell(
                prev_states=[prev_states[:, :-2, :].reshape(-1, self.state_size),
                             prev_states[:, 1:-1, :].reshape(-1, self.state_size),
                             prev_states[:, 2:, :].reshape(-1, self.state_size)],
                state_prior=states[:, :, :, d].transpose(1, 2).reshape(-1, self.state_size).clone(memory_format=torch.preserve_format)
            ).reshape(batch, -1, self.state_size).transpose(1, 2)
        # return new states
        return states.contiguous(memory_format=torch.contiguous_format)

    def forward2(self, states):
        # zero state
        batch = states.shape[0]
        dim = states.shape[2]
        device = states.device
        states = states.contiguous(memory_format=torch.channels_last)
        # make a loop
        for d in range(1, dim):
            prev_states = torch.cat([
                torch.zeros((batch, self.state_size, 1), device=device),
                states[:, :, d - 1, :],
                torch.zeros((batch, self.state_size, 1), device=device)
            ], dim=2).transpose(1, 2)
            # compute states
            states[:, :, d, :] = self.cell(
                prev_states=[prev_states[:, :-2, :].reshape(-1, self.state_size),
                             prev_states[:, 1:-1, :].reshape(-1, self.state_size),
                             prev_states[:, 2:, :].reshape(-1, self.state_size)],
                state_prior=states[:, :, d, :].transpose(1, 2).reshape(-1, self.state_size).clone(memory_format=torch.preserve_format)
            ).reshape(batch, -1, self.state_size).transpose(1, 2)
        # return new states
        return states.contiguous(memory_format=torch.contiguous_format)

    def forward3(self, states):
        # zero state
        batch = states.shape[0]
        dim = states.shape[2]
        device = states.device
        states = states.contiguous(memory_format=torch.channels_last)
        # make a loop
        for d in range(dim - 2, -1, -1):
            prev_states = torch.cat([
                torch.zeros((batch, self.state_size, 1), device=device),
                states[:, :, d + 1, :],
                torch.zeros((batch, self.state_size, 1), device=device)
            ], dim=2).transpose(1, 2)
            # compute states
            states[:, :, d, :] = self.cell(
                prev_states=[prev_states[:, :-2, :].reshape(-1, self.state_size),
                             prev_states[:, 1:-1, :].reshape(-1, self.state_size),
                             prev_states[:, 2:, :].reshape(-1, self.state_size)],
                state_prior=states[:, :, d, :].transpose(1, 2).reshape(-1, self.state_size).clone(memory_format=torch.preserve_format)
            ).reshape(batch, -1, self.state_size).transpose(1, 2)
        # return new states
        return states.contiguous(memory_format=torch.contiguous_format)


class SDN(nn.Module):
    def __init__(self, in_ch, out_ch, state_size, dirs, kernel_size, stride, padding, upsample=False):
        super().__init__()
        # project-in network
        cnn_module = nn.ConvTranspose2d if upsample else nn.Conv2d
        self.pre_cnn = cnn_module(in_ch, state_size, kernel_size, stride, padding)
        # update network
        sdn_update_blocks = []
        for dir in dirs:
            sdn_update_blocks.append(_SDNLayer(state_size, dir=dir))
        self.sdn_update_network = nn.Sequential(*sdn_update_blocks)
        # project-out network
        self.post_cnn = nn.Conv2d(state_size, out_ch, 1)

    def forward(self, x):
        # (I) project-in step
        x = self.pre_cnn(x)
        x = nn.functional.tanh(x)
        # (II) update step
        x = self.sdn_update_network(x)
        # (III) project-out step
        x = self.post_cnn(x)
        return x

class ResSDN(nn.Module):

    def __init__(self, in_ch, out_ch, state_size, dirs, kernel_size, stride, padding, upsample=False):
        super().__init__()
        self.sdn = SDN(in_ch, 2 * out_ch, state_size, dirs, kernel_size, stride, padding, upsample)
        cnn_module = nn.ConvTranspose2d if upsample else nn.Conv2d
        self.cnn = cnn_module(in_ch, out_ch, kernel_size, stride, padding)

    def forward(self, input):
        cnn_out = self.cnn(input)
        sdn_out, gate = self.sdn(input).chunk(2, 1)
        gate = torch.sigmoid(gate)
        return gate * cnn_out + (1-gate) * sdn_out
