import torch
import torch.nn as nn
import torch.nn.functional as F
from common.quaternion import qmul
import numpy as np


#Encoder Netwrok
class QuaterNet_Encoder(nn.Module):
    def __init__(self, num_joints, h_size, num_outputs=0):
        """
        Construct a QuaterNet neural network.
        Arguments:
         -- num_joints: number of skeleton joints.
         -- num_outputs: extra inputs/outputs (e.g. translations), in addition to joint rotations.
        """
        super().__init__()

        self.num_joints = num_joints
        self.num_outputs = num_outputs
        
        self.rnn = nn.GRU(input_size=num_joints*4 + num_outputs, hidden_size=h_size, num_layers=1, batch_first=True)
        self.h0 = nn.Parameter(torch.zeros(self.rnn.num_layers, 1, h_size).normal_(std=0.01), requires_grad=True)
        
        self.fc = nn.Linear(h_size, num_joints*4 + num_outputs)
    
    def forward(self, x, h=None):
        """
        Run a forward pass of this model.
        Arguments:
         -- x: input tensor of shape (N, L, J*4 + O + C), where N is the batch size, L is the sequence length,
               J is the number of joints, O is the number of outputs, and C is the number of controls.
               Features must be provided in the order J, O, C.
         -- h: hidden state. If None, it defaults to the learned initial state.
        """
        assert len(x.shape) == 3
        assert x.shape[-1] == self.num_joints*4 + self.num_outputs
        
        if h is None:
            h = self.h0.expand(-1, x.shape[0], -1).contiguous()
        x, h = self.rnn(x, h)

        return h


# Decoder Network
class QuaterNet_Decoder(nn.Module):
    def __init__(self, h_size, num_joints, num_outputs=0, model_velocities=False):
        """
        Construct a QuaterNet neural network.
        Arguments:
         -- num_joints: number of skeleton joints.
         -- num_outputs: extra inputs/outputs (e.g. translations), in addition to joint rotations.
        """
        super().__init__()

        self.num_joints = num_joints
        self.num_outputs = num_outputs
        
        self.rnn = nn.GRU(input_size=num_joints*4 + num_outputs, hidden_size=h_size, num_layers=1, batch_first=True)
        self.h0 = nn.Parameter(torch.zeros(self.rnn.num_layers, 1, h_size).normal_(std=0.01), requires_grad=True)
        
        self.fc = nn.Linear(h_size, num_joints*4 + num_outputs)
        self.model_velocities = model_velocities
    
    def forward(self, x, h, return_prenorm=False):
        """
        Run a forward pass of this model.
        Arguments:
         -- x: input tensor of shape (N, L, J*4 + O + C), where N is the batch size, L is the sequence length,
               J is the number of joints, O is the number of outputs, and C is the number of controls.
               Features must be provided in the order J, O, C.
         -- h: hidden state. If None, it defaults to the learned initial state.
         -- return_prenorm: if True, return the quaternions prior to normalization.
        """
        # print(x.shape)
        assert len(x.shape) == 3
        assert x.shape[-1] == self.num_joints*4 + self.num_outputs
        
        x_orig = x
        
        x, h = self.rnn(x, h)
        
        x = self.fc(x)
        
        pre_normalized = x[:, :, :self.num_joints*4].contiguous()
        normalized = pre_normalized.view(-1, 4)
        if self.model_velocities:
            normalized = qmul(normalized, x_orig[:, :, :self.num_joints*4].contiguous().view(-1, 4))
        normalized = F.normalize(normalized, dim=1).view(pre_normalized.shape)
        
        if self.num_outputs > 0:
            x = torch.cat((normalized, x[:, :, self.num_joints*4:]), dim=2)
        else:
            x = normalized
        
        if return_prenorm:
            return x, h, pre_normalized
        else:
            return x, h

# seq2seq class
class seq2seq(nn.Module):
    def __init__(self, prefix_length, num_joints, num_outputs, model_velocities):
        super().__init__()

        self.translations_size = num_outputs
        self.model_velocities = model_velocities
        
        h_size = 1000 # size of hidden layer
        self.encoder = QuaterNet_Encoder(num_joints, 1000, num_outputs)
        self.decoder = QuaterNet_Decoder(1000, num_joints, num_outputs, model_velocities)
        
        self.prefix_length = prefix_length  # 50 is the prefix length

    def forward(self, inputs, target_length, teacher_forcing_ratio):

        terms = []
        predictions = []
        #print("HIII",self.prefix_length, inputs.shape, target_length)
        # Initialize with prefix
        hidden = self.encoder.forward(inputs[:, :self.prefix_length])

        decoder_input = inputs[:,self.prefix_length-1:self.prefix_length] #last element of the input sequence
        predicted, hidden, term = self.decoder.forward(decoder_input, hidden, True)
        predictions.append(predicted)

        tf_mask = np.random.uniform(size=target_length-1) < teacher_forcing_ratio
        i = 0
        while i < target_length - 1:
            contiguous_frames = 1
            # Batch together consecutive "teacher forcings" to improve performance
            if tf_mask[i]:
                while i + contiguous_frames < target_length - 1 and tf_mask[i + contiguous_frames]:
                    contiguous_frames += 1
                # Feed ground truth
                predicted, hidden, term = self.decoder.forward(inputs[:, self.prefix_length+i:self.prefix_length+i+contiguous_frames],
                                                        hidden, True)
            else:
                # Feed own output
                predicted, hidden, term = self.decoder.forward(predicted, hidden, True)
            terms.append(term)
            predictions.append(predicted)
            if contiguous_frames > 1:
                predicted = predicted[:, -1:]
            i += contiguous_frames

        return predictions, terms