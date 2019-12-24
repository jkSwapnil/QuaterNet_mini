# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from common.quaternet import seq2seq, QuaterNet_Decoder, QuaterNet_Encoder
from common.quaternion import qeuler
from time import time
import matplotlib.pyplot as plt


class PoseNetwork:
    def __init__(self, prefix_length, num_joints, num_outputs, num_controls, model_velocities):
        self.translations_size = num_outputs
        self.controls_size = num_controls
        self.model_velocities = model_velocities

        self.model = seq2seq(prefix_length, num_joints, num_outputs, model_velocities)
        self.use_cuda = False
        
        dec_params = 0
        for parameter in self.model.parameters():
            dec_params += parameter.numel()
        print('# parameters:', dec_params)

        self.prefix_length = prefix_length
        
    def cuda(self):
        self.use_cuda = True
        self.model.cuda()
        return self
    
    def eval(self):
        self.model.eval()
        return self
            
    def _prepare_next_batch_impl(self, batch_size, dataset, target_length, sequences):
        # This method must be implemented by the subclass
        pass
    
    def _loss_impl(self, predicted, expected):
        # This method must be implemented by the subclass
        pass
    
    def train(self, dataset, target_length, sequences_train, sequences_valid, batch_size, n_epochs=3000, rot_reg=0.01):
        # np.random.seed(1234)
        # self.model.train()
        
        lr = 0.001
        batch_size_valid = 30
        lr_decay = 0.999
        teacher_forcing_ratio = 1 # Start by forcing the ground truth
        tf_decay = 0.995 # Teacher forcing decay rate
        gradient_clip = 0.1
        
        optimizer = optim.Adam(self.model.parameters(), lr=lr)

        if len(sequences_valid) > 0:
            batch_in_valid, batch_out_valid = next(self._prepare_next_batch_impl(
                batch_size_valid, dataset, target_length, sequences_valid))
            inputs_valid = torch.from_numpy(batch_in_valid)
            outputs_valid = torch.from_numpy(batch_out_valid)
            if self.use_cuda:
                inputs_valid = inputs_valid.cuda()
                outputs_valid = outputs_valid.cuda()
        
        losses = []
        valid_losses = []
        gradient_norms = []
        print('Training for %d epochs' % (n_epochs))
        start_time = time()
        start_epoch = 0
        try:
            for epoch in range(n_epochs):
                batch_loss = 0.0
                N = 0
                for batch_in, batch_out in self._prepare_next_batch_impl(batch_size, dataset, target_length, sequences_train):
                    # Pick a random chunk from each sequence
                    inputs = torch.from_numpy(batch_in)
                    outputs = torch.from_numpy(batch_out)
                    
                    if self.use_cuda:
                        inputs = inputs.cuda()
                        outputs = outputs.cuda()

                    optimizer.zero_grad()

                    terms = []
                    predictions = []

                    predictions, terms = self.model.forward(inputs, target_length, teacher_forcing_ratio)

                    terms = torch.cat(terms, dim=1)
                    terms = terms.view(terms.shape[0], terms.shape[1], -1, 4)
                    penalty_loss = rot_reg * torch.mean((torch.sum(terms**2, dim=3) - 1)**2)

                    predictions = torch.cat(predictions, dim=1)
                    loss = self._loss_impl(predictions, outputs)

                    loss_total = penalty_loss + loss
                    loss_total.backward()
                    nn.utils.clip_grad_norm_(self.model.parameters(), gradient_clip)
                    optimizer.step()

                    # Compute statistics
                    batch_loss += loss.item() * inputs.shape[0]
                    N += inputs.shape[0]

                batch_loss = batch_loss / N
                losses.append(batch_loss)

                # Run validation
                if len(sequences_valid) > 0:
                    with torch.no_grad():
                        predictions = []
                        predictions, terms = self.model.forward(inputs_valid, target_length, 0) #turn off teacher forcing
                        predictions = torch.cat(predictions, dim=1)
                        loss = self._loss_impl(predictions, outputs_valid)
                        valid_loss = loss.item()
                        valid_losses.append(valid_loss)
                        print('[%d] loss: %.5f valid_loss %.5f lr %f tf_ratio %f' % (epoch + 1, batch_loss, valid_loss,
                                                                  lr, teacher_forcing_ratio))
                else:
                    print('[%d] loss: %.5f lr %f tf_ratio %f' % (epoch + 1, batch_loss,
                                                              lr, teacher_forcing_ratio))
                teacher_forcing_ratio *= tf_decay
                lr *= lr_decay
                for param_group in optimizer.param_groups:
                    param_group['lr'] *= lr_decay

                if epoch > 0 and (epoch+1) % 10 == 0:
                    next_time = time()
                    time_per_epoch = (next_time - start_time)/(epoch - start_epoch)
                    print('Benchmark:', time_per_epoch, 's per epoch')
                    start_time = next_time
                    start_epoch = epoch
        except KeyboardInterrupt:
            print('Training aborted.')
            
        print('Done.')
        #print('gradient_norms =', gradient_norms)
        #print('losses =', losses)
        #print('valid_losses =', valid_losses)
        print('Saving the plots of training and validation losses')
        #plotting for training and validation loss
        plt.plot(list(range(1,len(losses)+1)), losses)
        plt.plot(list(range(1,len(valid_losses)+1)), valid_losses)
        plt.xlabel("Epochs")
        plt.ylabel("Loss Values")
        plt.title("Validation and Training Loss vs Epochs")
        plt.legend(("Training loss", "Validation loss"))
        plt.savefig("loss.png")
        return losses, valid_losses, gradient_norms

        
    def save_weights(self, model_file):
        print('Saving weights to', model_file)
        torch.save(self.model.state_dict(), model_file)
        
    def load_weights(self, model_file):
        print('Loading weights from', model_file)
        self.model.load_state_dict(torch.load(model_file, map_location=lambda storage, loc: storage))
        
