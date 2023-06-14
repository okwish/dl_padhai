## Encoder Decoder Model
# Encoder - GRU
# Decoder -GRU

import torch
import torch.nn as nn



class EncoderDecoder(nn.Module):

    MAX_OUTPUT_CHARS = 30
    
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        
        # there is a state vector in encoder and one in decoder.
        # here we are taking both to be of same dim.(need not be so.)
        
        self.hidden_size = hidden_size # same for encoder and decoder
        self.output_size = output_size
        
        # BLOCKS:
        self.encoder_rnn_cell = nn.GRU(input_size, hidden_size)
        self.decoder_rnn_cell = nn.GRU(output_size, hidden_size)
        # input to decoder RNN is yi(ouput of prev RNN cell itself)
        # thus input dim is output dim.
        
        self.h2o = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=2)
        
        
    def forward(self, input_, max_output_chars = MAX_OUTPUT_CHARS, device = 'cpu', ground_truth = None):
        
        # ENCODER:
        out, hidden = self.encoder_rnn_cell(input_)
        # one go - input as a tensor.
        # internally it happens in sequence- step by step.
        
        # output is a tensor with all the outputs(output at each step)
        # hidden is a single vector with last hidden state. it gets over-written in the steps.
        
        # DECODER:
        decoder_state = hidden # S_0 = h_T
        # here encoder and decoder state size is same - therefore above can be done direclty.
        # else some linear layer in between to do the transposition
        decoder_input = torch.zeros(1, 1, self.output_size).to(device) # y_0 <sos> -ish.
        # input to decoder same dimension as output.
        outputs = []       
        
        # not invoking decoder in a single call (like the encoder.)
        # but how does the program know if we are looping or not(??)
        # BASED ON THE INPUT
        # INTERNAL IMPLEMENTATION : LOOP OVER INPUT 'SEQUENCE'
        
        # LOOP:
        
        for i in range(max_output_chars):
            
            # y_1, s_1 from y_0, s_0
            out, decoder_state = self.decoder_rnn_cell(decoder_input, decoder_state)            
           
            # output from state
            out = self.h2o(decoder_state)
            out = self.softmax(out)
            
            outputs.append(out.view(1, -1)) # list of outputs - this is returned.       
            
            max_idx = torch.argmax(out, 2, keepdim=True) # index of max
            
            # if ground truth - ie, if its mentioned pass ground truth as input to next step
            # then one hot with the ground truth index and pass that.
            # else one hot with max-index
            if not ground_truth is None:
                max_idx = ground_truth[i].reshape(1, 1, 1)
                
            # make one hot vector out of the index
            one_hot = torch.FloatTensor(out.shape).to(device)
            one_hot.zero_()
            one_hot.scatter_(2, max_idx, 1)
            
            # any function .. in the computation graph.
            # back prop through all.
            # sometimes we don't want BP through some.
            # eg - ouput of one step passes as input of next step. - this is also a link
            # but we don't want to BP through that.
            decoder_input = one_hot.detach()
            # detach() -> don't pass anymore grad through this tensor.
            # ie, not part of the computational graph for gradient.

            # detach() will remove the computational graph maping in a tensor      
            
        return outputs