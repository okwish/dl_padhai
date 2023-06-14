## Encoder Attend Decoder Model
# Encoder - GRU
# Decoder -GRU with attention

import torch
import torch.nn as nn
import torch.nn.functional as F 



class EncoderAttendDecoder(nn.Module):

    MAX_OUTPUT_CHARS = 30
    
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        self.encoder_rnn_cell = nn.GRU(input_size, hidden_size)
        
        # this is like option 2 of encoder decoder. concatenating encoding at each step
        # attention - concatenating 'refined encoding' at each step
        # therefore input to GRU cell - twice the size.
        self.decoder_rnn_cell = nn.GRU(hidden_size*2, hidden_size)
        
        self.h2o = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=2)
        
        # additional layers.
        # Ws, Uh in attention 'function' - as linear layers.
        # similarly Vatt.
        self.U = nn.Linear(self.hidden_size, self.hidden_size) # Uatt
        self.W = nn.Linear(self.hidden_size, self.hidden_size) # Watt
        self.attn = nn.Linear(self.hidden_size, 1) # Vatt
        
        # decoder output to state (to match dimension.)  (??)
        self.out2hidden = nn.Linear(self.output_size, self.hidden_size)   
        
        
    def forward(self, input, max_output_chars = MAX_OUTPUT_CHARS, device = 'cpu', ground_truth = None):
        
        # ENCODER: 
        
        # get only one hidden state. - but tensor of ouputs
        # we need all states for attention
        # instead of states - use the 'ouputs' of encoder. - this is a choice we make.
        # as all of them are available as a tensor.
        encoder_outputs, hidden = self.encoder_rnn_cell(input)
        encoder_outputs = encoder_outputs.view(-1, self.hidden_size)
        
        # DECODER : 
        
        decoder_state = hidden # S_0 = h_T
        decoder_input = torch.zeros(1, 1, self.output_size).to(device) # first decoder unit input
        
        outputs = []
        
        U = self.U(encoder_outputs) 
        # function that takes encoder outputs(which we are using instead of encoder states)
        # and pass through the U layer ..
        ## because that part in the attention function - not chainging in decoder steps.
        # h_i are fixed and each multiplied by U - so do that(once) and keep it.. to be used in decoder stage.
       
        
        # going step by step (loop) in decoder part.
        
        for i in range(max_output_chars):
            
            # get the refined encoding.
            
            W = self.W(decoder_state.view(1, -1).repeat(encoder_outputs.shape[0], 1)) # W * S_t-1
            # repeat - tile.. to make same size.
            
            V = self.attn(torch.tanh(U + W)) # e_tj
            
            attn_weights = F.softmax(V.view(1, -1), dim = 1)  # alpha_tj
            
            # weighted combination of encoder ouputs(instead of states, here), with alphas
            attn_applied = torch.bmm(attn_weights.unsqueeze(0),
                                 encoder_outputs.unsqueeze(0))
            
            
            # decoder input- concatenation of 'refined encoding' just found, input(decoder_ouput of prev unit)
            
            # not decoder input direcltly, but after trasnforming to same size.
            embedding = self.out2hidden(decoder_input) # linear layer on decoder input 
            # to make it the same size as the refined encoding.. so that same contribution in terms of size..
            # that layer for this - parameters - also learned.
            
            decoder_input = torch.cat((embedding[0], attn_applied[0]), 1).unsqueeze(0)         
            
            # here is where we are finding the decoder unit output 
            # first we find the decoder input - concatening decoder_ouput(transformed) and refined_encoding.
            # done above.
            # then pass that input to the decoder cell.
            out, decoder_state = self.decoder_rnn_cell(decoder_input, decoder_state)
                
            out = self.h2o(decoder_state)
            out = self.softmax(out)
            outputs.append(out.view(1, -1))

            max_idx = torch.argmax(out, 2, keepdim=True)
            if not ground_truth is None:
                max_idx = ground_truth[i].reshape(1, 1, 1)
            one_hot = torch.zeros(out.shape, device=device)
            one_hot.scatter_(2, max_idx, 1) 
            
            decoder_input = one_hot.detach()
            
        return outputs