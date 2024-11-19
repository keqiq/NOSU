import torch
import torch.nn as nn
import torch.nn.functional as F

class PositionEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(PositionEncoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # LSTM Encoder
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
    
    def forward(self, x):
        # Initializing hidden states with 0
        batch_size = x.batch_sizes[0]
        h_0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.data.device)  # hidden state
        c_0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.data.device)  # cell state
        
        # Forward propagate through LSTM
        _, (h_n, c_n) = self.lstm(x, (h_0, c_0))
        
        # h_n: hidden state for each layer, c_n: cell state for each layer
        return (h_n, c_n)
    
class PositionDecoder(nn.Module):
    def __init__(self, hidden_size, num_layers):
        super(PositionDecoder, self).__init__()
        
        # LSTM decoder
        self.lstm = nn.LSTM(2, hidden_size, num_layers, batch_first=True)  # Input size includes position and keypress

        # Regression position output (x, y) with hidden state from encoder
        self.pos = nn.Linear(hidden_size, 2)
        
        # self.dropout = nn.Dropout(p=0.2)
        
    def forward(self, x, hidden):
         # Forward pass through the LSTM
        output, hidden = self.lstm(x, hidden)
        
        # position regression output
        # output = self.dropout(output)
        pos_output = self.pos(output)
        
        return pos_output, hidden
    
class OSUModelPos(nn.Module):
    def __init__(self, pos_encoder, pos_decoder, device):
        super(OSUModelPos, self).__init__()
        self.pos_encoder = pos_encoder
        self.pos_decoder = pos_decoder
        self.device = device
    
    def forward(self, input, targets, teacher_forcing_ratio = 0):
        
        # Encode the source sequence
        pos_hidden = self.pos_encoder(input)
        
        # Initial input for position decoder (batch_size, 1, 2)
        pos_decoder_input = targets[:, 0, :].unsqueeze(1)
        
        pos_output, pos_hidden = self.pos_decoder(pos_decoder_input, pos_hidden)
        
        return pos_output
    
class KeypressEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(KeypressEncoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # LSTM Encoder
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
    
    def forward(self, x):
        # Initializing hidden states with 0
        batch_size = x.batch_sizes[0]
        h_0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.data.device)  # hidden state
        c_0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.data.device)  # cell state
        
        # Forward propagate through LSTM
        _, (h_n, c_n) = self.lstm(x, (h_0, c_0))
        
        # h_n: hidden state for each layer, c_n: cell state for each layer
        return (h_n, c_n)
        
class KeypressDecoder(nn.Module):
    def __init__(self, hidden_size, num_layers, num_keys=2):
        super(KeypressDecoder, self).__init__()
        
        # LSTM decoder
        self.lstm = nn.LSTM(num_keys, hidden_size, num_layers, batch_first=True)
        
        # Classification key press logit output (0, 1, 2, 11) with hidden state from encoder
        self.key = nn.Linear(hidden_size, num_keys)
        
        self.dropout = nn.Dropout(p=0.5)
    
    def forward(self, x, hidden):
        # Forward pass through the LSTM
        output, hidden = self.lstm(x, hidden)
        output = self.dropout(output)
        # Key press classification output
        key_output = self.key(output)
        
        return key_output, hidden
    
class OSUModelKey(nn.Module):
    def __init__(self, key_encoder, key_decoder, device):
        super(OSUModelKey, self).__init__()
        self.key_encoder = key_encoder
        self.key_decoder = key_decoder
        self.device = device
        
    def forward(self, input, targets, teacher_forcing_ratio=0):
        key_hidden = self.key_encoder(input)
        # Initial input for keypress decoder (batch_size, 1, num_keys)
        key_decoder_input = targets[:, 0, :].unsqueeze(1)
        
        key_outputs = []
        for t in range(1, targets.size(1)):
            key_pred, key_hidden = self.key_decoder(key_decoder_input, key_hidden)
            
            # Teacher forcing
            if torch.rand(1).item() < teacher_forcing_ratio:
                key_decoder_input = targets[:, t, :].unsqueeze(1)
            else:
                key_idx = torch.argmax(key_pred, dim=-1)
                key_decoder_input = F.one_hot(key_idx, num_classes=2).float()

            key_outputs.append(key_pred)
            
        key_outputs = torch.cat(key_outputs, dim=1)
        
        return key_outputs
    
    
    # class OSUModel(nn.Module):
#     def __init__(self, pos_encoder, key_encoder, pos_decoder, key_decoder, device):
#         super(OSUModel, self).__init__()
#         self.pos_encoder = pos_encoder
#         self.key_encoder = key_encoder
#         self.pos_decoder = pos_decoder
#         self.key_decoder = key_decoder
#         self.device = device
    
#     def forward(self, input, targets, teacher_forcing_ratio):
        
#         # Encode the source sequence
#         pos_hidden = self.pos_encoder(input)
#         key_hidden = self.key_encoder(input)
        
#         # Initial input for position decoder (batch_size, 1, 2)
#         pos_decoder_input = targets[:, 0, :2].unsqueeze(1)
        
#         # Initial input for keypress decoder (batch_size, 1, num_keys)
#         key_decoder_input = targets[:, 0, 2:].unsqueeze(1)
        
#         # Decode one step at a time
#         pos_outputs = []
#         key_outputs = []
#         for t in range(1, targets.size(1)):
            
#             pos_pred, pos_hidden = self.pos_decoder(pos_decoder_input, pos_hidden)
#             key_pred, key_hidden = self.key_decoder(key_decoder_input, key_hidden)
            
#             # Teacher forcing
#             if torch.rand(1).item() < teacher_forcing_ratio:
#                 pos_decoder_input = targets[:, t, :2].unsqueeze(1)
#                 key_decoder_input = targets[:, t, 2:].unsqueeze(1)
#             else:
#                 pos_decoder_input = pos_pred
#                 key_idx = torch.argmax(key_pred, dim=-1)
#                 key_decoder_input = F.one_hot(key_idx, num_classes=4).float()
            
#             pos_outputs.append(pos_pred)
#             key_outputs.append(key_pred)
                
#         pos_outputs = torch.cat(pos_outputs, dim=1)
#         key_outputs = torch.cat(key_outputs, dim=1)
        
#         return pos_outputs, key_outputs
