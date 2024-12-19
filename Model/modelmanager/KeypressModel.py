import torch
from .ModelManager import ModelManager
from ..OSUModel import KeypressEncoder, KeypressDecoder, OSUModelKey
from tqdm import tqdm
from torch.nn.utils.rnn import pack_padded_sequence, pad_sequence

class KeypressModel(ModelManager):
    def __init__(self, name, config):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        encoder = KeypressEncoder(config['key_input_size'], config['key_hidden_size'], config['key_num_layers'])
        decoder = KeypressDecoder(config['key_hidden_size'], config['key_num_layers'])
        model = OSUModelKey(encoder, decoder, self.device)
        super().__init__(name, config, model)
        
        self.criterion = torch.nn.CrossEntropyLoss(reduction='none')
        
    
    def _calculate_losses(self, outputs, data):
        # Keypress targets (batch_size, sequence_len - 1, num_keys)
        targets = data['targets'][:, 1:, :]
        outputs = outputs[:, 1:, :]
        
        # Creating mask for variable length sequences
        max_len = outputs.size(1)
        
        # True when value is less than target length
        mask = (torch.arange(max_len).unsqueeze(0)).to(self.device) < data['length'].unsqueeze(1) # Shape (batch_size, max_len)
        mask = mask.to(self.device)
        
        # Keypress cross entropy loss
        target_idx = torch.argmax(targets, dim=2)
        loss = self.criterion(outputs.permute(0, 2, 1), target_idx)
        loss = (loss * mask).sum() / mask.sum()
        
        return [loss]
    
    def _print_progress(self, train_losses, valid_losses, curr_lr, epoch):
        print(
            f"Epoch [{epoch+1}/{self.max_epoch}] "
            f"LR [{curr_lr:.0e}/{self.min_lr:.0e}] "
            f"T Loss Replay: {train_losses[0]:.4f} | "
            f"V Loss Replay: {valid_losses[0]:.4f}"
        )
        
    def predict(self, inputs):
        inputs = [t.pin_memory() for t in inputs]
        predictions = []
        with torch.no_grad():
            # initialize the first keypress
            # Keypress will be key1 (1.0, 0.0)
            prev_pred = torch.tensor([1.0, 0.0], device=self.device).unsqueeze(0).unsqueeze(1)
            
            for input in tqdm(inputs, total=len(inputs), desc="Predicting Keypress"):
                input_length = torch.tensor([len(input)], dtype=torch.long)
                input = input.to(self.device).unsqueeze(0)
                padded_input = pad_sequence(input, batch_first=True, padding_value=0)
                packed_input = pack_padded_sequence(padded_input, input_length, batch_first=True, enforce_sorted=False)

                output = self.model(packed_input, prev_pred)
                
                prev_pred = output
                
                key_index = torch.argmax(output, 2)
                predictions.append(key_index)
                
        return predictions
        
        