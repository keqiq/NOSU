import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence

class OsuDataset(Dataset):
    def __init__(self, input_sequences, target_sequences):
        self.input_sequences = input_sequences
        self.target_sequences = target_sequences
        
    def __len__(self):
        # Return the total number of sequences
        return len(self.input_sequences)
    
    def __getitem__(self, idx):
        # Fetch the input and target sequence at index `idx`
        input_seq = self.input_sequences[idx]
        target_seq = self.target_sequences[idx]
        return input_seq, target_seq
    
# def collate_fn(batch):
#     inputs, targets = zip(*batch)
    
#     # Padding
#     input_lengths = torch.tensor([len(seq) for seq in inputs])
#     padded_inputs = pad_sequence(inputs, batch_first=True, padding_value=0)
    
#     # target_lengths = torch.tensor([len(seq) for seq in targets])
#     # padded_targets = pad_sequence(targets, batch_first=True, padding_value=0)
    
#     targets = torch.stack(targets)
    
#     # Packing
#     packed_inputs = pack_padded_sequence(padded_inputs, input_lengths, batch_first=True, enforce_sorted=False)
    
#     return packed_inputs, targets, input_lengths

def collate_fn(batch):
    inputs, targets = zip(*batch)
    
    # Padding
    input_lengths = torch.tensor([len(seq) for seq in inputs])
    padded_inputs = pad_sequence(inputs, batch_first=True, padding_value=0)
    
    target_lengths = torch.tensor([len(seq) for seq in targets])
    padded_targets = pad_sequence(targets, batch_first=True, padding_value=0)
    
    # Packing
    packed_inputs = pack_padded_sequence(padded_inputs, input_lengths, batch_first=True, enforce_sorted=False)
    
    return packed_inputs, padded_targets, input_lengths, target_lengths