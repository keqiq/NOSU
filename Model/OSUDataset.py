import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence

class OsuDataset(Dataset):
    def __init__(self, input_sequences, target_sequences, target_objects=None):
        self.input_sequences = input_sequences
        self.target_sequences = target_sequences
        self.target_objects = target_objects
        
    def __len__(self):
        # Return the total number of sequences
        return len(self.input_sequences)
    
    def __getitem__(self, idx):
        # Fetch the input and target sequence at index `idx`
        # Include target_obj during training, None otherwise
        input_seq = self.input_sequences[idx]
        target_seq = self.target_sequences[idx]
        
        if self.target_objects is not None:
            target_obj = self.target_objects[idx]
        else:
            target_obj = None
        
        return input_seq, target_seq, target_obj
    
def pos_collate_fn(batch):
    inputs, targets, objects = zip(*batch)
    
    # Padding
    input_lengths = torch.tensor([len(seq) for seq in inputs])
    padded_inputs = pad_sequence(inputs, batch_first=True, padding_value=0)
    
    # Packing
    packed_inputs = pack_padded_sequence(padded_inputs, input_lengths, batch_first=True, enforce_sorted=False)
    
    # Stacking tensors
    targets = torch.stack(targets)
    
    if objects[0] is not None:
        objects = torch.stack(objects)
    else: 
        objects = None
    
    return packed_inputs, targets, input_lengths, objects

def key_collate_fn(batch):
    inputs, targets, _ = zip(*batch)
    
    # Padding
    input_lengths = torch.tensor([len(seq) for seq in inputs])
    padded_inputs = pad_sequence(inputs, batch_first=True, padding_value=0)
    
    target_lengths = torch.tensor([len(seq) for seq in targets])
    padded_targets = pad_sequence(targets, batch_first=True, padding_value=0)
    
    # Packing
    packed_inputs = pack_padded_sequence(padded_inputs, input_lengths, batch_first=True, enforce_sorted=False)
    
    return packed_inputs, padded_targets, input_lengths, target_lengths