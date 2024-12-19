import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
import os

class OSUDataset(Dataset):
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
        
        if self.target_objects:
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

class OSUDataloader():
    def __init__(self, data_parser, config, regenerate=False):
        self.data_parser = data_parser
        self.type = type(data_parser).__name__
        self.t_b_size = config['train_batch_size']
        self.v_b_size = config['valid_batch_size']
        self.regenerate = regenerate
        
    def get_loaders(self, paths):
        data = None
        if self.type == 'PositionData':
            collate_fn = pos_collate_fn
        else:
            collate_fn = key_collate_fn
            
        if os.path.exists(f'{paths['train']}/train_dataset.pth') and not self.regenerate:
            print("Creating training dataloader...", end="")
            train_dataset = torch.load(f'{paths['train']}/train_dataset.pth')
        else:
            data = self.data_parser.generate()
            train_data = data['train']
            print("Creating training dataloader...", end="")
            train_dataset = OSUDataset(train_data[0], train_data[1], train_data[2])
            torch.save(train_dataset, f'{paths['train']}/train_dataset.pth')
        
        train_loader = DataLoader(train_dataset, 
                                  batch_size=self.t_b_size, 
                                  shuffle=True, 
                                  drop_last=True,
                                  collate_fn=collate_fn,
                                  pin_memory=True)
        
        print("Complete!")
        valid_loaders = []
        valid_folders = os.listdir(paths['valid'])
        valid_data_size = sum(os.path.isdir(os.path.join(paths['valid'], folder)) for folder in valid_folders)
        print("Creating validation dataloaders...", end="")
        for idx in range(valid_data_size):
            if os.path.exists(f'{paths['valid']}/valid_dataset_{idx}.pth') and not self.regenerate:
                valid_dataset = torch.load(f'{paths['valid']}/valid_dataset_{idx}.pth')
            else:
                if data is not None:
                    valid_data = data['valid'][idx]
                    valid_dataset = OSUDataset(valid_data[0], valid_data[1], valid_data[2])
                    torch.save(valid_dataset, f'{paths['valid']}/valid_dataset_{idx}.pth')
            
            valid_loader = DataLoader(valid_dataset,
                                      batch_size=self.v_b_size,
                                      shuffle=False,
                                      drop_last=True,
                                      collate_fn=collate_fn)
            valid_loaders.append(valid_loader)
        print("Complete!")
        return train_loader, valid_loaders