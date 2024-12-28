import torch
import time
""" 
Super class of PositionModel and KeypressModel
Classes for handling training and inference
"""
class ModelManager():
    def __init__(self, name, config, model, device, weights=None):
        self.device = device
        self.model = model.to(self.device)
        self.name = name
        if weights is None:
            self.min_lr = config['early_stopping_learning_rate']
            self.max_epoch = config['max_epoch']
            
            self.optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr = config['learning_rate'],
                weight_decay = config['weight_decay']
            )
            
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=0.1,
                patience=config['patience'],
                threshold=1e-4
            )
            
            self.hyperparameters = {
                "input_size": self.input_size,
                "hidden_size": self.hidden_size,
                "num_layers": self.num_layers,
                "context_size": self.context_size,
                "time_window": self.time_window
            }
        else:
            self.model.load_state_dict(weights)
            
    """
    Function to unpack batch from dataloader
    """
    def _unpack(self, batch):
        data = {
            "inputs":   batch[0].to(self.device),
            "targets":  batch[1].to(self.device),
            "length":   batch[2].to(self.device),
            "object":   batch[3].to(self.device) if self.__class__.__name__ == "PositionModel" else None
        }
        return data
        
    """
    Training and validation loop
    """
    def train(self, train_loader, valid_loaders):
        total_train_len = len(train_loader)
        total_valid_len = 0
        for valid_loader in valid_loaders:
            total_valid_len += len(valid_loader)
        
        epoch = 0
        start_time = time.time()
        while True if self.max_epoch is None else epoch < self.max_epoch:
            self.model.train()
            teacher_forcing_ratio = max(1 - (epoch / 20), 0)
            train_losses = []
            
            # Training loop
            for batch in train_loader:                
                train_data = self._unpack(batch)
                    
                self.optimizer.zero_grad()
                
                # Forward pass
                train_outputs = self.model(train_data['inputs'], train_data['targets'], teacher_forcing_ratio)
                
                # Computing training losses
                output_losses_train = self._calculate_losses(train_outputs, train_data)
                
                # Initializing train_losses dynamically if empty
                if not train_losses:
                    train_losses = [0.0] * len(output_losses_train)
                
                # Accumulating training losses
                for idx, loss in enumerate(output_losses_train):
                    train_losses[idx] += loss.item()
                
                # Backwards pass
                combined_loss = sum(output_losses_train)
                combined_loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                # Updating weights
                self.optimizer.step()
                
            train_losses = [loss / total_train_len for loss in train_losses]
                
            # Validation loop
            self.model.eval()
            valid_losses = []
            
            with torch.no_grad():
                for valid_loader in valid_loaders:
                    
                    for batch in valid_loader:
                        valid_data = self._unpack(batch)
                        
                        # Forward pass with no gradient calculation
                        valid_outputs = self.model(valid_data['inputs'], valid_data['targets'])
                        
                        # Computing validation losses
                        output_losses_valid = self._calculate_losses(valid_outputs, valid_data)
                        
                        # Initialize valid losses dynamically if empty
                        if not valid_losses:
                            valid_losses = [0.0] * len(output_losses_valid)
                        
                        # Acumulating validation losses
                        for idx, loss in enumerate(output_losses_valid):
                            valid_losses[idx] += loss.item()
                            
            valid_losses = [loss / total_valid_len for loss in valid_losses]
            
            self.scheduler.step(valid_losses[0])
            
            curr_lr = self.optimizer.param_groups[0]['lr']
            
            if curr_lr <= self.min_lr:
                print("Reached early stopping condition")
                break
            
            self._print_progress(train_losses, valid_losses, curr_lr, epoch)
            
            epoch += 1
            
            if epoch >= self.max_epoch:
                print("Reached max epoch")
        
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Training completed in {elapsed_time:.2f} seconds")
    
    """
    Function to save model name, hyperparameters and weights as .pth file
    """
    def save_model(self, path):
        torch.save({
            "model_name": self.name,
            "hyperparameters": self.hyperparameters,
            "model_weights": self.model.state_dict()
        }, f'{path}/{self.name}.pth')
        
        print(f"Saved {self.name} to {path}/{self.name}.pth")