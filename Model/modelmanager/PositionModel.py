import torch
from .ModelManager import ModelManager
from ..OSUModel import PositionEncoder, PositionDecoder, OSUModelPos
from tqdm import tqdm
from torch.nn.utils.rnn import pack_padded_sequence

"""
PositionModel subclass provides loss function and predict function
"""
class PositionModel(ModelManager):
    def __init__(self, name, config, device, trained_model=None):
        if trained_model is None:
            encoder = PositionEncoder(config['pos_input_size'], config['pos_hidden_size'], config['pos_num_layers'])
            decoder = PositionDecoder(config['pos_hidden_size'], config['pos_num_layers'])
            self.input_size = config['pos_input_size']
            self.hidden_size = config['pos_hidden_size']
            self.num_layers = config['pos_num_layers']
            self.context_size = config['pos_context_size']
            self.time_window = config['pos_time_window']
            self.criterion = torch.nn.L1Loss(reduction='none')
            
            self.epsilon_min = config['epsilon_min']
            self.epsilon_max = config['epsilon_max']
            self.epsilon_expo = config['epsilon_expo']
            
            self.object_loss_weight = config['object_loss_weight']
            
            self.type_weight = torch.tensor([
                config['hit_circle_precision'],
                config['slider_head_precision'],
                config['slider_tick_precision'],
                config['spinner_start_precision'],
                config['spinner_tick_precision']
                ], dtype=torch.float32).to(device)
            
            self.buzz_epsilon_mulit = config['buzz_epsilon_multi']

            weights = None
        else:
            hyperparameters = trained_model['hyperparameters']
            encoder = PositionEncoder(hyperparameters['input_size'], hyperparameters['hidden_size'], hyperparameters['num_layers'])
            decoder = PositionDecoder(hyperparameters['hidden_size'], hyperparameters['num_layers'])
            weights = trained_model['model_weights']
            name = trained_model['model_name']
        
        model = OSUModelPos(encoder, decoder)
        
        super().__init__(name, config, model, device, weights)
        
    """
    Loss calculation of combined losses
    Replay loss is the loss between player replay data and model prediction
    Object loss is the loss between the object with smallest time delta and model prediction
    """
    def _calculate_losses(self, outputs, data):
        # Position targets (batch_size, 2, 2)
        # Removing previous positions from targets
        targets = data['targets'][:, 1:, :]              # Shape: (batch_size, 1, 2)
        
        # Position loss with absolute error
        replay_loss = self.criterion(outputs, targets)  # Shape: (batch_size, 1, 2)
        
        # Sum the loss along the last dimension and compute the mean over the batch
        replay_loss_mean = replay_loss.sum(dim=2).mean()
        
        # Immediate object is the object which has the smallest delta time value
        # This is used for object loss where the loss is recorded for predictions
        # outside of a distance threshold when the delta time to that object is small
        immediate_object = data['object']               # Shape: (batch_size, 11)
        
        # Object position (batch_size, 2)
        object_pos = immediate_object[:, :2]             # Shape: (batch_size, 2)
        object_pos = object_pos.unsqueeze(1)            # Shape: (batch_size, 1, 2) this is the expect shape for criterion
        
        # Compute loss between outputs position and object position
        object_loss = self.criterion(outputs, object_pos)  # Shape: (batch_size, 1, 2)
        object_loss = object_loss.sum(dim=2).squeeze(1)  # Shape: (batch_size,)
        
        # Buzz slider indicator used to increase epsilon
        # This is experimental and allows the model more 'slip' for short sliders
        slider_mask = immediate_object[:, 7]
        
        # Delta time values of immediate object
        # Abs value is taken as the time delta could be negative due to 60hz sampling
        time_values = torch.abs(immediate_object[:, 8]) # Shape: (batch_size, 1)
        
        # Valid Object types (hit_circle, slider_head, slider_tick) where object loss is applied
        # Invalid object types (spinner_start, spinner_ticks) where object loss is ignored
        object_type = immediate_object[:, 2:7]          # Shape (batch_size, 5)
        # A multiplier is applied to epsilon to emphasize certain object types
        object_type_multiplier = torch.sum(object_type * self.type_weight, dim=1)
        
        # Time weights for object loss
        # The weight is large for objects with small time deltas (needs to be hit)
        # No weight is applied for object outside of the time threshold
        time_weights = torch.clamp((0.1 - time_values) / 0.1, min=0, max=1)
        
        """
        Epsilon is the acceptable margin of error where no object loss is applied
        Epsilon decreases when time delta is small
        Epsilon is decreased for high type weights
        Epsilon is scaled up for buzz/short sliders
        """
        epsilon = self.epsilon_min + (self.epsilon_max - self.epsilon_min) * torch.pow(time_values, self.epsilon_expo)
        epsilon = (epsilon + (epsilon * slider_mask * self.buzz_epsilon_mulit)) * (10 / object_type_multiplier)
        
        # Adjusted object loss masked by epsilon
        adjusted_object_loss = torch.clamp(object_loss - epsilon, min=0)
        
        # Applying a multiplier to object loss to emphasize importance and apply time weights
        weighted_object_loss = adjusted_object_loss * self.object_loss_weight * time_weights
        weighted_object_loss_mean = weighted_object_loss.mean()
        
        return [replay_loss_mean, weighted_object_loss_mean]
    
    def _print_progress(self, train_losses, valid_losses, curr_lr, epoch):
        print(
            f"Epoch [{epoch+1}/{self.max_epoch}] "
            f"LR [{curr_lr:.0e}/{self.min_lr:.0e}] "
            f"T Loss Replay: {train_losses[0]:.4f} | "
            f"V Loss Replay: {valid_losses[0]:.4f} | "
            f"T Loss Object: {train_losses[1]:.4f} | "
            f"V Loss Object: {valid_losses[1]:.4f}"
            )
    
    """
    Prediction loop
    Autoregressive predictions happen one at a time, batch size is 1 and this is basically a sequential for loop
    Could use futher optimization
    """
    def predict(self, inputs, progress_callback=None):
        self.model.eval()
        inputs = [[padded_input.pin_memory(), input_length.pin_memory()] for padded_input, input_length in inputs]
        predictions = []
        total_inputs = len(inputs)
        with torch.no_grad():
            # Initializing first cursor position (center)
            prev_pred = torch.tensor([0.5, 0.5], device=self.device).unsqueeze(0).unsqueeze(1)  # Shape (1, 1, 2)
            
            for idx, (padded_input, input_length) in tqdm(enumerate(inputs), total=total_inputs, desc="Predicting Position"):
                padded_input = padded_input.to(self.device)

                # Calculating relative distance from objects in the sequence and prev_pred
                dist_x = (padded_input[:, :, 0] - prev_pred[:, :, 0]).unsqueeze(2)     # Shape (1, seq_len, 1)
                dist_y = (padded_input[:, :, 1] - prev_pred[:, :, 1]).unsqueeze(2)     # Shape (1, seq_len, 1)
                
                # Conatenating along the feature dimension (dim=2)
                padded_input = torch.cat((padded_input, dist_x, dist_y), dim=2)  # Shape (1, seq_len, feature_size + 2)
                
                # Packing inputs
                packed_input = pack_padded_sequence(padded_input, input_length, batch_first=True, enforce_sorted=False)

                output = self.model(packed_input, prev_pred)
                
                # Updating prev_pred for next iteration
                prev_pred = output
                
                predictions.append(output)
                
                # Call back to update gui progress bars
                if progress_callback and (idx + 1) % 1000 == 0:
                    progress_callback((idx + 1) / total_inputs, 'Position')
                
        # On complete set the progress to 100%
        if progress_callback:
            progress_callback(1.0, 'Position')
                
        return predictions