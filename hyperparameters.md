<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li><a href="#Data">Data</a></li>
    <li><a href="#Model">Model</a></li>
    <li><a href="#Loss">Loss</a></li>
    <li><a href="#Train">Train</a></li>
  </ol>
</details>

# Data
### Context size (int)
Context size applies to both Position and Keypress models and determines the **maximum** amount of objects the model can perceive at any given time. For example, at a certain point on a beatmap, if there are 10 objects visible and the context size is 10, then the model will perceive all objects. If there are 20 objects visible then the model will only perceive the first 10. Giving too little context can make the model lose the ability to identify patterns and giving too much may confuse the model on how to approach the immediate pattern.
### Time window (int)(ms)
Time window applies to both Position and Keypress models and determines how early the model is able to perceive objects. For example at approach rate 9, objects appear on the screen 600ms before they need to be hit [**(more)**](https://osu.ppy.sh/wiki/en/Beatmap/Approach_rate) and a time window of 600 will ensure the model perceives the object as soon as they appear. It's recommended to choose a value that is somewhat close to the approach rates of your chosen data.
### Buzz thresholds (float)(osu! unit)
Buzz thresholds apply to Position model and indicates to the model whether a slider (linear, circle, bezier) needs to be followed precisely. For a chosen threshold **x**, any slider which does not move **x** units from the slider head will be considered a buzz slider.
### Batch sizes (int)
Batch sizes determine how much data is fed into the model in parallel during training. Setting a high batch size will utilize more GPU and VRAM and may reduce training time. A smaller batch size will increase training time but may result in better gradient updates. At the default training batch size of 1024 and validation batch size of 64 around 2.5Gb of VRAM is used. 

# Model
### Input size (Do not change)
Size of the input layer for the network. This is determined by the shape of input tensors so unless the data processing steps are changed do not modify this value.

### Hidden size (int)
Hidden size is the number of units/neurons in the hidden layers. More units will allow the model to capture more complex patterns and dependencies at the cost of more computation.

### Num layers (int)
Num layers is the number of stacked LSTM layers. More layers can allow the model to break down the data into more hierarchical 'stages' to capture complex dependencies but at a much higher computational cost.

# Loss (Position model only)
Evaluating model performance by comparing predictions and replay data (replay loss) does not guarantee good real world performance. Even big datasets cannot encompass every pattern and human data is inherently noisy and variant so I've introduced a helper loss called object loss. Object loss is the absolute error between the model prediction and the center of objects and it encourages the model to predict inside the object or in other words be more accurate. Below are hyperparameters that control object loss.

### Epsilon (float)(osu! units)
Epsilon is the acceptable margin of error for position model predictions. If the predicted position is within epsilon of the object center, then no additional object loss is applied. Epsilon decreases as the time to hit an object approaches. The formula for epsilon is shown [**here**](https://www.desmos.com/calculator/8emwjbqngj) (x-axis is normalized time between 0 (needs to be hit) and 1 (just appeared)). Both prediction x,y position and epsilon are normalized values between 0 and 1. No additional loss is applied regardless of epsilon if the normalized time to object is > 0.1 meaning the model is only penalized for being too far from the object when the object needs to be hit soon.

### Precision (float)
Precision ranges from 10 (very precise) and 0 (not precise). This value serves as a mulitplier to increase epsilon for different object types. Slider tick precision is set to 5 by default which results in (10 / 5) = 2, so 2 times the acceptable margin of error for slider ticks. 0 precision is set for spinner which is (10 / 0) = inf according to PyTorch so essentially no additional object loss is applied for spinners, which is what we want.

### Buzz slider multi (float)
Buzz slider mulit is an additional multiplier for epsilon only applicable to sliders which are determined to be buzz or short sliders. This value is added as such (epsilon + (epsilon * buzz_slider_multi)) so with a default value of 2, epsilon is increased by 200%.

### Object loss weight (float)
Object loss weight scales the importance of object loss. At the default value of 10, we are penalizing the model heavily for missing or being outside of epsilon from the objects which need to be hit. This weight can be set lower as object loss is intended to help direct the model to be more accurate. Given a very good dataset the model can be accurate with less help or given a bad dataset this value can be increased for more enforcement. 

# Train
### Learning rate (float)
Learning rate (lr) is the initial lr and controls the magnitude of model weights updates. A higher learning rate will speed up training but as the optimizer approaches an optimal minima the learning rate should be reduced to not over-shoot it. This is done automatically with the scheduler and the default value of 0.01 should suffice.

### Weight decay (float)
Weight decay controls the penalty for large weights (indication of over-fitting) in the model. Setting the value lower will reduce the penalty and increase the risk of over-fitting and vice-versa.

### Patience (int)
Patience is the number of epochs the scheduler will wait when validation loss is not improving. At the default value of 15, when validation loss plateaus the scheduler will remain at the same lr for 15 epochs then decrease lr by a factor of 0.1.

### Max epoch (int)
The maximum epoch the training loop will reach, after which the training loop will exit and training will be completed. A higher max epoch will make the training longer given that it doesn't run into an early stopping condition. A lower epoch will limit the time training takes but may cause the model to miss convergence on the optimum depending on other hyperparameters.

### Early stopping learning rate (float)
The minimum lr that will be reached by the scheduler before triggering the early stopping condition and ending training. Increasing this value can decrease training time but may cause the model to miss convergence on the optimum. Decreasing this value can increase training time but allow the model to settle more closely to the optimum. However keep in mind with a small lr there won't be many meaningful weight updates.

### Seed (int)
Seed for random number generators. 