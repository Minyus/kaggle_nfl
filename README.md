# Public 45th solution for Kaggle NFL: 
 
[Kaggle NFL Big Data Bowl](
https://www.kaggle.com/c/nfl-big-data-bowl-2020/leaderboard
)

## Summary:
2d-CNN (Convolutional Neural Network) for sparse heatmap images and MLP for tabular data

### Input tensor for CNN

Generated heatmap-like field images of 30 ((YardLine - 10) <= X < (YardLine + 20)) x 54 (0 <= Y < 54) yards grid 
(rounded to integer).

After several experiments, the following combinations of 18 (= 3 x 3 x 2) channels worked best.

#### 3 player categories:
- 11 defense players
- 10 offense players excluding the rusher
- The rusher (ball carrier)

#### 3 variables:
- A (acceleration)
- S_X (speed in X axis)
- S_Y (speed in Y-axis)

#### 2 frames:
Computed another snapshot of 1 second later by adding the speed.
(Also tried adding acceleration, but did not improve the performance.)


## CNN architecture

- 4 layers
- Kept the X-direction until the dense layer and compressed in only Y-direction (stride=[1, 2]) 
as the X-direction is related to the outcome variable (Yards).
- Concatenated different kernel sizes as introduced in the Inception architecture
- CELU activation (slightly faster training than ReLU)

CNN part of the architecture was configured in YAML for PyTorch as follows.

 (Please see [PipelineX](https://github.com/Minyus/pipelinex) for the syntax)
```yaml
  =: torch.nn.Sequential
  _:
    - {=: pipelinex.TensorSlice, end: 18}
    - =: pipelinex.ModuleConcat
      _:
        - {=: pipelinex.TensorConv2d, in_channels: 18, out_channels: 10, kernel_size: [3, 3]}
        - {=: pipelinex.TensorConv2d, in_channels: 18, out_channels: 10, kernel_size: [7, 7]}
        - {=: pipelinex.TensorConv2d, in_channels: 18, out_channels: 10, kernel_size: [3, 9]}
    - {=: torch.nn.CELU, alpha: 1.0}
    - =: pipelinex.ModuleConcat
      _:
        - {=: pipelinex.TensorAvgPool2d, stride: [1, 2], kernel_size: [3, 3]}
        - {=: pipelinex.TensorConv2d, stride: [1, 2], in_channels: 30, out_channels: 10, kernel_size: [3, 3]}
        - {=: pipelinex.TensorConv2d, stride: [1, 2], in_channels: 30, out_channels: 10, kernel_size: [7, 7]}
        - {=: pipelinex.TensorConv2d, stride: [1, 2], in_channels: 30, out_channels: 10, kernel_size: [3, 9]}
    - {=: torch.nn.CELU, alpha: 1.0}
    - =: pipelinex.ModuleConcat
      _:
        - {=: pipelinex.TensorAvgPool2d, stride: [1, 2], kernel_size: [3, 3]}
        - {=: pipelinex.TensorConv2d, stride: [1, 2], in_channels: 60, out_channels: 20, kernel_size: [3, 3]}
        - {=: pipelinex.TensorConv2d, stride: [1, 2], in_channels: 60, out_channels: 20, kernel_size: [7, 7]}
        - {=: pipelinex.TensorConv2d, stride: [1, 2], in_channels: 60, out_channels: 20, kernel_size: [3, 9]}
      # -> [N, 120, 30, 14]
    - {=: torch.nn.CELU, alpha: 1.0}
    - =: pipelinex.ModuleConcat
      _:
        - =: torch.nn.Sequential
          _:
            - {=: torch.nn.AvgPool2d, stride: [1, 2], kernel_size: [3, 14]}
            # -> [N, 120, 28, 1]
            - {=: pipelinex.TensorConv2d, in_channels: 120, out_channels: 20, kernel_size: [1, 1]}
            - {=: pipelinex.TensorFlatten, _: }
            - {=: torch.nn.CELU, _: }
        - =: torch.nn.Sequential
          _:
            - {=: torch.nn.MaxPool2d, stride: [1, 2], kernel_size: [3, 14]}
            # -> [N, 120, 28, 1]
            - {=: pipelinex.TensorConv2d, in_channels: 120, out_channels: 20, kernel_size: [1, 1]}
            - {=: pipelinex.TensorFlatten, _: }
            - {=: torch.nn.CELU, _: }
```

### MLP (Multilayer Perceptrons)

Add another channel to encode tabular features.

##### Continuous features
Max, Min, Mean, Stdev for axis (X, Y) and player categories (Defense, Offense)

##### Categorical features (One-hot encoded)
- Down (1, 2, 3, 4)
- Flag of whether offense is home
- OffenseFormation
- DefendersInTheBoxCode

### Computing the CDF output
1. Concatenate outputs of CNN and MLP.
2. Add base probabilities computed by counting the Yards in the whole train dataset in 2018 to let the
neural network learn the residual.
3. Forced predicted probabilities for the yards beyond the goal line to 0
4. Pad 0 to lower (-99 <= Yards < -10 yards) and upper (90 <= Yards < 100).
5. Divide by the sum through Yards to make the sum to 1 (SoftMax without exponential)
6. Compute cumulative sum through Yards

### Loss function
CRPS with yards clipped to -10 to 29 yards

### Other settings

- Subtle augmentation
  - random shift in X-axis: 0.1 yards stdev of normal distribution
  - random shift in Y-axis: 1.0 yards stdev of normal distribution
  - (random flip in Y-axis decreased the performance thus not used.) 
- Discarded all 2017 data which was very different from 2018 due to sensor issues and hard to adjust
- Batch size: 32
- Optimizer: Adam
- Learning rate scheduler: LinearCyclicalScheduler (slightly better than CosineAnnealingScheduler)

### What did not work:
- Treat players as graph
  - Use the reciprocal of distance between players as edge weights, compute 
Laplacian spectrum, count the number of 0 eigenvalues which equals to number of connected subgraphs 
to use additional features
- Scaling
  - RankGauss
  - StandardScaler


## Dependencies available in Kaggle Kernel
- torch==1.1.0
- pytorch-ignite==0.2.0
- pandas==0.25.1
- numpy==1.16.4

## Dependencies not available in Kaggle Kernel
- [pipelinex](https://github.com/Minyus/pipelinex) (developed with/for this competition and open-sourced)

## Dependencies only for experimentation (not used in Kaggle Kernel)
- kedro
- mlflow 
