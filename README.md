# Public 45th solution for Kaggle NFL Big Data Bowl: CNN + MLP
 
[Kaggle NFL Big Data Bowl](
https://www.kaggle.com/c/nfl-big-data-bowl-2020/leaderboard
)

## Summary:
2d-CNN for sparse heatmap images and MLP for tabular data

### CNN (Convolutional Neural Network)

Generated heatmap-like field images of 30 ((YardLine - 10) <= X < (YardLine + 20)) x 54 (0 <= Y < 54) yards grid 
(rounded to integer).

Usual color image consists of 3 channels of RGB, but I added more channels (3 x 3 x 2 = 18)

##### 3 player categories:
- 11 defense players
- 10 offense players excluding the rusher
- The rusher (ball carrier)

##### 3 variables:
- A (acceleration)
- S_X (speed in X axis)
- S_Y (Speed in Y-axis)

##### 2 frames:
Computed another snapshot of 1 second later by adding the speed.
(Also tried adding acceleration, but did not improve the performance.)

The final tensor size for CNN was [Batch size, Channel, X, Y] was [32, 18, 30, 54] 

Kept the X-direction until the dense layer and compressed in only Y-direction (stride=[1, 2]) 
as the X-direction is related to the outcome variable (Yards).


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
1. CNN and MLP
2. Concatenated outputs of CNN and MLP.
3. Add base probabilities computed by counting the Yards in whole train dataset in 2018 to let neural network
4. Forced predicted probabilities for the yards beyond the goal line to 0
5. Pad 0 to lower (-99 <= Yards < -10 yards) and upper (90 <= Yards < 100).
6. Divide by the sum through Yards to make the sum to 1 (SoftMax without exponential)
7. Compute cumulative sum through Yards

### Loss function
CRPS with yards clipped to -10 to 29 yards

### Other tricks

- Concatenated different kernel sizes as introduced in Inception architecture
- Subtle augmentation
  - random shift in X-axis: 0.1 yards stdev of normal distribution
  - random shift in Y-axis: 1.0 yards stdev of normal distribution
  - (random flip in Y-axis decreased the performance thus not used.) 
- Discarded all 2017 data which was very different from 2018 due to sensor issues and hard to adjust


### Tried but did not work well:
- Treat players as graph
  - e.g. Use the reciprocal of distance between players as edge weights, compute 
Laplacian spectrum, count the number of 0 eigenvalues which equals to number of connected subgraphs 
to use additional features


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
