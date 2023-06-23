# Assignment S8

This repo contains my submission for the 8th assignment of TSAI ERA V1 Course. The task was to train a fully convolutional neural network that satisfies the following constraints on the CIFAR10 dataset

- Skeleton of C1 C2 c3 P1 C4 C5 C6 c7 P2 C8 C9 C10 GAP c11 where
  - C —> 3x3 conv layer
  - c —> 1x1 conv layer
  - P —> 2x2 Max Pooling layer
  - GAP —> Global Avg Pooling layer
- ~50K parameters
- Trained for not longer than 20 epochs
- 3 models in total (one per Batch, Group and Layer Normalization)
- 70%+ test accuracy for each model

Owing to shortage of time, I did not evaluate misclassified images. This will be included in future work
