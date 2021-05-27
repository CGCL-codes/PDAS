# PDAS
PDAS (Progressive Differentiable Architecture Search) is a novel network pruning algorithm, which aims to automatically find an appropriate layer width for each layer to reduce network redundancy.
We divide the whole search procedure of neural networks into two stages to approach our search target gradually. The first stage is responsible for searching the sizes of the first few convolution layers in residual blocks to obtain a semi-compact network, of which the layer widths are the candidate number of channels with the highest probability. While the second stage continues to search the widths of the remaining layers to get the final pruned network.
stage 1: python train_search_param1.py --change --data /path to your data --save log_path
stage 2: python train_search_param1.py --change --data /path to your data --save log_path
stage 1 (for resnet-164): python train_search_param164_1.py --change --data /path to your data --save log_path
stage 2 (for resnet-164): python train_search_param164_2.py --change --data /path to your data --save log_path
