from collections import namedtuple

PruneTypes = namedtuple('PruneTypes', 'index prune_ratio')

ResNet20_Channel_Prune = PruneTypes(
    index = [9, 28, 47],
    prune_ratio = [
        [6, 6, 7, 7],
        [7, 10, 13, 15],
        [15, 21, 27, 31],
    ]
)

ResNet32_Channel_Prune = PruneTypes(
    index = [9, 40, 71],
    prune_ratio = [
        [3, 4, 6, 7],
        [7, 10, 13, 15],
        [15, 21, 27, 31],
    ]
)

ResNet56_Channel_Prune = PruneTypes(
    index = [9, 64, 119],
    prune_ratio = [
        [3, 4, 6, 7],
        [7, 10, 13, 15],
        [15, 21, 27, 31],
    ]
)

ResNet110_Channel_Prune = PruneTypes(
    index = [9, 118, 227],
    prune_ratio = [
        [6, 6, 7, 7],
        [7, 10, 13, 15],
        [15, 21, 27, 31],
    ]
)

ResNet164_Channel_Prune = PruneTypes(
    index = [10, 155, 300],
    prune_ratio = [
        [7, 7, 7, 7],
        [7, 10, 13, 15],
        [15, 21, 27, 31],
    ]
)

