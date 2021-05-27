from collections import namedtuple

PruneTypes = namedtuple('PruneTypes', 'index prune_ratio')

ResNet20_Channel_Prune = PruneTypes(
    index = [9, 28, 47]
    prune_ratio = [
        [0, 2, 4, 7],
        [0, 5, 10, 15],
        [0, 11, 21, 31],
    ]
)