from itertools import tee


def sliding_window(iterable, size: int = 2):
    """
    :param iterable:    Iterable collection.
    :param size:        Size of the sliding window.
    :return: sliding window over iterator.
    """
    iterators = tee(iter(iterable), size)
    for i in range(1, size):
        for it in iterators[-i:]:
            next(it)
    yield from zip(*iterators)


def either(a, b):
    """
    :param a: Uncertain value (might be None).
    :param b: Default value.
    :return: Either the uncertain value if it is not None or the default value.
    """
    return b if a is None else a


def lookahead(iterable):
    """
    :param iterable: Iterable collection.
    :return: Iterator over collection that yields current element and peeked next element
    """
    it = iter(iterable)
    prev = next(it)
    for val in it:
        yield prev, val
        prev = val
    yield prev, None
