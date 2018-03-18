import numpy as np


class RegularTimeSeries(object):
    """
    Utility class for evenly spaced time series i.e. time series whose
    spacing of observation times is constant
    """
    def __init__(self, observations: np.ndarray, interval: int, start_timestamp: int):
        """
        :param observations:        Numpy array containing all observations.
        :param interval:            Interval between two consecutive observations.
        :param start_timestamp:     Timestamp of the first observation.
        """
        if interval <= 0:
            raise ValueError("Interval must be a positive number.")
        if start_timestamp < 0:
            raise ValueError("Start timestamp must not be a negative number.")
        self.observations = observations
        self.interval = interval
        self.start_timestamp = start_timestamp

    def __iter__(self):
        """
        :return: Zip iterator over timestamps and observations.
        """
        return zip(self.timestamps, self.observations)

    def __len__(self):
        """
        :return: Number of observations.
        """
        return len(self.observations)

    def __getitem__(self, item):
        """
        :param item: Observation index or a slice.
        :return: Observation at given index and the corresponding timestamp or a sliced regular time series object.
        """
        if isinstance(item, slice):
            if item.step is None:
                interval = self.interval
            elif item.step < 0:
                raise ValueError("Slice step cannot be negative.")
            else:
                interval = self.interval * item.step
            if item.start is None:
                start = 0
            elif item.start < 0:
                start = len(self) + item.start
            else:
                start = item.start
            observations = self.observations[item]
            start_timestamp = self.start_timestamp + start * self.interval
            return RegularTimeSeries(observations, interval, start_timestamp)
        elif isinstance(item, int):
            return self.start_timestamp + item * self.interval, self.observations[item]
        else:
            raise TypeError("Invalid argument type.")

    def at_timestamp(self, timestamp: int, exact=False):
        """
        :param timestamp:   Timestamp of an observation.
        :param exact:       If false the provided timestamp may lie between timestamps of two consecutive
                            observations in which case the earlier observation will be returned, otherwise
                            the provided timestamp must exactly match one of the timestamps.
        :return: An observation matching the provided timestamp.
        """
        if not self.start_timestamp <= timestamp < self.start_timestamp + len(self) * self.interval:
            raise IndexError("Invalid timestamp")
        if exact and (timestamp - self.start_timestamp) % self.interval != 0:
            raise ValueError("Timestamp is not exact.")
        index = (timestamp - self.start_timestamp) // self.interval
        return self.observations[index]

    def split(self, i: int):
        """
        :param i:   Index of the split.
        :return: Two regular time series split at the provided index.
        """
        return self[:i], self[i:]

    @property
    def shape(self):
        """
        :return: Shape of the observations array.
        """
        return self.observations.shape

    @property
    def dtype(self):
        """
        :return: Data type of the observation array.
        """
        return self.observations.dtype

    @property
    def ndim(self):
        """
        :return: Number of dimensions of the observation array.
        """
        return self.observations.ndim

    @property
    def timestamps(self):
        """
        :return: Numpy array containing timestamps of the observations.
        """
        return np.arange(len(self)) * self.interval + self.start_timestamp

    @property
    def end_timestamp(self):
        """
        :return: Timestamp of the last observation.
        """
        return self.start_timestamp + max(0, len(self) - 1) * self.interval
