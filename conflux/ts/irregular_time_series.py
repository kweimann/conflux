from conflux.ts.regular_time_series import RegularTimeSeries
from conflux import utils
import numpy as np


class IrregularTimeSeries(object):
    """
    Utility class for unevenly spaced time series i.e. time series whose
    spacing of observation times is not constant
    """
    def __init__(self, observations: np.ndarray, timestamps: np.ndarray):
        """
        :param observations:    Numpy array containing all observations.
        :param timestamps:      Numpy array containing all timestamps corresponding to the observations.
        """
        if len(observations) != len(timestamps):
            raise ValueError("Observations and timestamps have different lengths.")
        self.observations = observations
        self.timestamps = timestamps

    def __iter__(self):
        """
        :return: Zip iterator over timestamps and observations.
        """
        yield from zip(self.timestamps, self.observations)

    def __len__(self):
        """
        :return: Number of observations.
        """
        return len(self.observations)

    def interpolate(self, interval: int, start_timestamp: int=None, end_timestamp: int=None, method='linear'):
        """
        Interpolates observations within provided interpolation interval to produce regular time series
        :param interval:            Interval between two consecutive observations.
        :param start_timestamp:     Timestamp of the first observation. By default, first available timestamp
                                    will be used.
        :param end_timestamp:       Timestamp of the last observation. By default, last available timestamp
                                    will be used.
        :param method:              Interpolation method:
                                    `most_recent` - most recent value
                                    `linear` - linear interpolation (default).
        :return: Interpolated regular time series object.
        """
        if interval <= 0:
            raise ValueError("Interval must be positive.")
        if start_timestamp is not None and start_timestamp < 0:
            raise ValueError("Start timestamp cannot be negative.")
        if end_timestamp is not None and end_timestamp < 0:
            raise ValueError("End timestamp cannot be negative.")
        supported_methods = ["most_recent", "linear"]
        if method not in supported_methods:
            raise ValueError("Unknown interpolation method. Supported methods:", supported_methods)
        if len(self) == 0:
            raise ValueError("No values to interpolate.")

        start_timestamp = start_timestamp if start_timestamp is not None else self.timestamps[0]
        end_timestamp = end_timestamp if end_timestamp is not None else self.timestamps[-1]

        if end_timestamp < start_timestamp:
            raise ValueError("End timestamp must be smaller or equal start timestamp.")

        interpolated_len = (end_timestamp - start_timestamp) // interval + 1
        interpolated_shape = (interpolated_len, *self.observations.shape[1:])
        interpolated = np.empty(interpolated_shape, dtype=self.observations.dtype)

        if method == 'linear':
            interval_iterator = self._interval_iterator(start_timestamp, end_timestamp, interval)
            prev_ts, prev_val, start = next(interval_iterator)
            if prev_val is None:
                # values from first interval should be extrapolated
                prev_ts, prev_val, end = next(interval_iterator)
                if prev_val is None:
                    # special case: no observations within interpolation interval, end has been reached
                    interpolated[start:end] = None

            # define a function returning timestamp array according to observations' dimensionality
            if self.observations.ndim == 1:
                # returns row vector
                def timestamps(start_ts, end_ts):
                    return np.arange(start_ts, end_ts) * interval + start_timestamp
            elif self.observations.ndim == 2:
                # returns column vector
                def timestamps(start_ts, end_ts):
                    return np.arange(start_ts, end_ts).reshape((-1, 1)) * interval + start_timestamp
            else:
                raise ValueError("Linear interpolation supports only 1D and 2D data.")

            for next_interval, peeked_interval in utils.lookahead(interval_iterator):
                if peeked_interval is None:
                    # end has been reached
                    # values from last interval have been interpolated in previous iteration
                    break
                next_ts, next_val, end = next_interval
                _, peeked_val, peeked_end = peeked_interval
                if peeked_val is None:
                    # extrapolate values from last interval
                    end = peeked_end
                interpolated[start:end] = prev_val + (next_val - prev_val) * \
                                                     (timestamps(start, end) - prev_ts) / (next_ts - prev_ts)
                if peeked_val is None:
                    # values from last interval have been extrapolated
                    break
                prev_ts, prev_val, start = next_interval
        elif method == 'most_recent':
            for (_, prev_val, start), (_, next_val, end) in utils.sliding_window(
                    self._interval_iterator(start_timestamp, end_timestamp, interval, skip_within_interval=True)):
                interpolated[start:end] = prev_val if prev_val is not None else next_val

        return RegularTimeSeries(interpolated, interval, start_timestamp)

    def _interval_iterator(self, start_timestamp, end_timestamp, interval, skip_within_interval=False):
        """
        Iterator that yields irregularly spaced observations, their corresponding timestamps and array position in the
        interpolated, regular time series.
        :param start_timestamp:         Timestamp of the first interpolated value.
        :param end_timestamp:           Timestamp of the last interpolated value.
        :param interval:                Time series interval.
        :param skip_within_interval:    If true only the most recent observation within an interval will be returned.
        :return: (timestamp, value, interval_index) iterator
        """
        # calculate indices of the interpolation interval
        last_interval_idx = (end_timestamp - start_timestamp) // interval + 1
        start_idx = self._index_of(start_timestamp)

        if start_idx == -1:
            # first value lies outside of interpolation interval
            # so the values within first interval must be extrapolated
            prev_ts = start_timestamp
            prev_val = None
        else:
            prev_ts = self.timestamps[start_idx]
            prev_val = self.observations[start_idx]

        prev_interval_idx = 0
        next_ts, next_val = prev_ts, prev_val

        # iterate over all observations that lie within interpolation interval
        for i in range(start_idx + 1, len(self)):
            next_ts, next_val = self.timestamps[i], self.observations[i]
            # end of interpolation interval has been reached
            if next_ts > end_timestamp:
                break
            # round current interval index up
            interval_idx = self._ceil_div(next_ts - start_timestamp, interval)
            # yield previous observation
            if not skip_within_interval or interval_idx > prev_interval_idx:
                yield prev_ts, prev_val, prev_interval_idx
                prev_interval_idx = interval_idx
            # update previous observation for next iteration
            prev_ts = next_ts
            prev_val = next_val

        # yield the remaining previous observation
        yield prev_ts, prev_val, prev_interval_idx

        if end_timestamp > next_ts:
            # last value lies outside of interpolation interval
            # so the values within last interval must be extrapolated
            yield end_timestamp, None, last_interval_idx
        else:
            yield next_ts, next_val, last_interval_idx

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

    def _index_of(self, timestamp):
        """
        :param timestamp:   Timestamp of an observation.
        :return: Index of the most recent observation before the provided timestamp or -1 if
                 the provided timestamp comes before any observation timestamp.
        """
        for i, ts in enumerate(self.timestamps):
            if ts > timestamp:
                return i - 1
        return len(self) - 1

    @staticmethod
    def _ceil_div(n, d):
        return -(-n // d)
