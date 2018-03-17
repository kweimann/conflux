from conflux.ts.regular_time_series import RegularTimeSeries
import numpy as np


class DataSet(object):
    """
    Utility class for holding a data set created from a regular time series.
    """
    def __init__(self, x: np.ndarray, y: np.ndarray):
        if len(x) != len(y):
            raise ValueError("Features and labels have different lengths.")
        self.x = x
        self.y = y

    def __iter__(self):
        """
        :return: Features and labels of the data set.
        """
        yield self.x
        yield self.y

    def __len__(self):
        """
        :return: Size of the data set.
        """
        return len(self.x)

    def __getitem__(self, item):
        """
        :param item: Data index or slice object.
        :return: Feature and label array at given index or a sliced data set object.
        """
        if isinstance(item, slice):
            return DataSet(self.x[item], self.y[item])
        elif isinstance(item, int):
            return self.x[item], self.y[item]
        else:
            raise TypeError("Invalid argument type.")

    def reshape(self, shape_x=None, shape_y=None):
        """
        :param shape_x:     Shape of the new features. By default features will not be reshaped.
        :param shape_y:     Shape of the new labels. By default labels will not be reshaped.
        :return: Reshaped data set.
        """
        return DataSet(self.x.reshape(shape_x) if shape_x is not None else self.x,
                       self.y.reshape(shape_y) if shape_y is not None else self.y)

    @classmethod
    def from_regular_ts(cls,
                        ts: RegularTimeSeries, *,
                        n_in: int,
                        n_out: int,
                        n_time_steps: int = None,
                        step: int = None):
        """
        Creates a data set from a regular time series by sliding a window over the observations.
        :param ts:              Regular time series object.
        :param n_in:            Size of the input vector.
        :param n_out:           Size of the output vector.
        :param n_time_steps:    Number of time steps in the input vector. Supplying this argument
                                creates a 3D feature array. By default the time component is not considered.
        :param step:            Sliding window step size. By default equal to `n_out`.
        :return: Data set.
        """
        if n_in <= 0:
            raise ValueError("Input vector size must be a positive number.")
        if n_out <= 0:
            raise ValueError("Output vector size must be a positive number.")
        if n_time_steps is not None:
            if n_time_steps <= 0:
                raise ValueError("Number of time steps must be positive.")
            if n_in != n_out:
                raise ValueError("Output must have the same size as the input.")
        if step is not None and step <= 0:
            raise ValueError("Step size must be a positive number.")

        if ts.ndim == 1:
            array_length, obs_length = len(ts), 1
        elif ts.ndim == 2:
            array_length, obs_length = ts.shape
        else:
            raise ValueError("Only 1D or 2D time series are allowed.")

        block_size = n_in * (n_time_steps or 1) + n_out

        if array_length < block_size:
            raise ValueError("Not enough data for given input/output parameters.")

        step = step if step is not None else n_out

        remainder = (array_length - block_size) % step
        data_set_size = (array_length - block_size) // step + 1

        if n_time_steps is None:
            features = np.empty((data_set_size, n_in * obs_length))
            labels = np.empty((data_set_size, n_out * obs_length))
            for row, i in enumerate(range(block_size + remainder, array_length + 1, step)):
                features[row] = ts.observations[i - block_size:i - n_out].flatten()
                labels[row] = ts.observations[i - n_out:i].flatten()
        else:
            features = np.empty((data_set_size, n_time_steps, n_in * obs_length))
            labels = np.empty((data_set_size, n_out * obs_length))
            for row, i in enumerate(range(block_size + remainder, array_length + 1, step)):
                features[row] = ts.observations[i - block_size:i - n_out].reshape(n_time_steps, n_in * obs_length)
                labels[row] = ts.observations[i - n_out:i].flatten()

        return cls(features, labels)

    def split(self, i: int):
        """
        :param i:   Index of the split.
        :return: Two data sets split at the provided index.
        """
        return self[:i], self[i:]
