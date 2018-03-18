import abc
import numpy as np


class Solver(abc.ABC):
    @abc.abstractmethod
    def predict(self, x: np.ndarray, **kwargs) -> np.ndarray:
        """
        :param x:       Numpy input array of shape (num_examples, n_time_steps, n_in) or (num_examples, n_in).
                        Only requires support for handling input arrays that will be passed to the `forecast` method.
        :param kwargs:  Additional arguments.
        :return: 2D Numpy output array of shape (num_examples, n_out).
        """
        pass

    def forecast(self, x: np.ndarray, steps: int = None, **kwargs):
        """
        :param x:       Numpy input array of shape (num_examples, n_time_steps, n_in) or (num_examples, n_in).
        :param steps:   Number of forecast steps, i.e. how many values to forecast per input.
                        By default steps is equal n_out.
        :param kwargs:  Additional arguments for `predict` method.
        :return: 2D Numpy output array of shape (num_examples, steps).
        """
        y = self.predict(x, **kwargs)

        if steps is None:
            return y
        elif steps <= 0:
            raise ValueError("Number of steps must be positive.")
        if 2 > x.ndim > 3:
            raise ValueError("Only 2D or 3D input arrays are supported.")

        num_examples, n_out = y.shape

        if x.ndim == 3:
            _, _, n_in = x.shape
            if n_in != n_out:
                raise ValueError("Output must have the same size as the input for temporal data set.")

        if steps <= n_out:
            return y[:, :steps]

        n_forecasts = -(-steps // n_out)

        inputs = np.empty(x.shape, dtype=x.dtype)
        predicted = np.empty((num_examples, n_forecasts * n_out), dtype=y.dtype)

        inputs[:] = x[:]
        predicted[:, :n_out] = y

        if x.ndim == 2:
            num_examples, n_in = x.shape

            if n_in > n_out:
                for i in range(1, n_forecasts):
                    inputs[:, :-n_out] = inputs[:, n_out:]
                    inputs[:, -n_out:] = predicted[:, (i - 1) * n_out:i * n_out]
                    predicted[:, i * n_out:(i + 1) * n_out] = self.predict(inputs, **kwargs)
            else:
                for i in range(1, n_forecasts):
                    inputs[:] = predicted[:, i * n_out - n_in:i * n_out]
                    predicted[:, i * n_out:(i + 1) * n_out] = self.predict(inputs, **kwargs)
        elif x.ndim == 3:
            for i in range(1, n_forecasts):
                inputs[:, :-1, :] = inputs[:, 1:, :]
                inputs[:, -1, :] = predicted[:, (i - 1) * n_out:i * n_out]
                predicted[:, i * n_out:(i + 1) * n_out] = self.predict(inputs, **kwargs)

        return predicted[:, :steps]
