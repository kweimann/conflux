import conflux
import keras
import numpy as np
import matplotlib.pyplot as plt


# use keras.Sequential's predict method as
# the implementation of Solver's abstract predict method
class NeuralNetwork(keras.models.Sequential, conflux.Solver):
    pass


if __name__ == "__main__":
    """
    This is a simple example showing how to use the solver class
    for wrapping models (e.g. keras sequential) in order to 
    enable forecasting future values where number of forecast steps
    does not have to be equal to the number of output nodes of a neural network.
    
    Furthermore, this example shows how to transform evenly spaced
    time series into a data set that is later used for model fitting.
    """
    np.random.seed(67)

    # time series: 0..199
    data = np.arange(200, dtype=np.float32)
    # number of input / output nodes in the neural network
    n_in, n_out = 1, 1
    # number of forecasting steps
    steps_1, steps_2 = 20, 1

    # create evenly spaced time series from the data
    ts = conflux.ts.RegularTimeSeries(data, interval=1, start_timestamp=0)
    # transform time series into data set
    ds = ts.to_dataset(n_in=n_in, n_out=n_out)
    # split the data set into train and test
    # test_1 has exactly one input vector of shape (1,)
    # and one output vector of shape (20,) i.e. 20 forecast steps
    train, test_1 = ds.split_train_test(-1, test_n_out=steps_1)
    # test_2 has 20 input vectors of shape (1,)
    # and 20 out vectors of shape (1,) i.e. 1 forecast step
    train_, test_2 = ds.split_train_test(-20, test_n_out=steps_2)

    # assert train == train_
    assert np.all(np.equal(train.x, train_.x)) and np.all(np.equal(train.y, train_.y))

    # prepare a very simple neural network with just one dense layer
    model = NeuralNetwork()
    model.add(keras.layers.Dense(units=n_out, input_shape=(n_in,)))
    model.compile(optimizer=keras.optimizers.RMSprop(), loss=keras.losses.mean_squared_error)

    # train the model on the train set
    model.fit(train.x, train.y, epochs=500)

    # forecast future values from both test input arrays
    predicted_1 = model.forecast(test_1.x, steps=steps_1)
    predicted_2 = model.forecast(test_2.x, steps=steps_2)

    print("\nexpected 1:\nfeatures\toutputs")
    for xi, yi in zip(test_1.x, test_1.y):
        print("{}\t{}".format(xi, yi))

    print("\npredicted 1:\nfeatures\toutputs")
    for xi, yi in zip(test_1.x, predicted_1):
        print("{}\t{}".format(xi, yi))

    print("\nexpected 2:\nfeatures\toutputs")
    for xi, yi in zip(test_2.x, test_2.y):
        print("{}\t{}".format(xi, yi))

    print("\npredicted 2:\nfeatures\toutputs")
    for xi, yi in zip(test_2.x, predicted_2):
        print("{}\t{}".format(xi, yi))

    plt.plot(np.arange(len(ds)), ds.y, label="time series")
    plt.plot(np.arange(len(train), len(ds)), predicted_1.flatten(), label="predicted (20 steps)")
    plt.plot(np.arange(len(train), len(ds)), predicted_2, label="predicted (1 step)")
    plt.legend()
    plt.show()
