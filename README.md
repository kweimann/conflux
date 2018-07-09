# Conflux

Time series utility library for Python.

Features:
* classes for evenly / unevenly spaced time series
* interpolation of unevenly spaced time series ([interpolation example](#interpolation))
* transformation of evenly spaced time series into dataset ([forecast example](#neural-network-forecast))
* time series forecast wrapper for e.g. keras ([forecast example](#neural-network-forecast))


### Installation

Requirements:
* `Python3.5+`

Installation:
1. `git clone https://github.com/kweimann/conflux.git`
2. `cd conflux`
3. `pip install .`

### Examples

#### Interpolation

See `examples/interpolation.py` for complete example.

```python
# number of observations
n = 25  
# time interval i.e. first and last timestamp
t0, tn = [0, 200]
# function producing observation value from observation timestamp
f = lambda x: .00001 * (x - 100) * (x - 50) * (x - 150)
# 1000 values from time interval used for plotting
xs = np.linspace(t0, tn, 1000)

# randomly choose `n` timestamps from time interval
timestamps = np.sort(np.random.choice(np.arange(t0, tn + 1, 1), n, replace=False))
# calculate their respective observed values
observations = np.apply_along_axis(f, arr=timestamps, axis=0)

# create time series object for unevenly spaced time series
time_series = conflux.ts.IrregularTimeSeries(observations, timestamps)

# linearly interpolate the time series starting from `t0` until `tn`
linear_interp = time_series.interpolate(interval=1, start_timestamp=t0,
                                        end_timestamp=tn, method="linear")

# interpolate time series using `most recent` strategy starting from `t0` until `tn`
most_recent_interp = time_series.interpolate(interval=1, start_timestamp=t0,
                                             end_timestamp=tn, method="most_recent")
```

![interpolation_example](https://user-images.githubusercontent.com/8287691/42472678-799dfd42-83c2-11e8-8ac2-d885c5aac97a.png)

#### Neural Network Forecast

See `examples/neural_network_forecast.py` for complete example.

```python
# use keras.Sequential's predict method as
# the implementation of Solver's abstract predict method
class NeuralNetwork(keras.models.Sequential, conflux.Solver):
    pass

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

# prepare a very simple neural network with just one dense layer
model = NeuralNetwork()
model.add(keras.layers.Dense(units=n_out, input_shape=(n_in,)))
model.compile(optimizer=keras.optimizers.RMSprop(), loss=keras.losses.mean_squared_error)

# train the model on the train set
model.fit(train.x, train.y, epochs=500)

# forecast future values from both test input arrays
predicted_1 = model.forecast(test_1.x, steps=steps_1)
predicted_2 = model.forecast(test_2.x, steps=steps_2)
```
Output:
```
expected 1:
features	outputs
[179.]	[180. 181. 182. 183. 184. 185. 186. 187. 188. 189. 190. 191. 192. 193.
 194. 195. 196. 197. 198. 199.]

predicted 1:
features	outputs
[179.]	[180.09366 181.1879  182.2827  183.37808 184.47404 185.57059 186.66771
 187.76541 188.8637  189.96255 191.062   192.16202 193.26262 194.3638
 195.46556 196.5679  197.67082 198.77432 199.8784  200.98306]

expected 2:
features	outputs
[179.]	[180.]
[180.]	[181.]
[181.]	[182.]
[182.]	[183.]
[183.]	[184.]
[184.]	[185.]
[185.]	[186.]
[186.]	[187.]
[187.]	[188.]
[188.]	[189.]
[189.]	[190.]
[190.]	[191.]
[191.]	[192.]
[192.]	[193.]
[193.]	[194.]
[194.]	[195.]
[195.]	[196.]
[196.]	[197.]
[197.]	[198.]
[198.]	[199.]

predicted 2:
features	outputs
[179.]	[180.09366]
[180.]	[181.09418]
[181.]	[182.09471]
[182.]	[183.09525]
[183.]	[184.09576]
[184.]	[185.0963]
[185.]	[186.09682]
[186.]	[187.09735]
[187.]	[188.09787]
[188.]	[189.0984]
[189.]	[190.09892]
[190.]	[191.09946]
[191.]	[192.09998]
[192.]	[193.10051]
[193.]	[194.10104]
[194.]	[195.10156]
[195.]	[196.1021]
[196.]	[197.10262]
[197.]	[198.10315]
[198.]	[199.10367]
```
