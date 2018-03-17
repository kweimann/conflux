# conflux
time series utility library for Python

features:
* evenly / unevenly spaced time series support
* interpolation (unevenly spaced -> evenly spaced time series)
* dataset from evenly spaced time series

### usage

for complete example see: `example.py`

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

# create data set from linearly interpolated time series
# features shape: (200, 1) labels shape: (200, 1)
ds = conflux.ts.DataSet.from_regular_ts(linear_interp, n_in=1, n_out=1)  

# split dataset into train and test, test includes last 20 feature-label pairs
_, (test_features, test_labels) = ds.split(-20)

print("test set:\nfeatures\tlabels")
for x, y in zip(test_features, test_labels):
    print("{}\t{}".format(x, y))

plt.plot(xs, f(xs), label="f(x)")
plt.plot(timestamps, observations, 'o', label="observations")
plt.plot(linear_interp.timestamps, linear_interp.observations, label='linear interp')
plt.plot(most_recent_interp.timestamps, most_recent_interp.observations, label='most recent interp')
plt.legend()
plt.show()

```

![example](https://user-images.githubusercontent.com/8287691/37554460-5cb060a0-29d9-11e8-8dfc-c36cc8945e69.png)

```
test set:
features	labels
[ 3.17712]	[ 3.3396]
[ 3.3396]	[ 3.50208]
[ 3.50208]	[ 3.66456]
[ 3.66456]	[ 3.82704]
[ 3.82704]	[ 4.01625]
[ 4.01625]	[ 4.21574]
[ 4.21574]	[ 4.41523]
[ 4.41523]	[ 4.61472]
[ 4.61472]	[ 4.81421]
[ 4.81421]	[ 5.0137]
[ 5.0137]	[ 5.21319]
[ 5.21319]	[ 5.41268]
[ 5.41268]	[ 5.61217]
[ 5.61217]	[ 5.81166]
[ 5.81166]	[ 6.01115]
[ 6.01115]	[ 6.21064]
[ 6.21064]	[ 6.41013]
[ 6.41013]	[ 6.60962]
[ 6.60962]	[ 6.80911]
[ 6.80911]	[ 7.0086]
```
