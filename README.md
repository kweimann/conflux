# conflux
time series utility library for Python

### usage

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
linearly_interp = time_series.interpolate(interval=1, start_timestamp=t0,
                                          end_timestamp=tn, method="linear")

# interpolate time series using `most recent` strategy starting from `t0` until `tn`
most_recent_interp = time_series.interpolate(interval=1, start_timestamp=t0,
                                             end_timestamp=tn, method="most_recent")

plt.plot(xs, f(xs), label="f(x)")
plt.plot(timestamps, observations, 'o', label="observations")
plt.plot(linearly_interp.timestamps, linearly_interp.observations, label='linear interp')
plt.plot(most_recent_interp.timestamps, most_recent_interp.observations, label='most recent interp')
plt.legend()
plt.show()

```

![example](https://user-images.githubusercontent.com/8287691/37554460-5cb060a0-29d9-11e8-8dfc-c36cc8945e69.png)

for complete example see: `example.py`
