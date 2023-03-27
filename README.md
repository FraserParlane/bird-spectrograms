# Spectrograms

Spectrograms plot a spectrum's **amplitude** as a function of **time** (x-axis) and **frequency** (y-axis). Audio spectrograms, for example, reveal the underlying component frequencies of a sound. The combined signal is decomposed into the component frequencies via Fast Fourier Transform (FFT) operating on a moving window across the time series.

Note that there is a trade-off between frequency and time resolution. A large temporal window results in high frequency resolution and low temporal resolution. The opposite is also true: a small temporal window results in low frequency resolution and high temporal resolution.

Below are some examples of how Python can be used to visualize these spectrograms.


[![Watch the video](https://img.youtube.com/vi/sHFS9C0AFZ0/default.jpg)](https://youtu.be/sHFS9C0AFZ0)