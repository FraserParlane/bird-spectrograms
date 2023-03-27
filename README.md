# If a bird wrote down its song as sheet music, what would it look like?

<a href="https://youtu.be/BFwcstHAd04" target="_blank">
    <img src="readme/bird-video.png">
</a>

# What am I looking at?

A spectrogram is a plot a spectrum's **amplitude** (as a color scale) as a function of **time** (x-axis) and **frequency** (y-axis). A spectrogram reveals the underlying frequencies that compose the audio signal. The audio signal is decomposed into the component frequencies with a Fast Fourier Transform (FFT). A moving window is passed across the time series data, and the FFT is calculated on each window.

Note that there is a trade-off between frequency and time resolution. A large temporal window results in high frequency resolution and low temporal resolution. The opposite is also true: a small temporal window results in low frequency resolution and high temporal resolution.

[//]: # ([![Watch the video]&#40;https://img.youtube.com/vi/sHFS9C0AFZ0/default.jpg&#41;]&#40;https://youtu.be/sHFS9C0AFZ0&#41;)