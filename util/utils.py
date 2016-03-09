import scipy.io.wavfile as wav


def save_wavfile(signal, file_prefix, rate=16000):
    num_samples = signal.shape[0]
    time_length = signal.shape[1]

    for s in xrange(num_samples):
        file_path = file_prefix+'_{}.wav'.format(s)
        wav.write(file_path, rate, signal[s][:])