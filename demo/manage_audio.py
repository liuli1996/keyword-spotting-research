import numpy as np
import librosa

def dct(n_filters, n_input):
    """Discrete cosine transform (DCT type-III) basis.

    .. [1] http://en.wikipedia.org/wiki/Discrete_cosine_transform

    Parameters
    ----------
    n_filters : int > 0 [scalar]
        number of output components (DCT filters)

    n_input : int > 0 [scalar]
        number of input components (frequency bins)

    Returns
    -------
    dct_basis: np.ndarray [shape=(n_filters, n_input)]
        DCT (type-III) basis vectors [1]_

    Notes
    -----
    This function caches at level 10.

    """

    basis = np.empty((n_filters, n_input))
    basis[0, :] = 1.0 / np.sqrt(n_input)

    samples = np.arange(1, 2*n_input, 2) * np.pi / (2.0 * n_input)

    for i in range(1, n_filters):
        basis[i, :] = np.cos(i*samples) * np.sqrt(2.0/n_input)

    return basis

class AudioPreprocessor():
    def __init__(self, sr=16000, n_dct_filters=40, n_mels=40, f_max=4000, f_min=20, n_fft=480, hop_ms=10):
        super().__init__()
        self.n_mels = n_mels
        self.dct_filters = dct(n_dct_filters, n_mels)
        self.sr = sr
        self.f_max = f_max if f_max is not None else sr // 2
        self.f_min = f_min
        self.n_fft = n_fft
        self.hop_length = sr // 1000 * hop_ms

    def compute_mfccs(self, data):
        data = librosa.feature.melspectrogram(
            data,
            sr=self.sr,
            n_mels=self.n_mels,
            hop_length=self.hop_length,
            n_fft=self.n_fft,
            fmin=self.f_min,
            fmax=self.f_max)
            # pad_mode='reflect')
        data[data > 0] = np.log(data[data > 0])
        data = [np.matmul(self.dct_filters, x) for x in np.split(data, data.shape[1], axis=1)]
        data = np.array(data, order="F").astype(np.float32)
        return data

if __name__ == "__main__":
    pass
