from scipy.signal import hilbert
import matplotlib.pyplot as plt
import numpy as np

plt.style.use('seaborn-poster')


def get_envelope(x, n=None):
    """use the Hilbert transform to determine the amplitude envelope.
    Parameters:
    x : ndarray
        Real sequence to compute  amplitude envelope.
    N : {None, int}, optional, Number of Fourier components. Default: x.shape[axis]
        Length of the hilbert.

    Returns:
    amplitude_envelope: ndarray
        The amplitude envelope.

    """

    analytic_signal = hilbert(x, N=n)
    amplitude_envelope = np.abs(analytic_signal)
    return amplitude_envelope


x = np.linspace(0, 20, 201)
y = np.sin(x)
amplitude_envelope = get_envelope(y)
plt.figure(figsize=(8, 6))
plt.plot(x, y, label='signal')
plt.plot(x, amplitude_envelope, label='envelope')
plt.ylabel('Amplitude')
plt.xlabel('Location (x)')
plt.legend()

plt.show()
