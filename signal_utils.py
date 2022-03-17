import numpy as np
from matplotlib import pyplot as plt

f0 = 1.023E6  # reference (fundamental) chip rate, in Hz


def psd_bpsk(n):
    # chip rate (frequency), in Hz
    fc = n * f0

    # f -> chip rate
    Tc = 1.0 / fc  # chip period

    # frequency range from -100 MHz to 100 MHz
    freqs = np.arange(-20, 20, 0.1)  # in MHz
    _freqs = freqs * 10 ** 6

    # generate power spectral density (psd)
    # psd = Tc * np.sinc(freqs * 10 ** 6 * Tc) ** 2
    psd = fc * np.sin(np.pi * _freqs / fc) ** 2 / (np.pi * _freqs) ** 2

    psd = 10 * np.log10(psd)

    return freqs, psd


def boc(n_s, n_c, cos=False):
    # get BOC ratio integer
    n = int(2 * n_s / n_c)

    # check if integer ratio is even or odd
    even = True if n % 2 == 0 else False

    # frequency range from -100 MHz to 100 MHz
    freqs = np.arange(-20, 20, 0.01)  # in MHz
    _freqs = freqs * 10 ** 6

    if cos is False:
        # Sine BOC (default)

        if even is True:
            # 2 * n_s / n_c is even
            psd = n_c * f0 * np.sin(np.pi * _freqs / n_c / f0) ** 2 / (np.pi * _freqs) ** 2 * np.tan(
                np.pi * _freqs / 2 / n_s / f0) ** 2

        else:
            # 2 * n_s / n_c is odd
            psd = n_c * f0 * np.cos(np.pi * _freqs / n_c / f0) ** 2 / (np.pi * _freqs) ** 2 * np.tan(
                np.pi * _freqs / 2 / n_s / f0) ** 2

    else:
        _tmp = np.cos(np.pi * _freqs / 2 / n_s / f0)

        # Cosine BOC
        if even is True:
            # 2 * n_s / n_c is even
            psd = n_c * f0 * np.sin(np.pi * _freqs / n_c / f0) ** 2 / (np.pi * _freqs) ** 2 * (
                    1 - _tmp) ** 2 / _tmp ** 2

        else:
            # 2 * n_s / n_c is odd
            psd = n_c * f0 * np.cos(np.pi * _freqs / n_c / f0) ** 2 / (np.pi * _freqs) ** 2 * (
                    1 - _tmp) ** 2 / _tmp ** 2
    return psd, freqs


def psd_boc(n_s, n_c, cos=False):
    psd, freqs = boc(n_s, n_c, cos)

    # convert psd to dBW
    psd = 10 * np.log10(psd)

    return freqs, psd


def psd_cboc(n_s, n_c, ratio):
    psd_1, freqs = boc(1, 1, False)
    psd_2, freqs = boc(n_s, n_c, False)

    psd = (1 - ratio) * psd_1 + ratio * psd_2

    # convert psd to dBW
    psd = 10 * np.log10(psd)

    return freqs, psd


def psd_altboc(n_s, n_c):
    # get BOC ratio integer
    n = int(2 * n_s / n_c)

    # check if integer ratio is even or odd
    even = True if n % 2 == 0 else False

    # frequency range from -100 MHz to 100 MHz
    freqs = np.arange(-45, 45, 0.01)  # in MHz
    _freqs = freqs * 10 ** 6

    if even is True:
        # 2 * n_s / n_c is even
        psd = 8 * n_c * f0 * np.sin(np.pi * _freqs / n_c / f0) ** 2 / (
                np.pi * _freqs) ** 2 / (np.cos(np.pi * _freqs / 2 / n_s / f0)) ** 2 * (
                      1 - np.cos(np.pi * _freqs / 2 / n_s / f0))

    else:
        # 2 * n_s / n_c is odd
        psd = 8 * n_c * f0 * np.cos(np.pi * _freqs / n_c / f0) ** 2 / (
                np.pi * _freqs) ** 2 / (np.cos(np.pi * _freqs / 2 / n_s / f0)) ** 2 * (
                      1 - np.cos(np.pi * _freqs / 2 / n_s / f0))

    # convert psd to dBW
    psd = 10 * np.log10(psd)

    return freqs, psd


def show():
    plt.show()


""""""""""""""""""""""""""""""
""" PSD function callables """
""""""""""""""""""""""""""""""


def call_bpsk(f, n):
    # Binary Phase Shift Keying PSD function (callable)

    # chip rate (frequency), in Hz
    fc = n * f0
    return fc * np.sin(np.pi * f / fc) ** 2 / (np.pi * f) ** 2


def call_boc(f, cos, n_s, n_c):

    # get BOC ratio integer
    n = int(2 * n_s / n_c)

    # check if integer ratio is even or odd
    even = True if n % 2 == 0 else False

    if cos is False:
        # Sine BOC (default)

        if even is True:
            # 2 * n_s / n_c is even
            psd = n_c * f0 * np.sin(np.pi * f / n_c / f0) ** 2 / (np.pi * f) ** 2 * np.tan(
                np.pi * f / 2 / n_s / f0) ** 2

        else:
            # 2 * n_s / n_c is odd
            psd = n_c * f0 * np.cos(np.pi * f / n_c / f0) ** 2 / (np.pi * f) ** 2 * np.tan(
                np.pi * f / 2 / n_s / f0) ** 2

    else:
        _tmp = np.cos(np.pi * f / 2 / n_s / f0)

        # Cosine BOC
        if even is True:
            # 2 * n_s / n_c is even
            psd = n_c * f0 * np.sin(np.pi * f / n_c / f0) ** 2 / (np.pi * f) ** 2 * (
                    1 - _tmp) ** 2 / _tmp ** 2

        else:
            # 2 * n_s / n_c is odd
            psd = n_c * f0 * np.cos(np.pi * f / n_c / f0) ** 2 / (np.pi * f) ** 2 * (
                    1 - _tmp) ** 2 / _tmp ** 2

    return psd
