import numpy as np
from matplotlib import pyplot as plt
from scipy.integrate import quad

from signal_utils import call_boc, call_bpsk, f0

pi = np.pi
SPEED_OF_LIGHT = 299792458  # [m/s]


def plot(x, y, label="", xlabel="", ylabel="", title="", ax=None):
    # plot data
    if ax is None:
        fig, ax = plt.subplots()
    ax.plot(x, y, label=label)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    if label != "":
        ax.legend()
    ax.grid(visible=True)

    return ax


def show():
    plt.show()


def unit_tests():
    print("Unit Tests:")
    print("Integral of PSD from -inf to inf in frequency domain must be 1, since these functions must be normalized\n"
          "to unit area.")

    f_bpsk = lambda x: call_bpsk(x, 1)
    y = quad(f_bpsk, -10000000, 0)
    print(f"Integrating BPSK PSD from -inf to 0. Expected/theoretical value = 0.5. Quadrature = {y[0]}")

    f_boc = lambda x: call_boc(x, False, 1, 1)
    y = quad(f_boc, -10000000, 0)
    print(f"Integrating BOC PSD from -inf to 0. Expected/theoretical value = 0.5. Quadrature = {y[0]}")


def dll_measurement_error(psd_fun, Tc, Bfe, Bn, T, D, C_N_dB) -> float:
    """
    implementing Eq. 8.87 of Kaplan : Thermal noise code tracking error for non-coherent DLL discriminator

    Variables:
        Tc : Chip period [s]
        Bfe : Double-sided front-end bandwidth [Hz]
        psd_fun : power spectral density function of f, normalized to unit area over infinite bandwidth. See unit tests.
        Bn : Code loop noise bandwidth [Hz]
        T : pre-detection integration time [s]
        D : early-to-late correlator spacing (chips).
    """

    C_N_linear = 10 ** (C_N_dB / 10)

    f1_num = lambda f: psd_fun(f) * np.sin(pi * f * D * Tc) ** 2
    f1_den = lambda f: f * psd_fun(f) * np.sin(pi * f * D * Tc)
    f2_num = lambda f: psd_fun(f) * np.cos(pi * f * D * Tc) ** 2
    f2_den = lambda f: psd_fun(f) * np.cos(pi * f * D * Tc)

    # Perform quadratures. Separate the intervals to avoid numerical problems
    y1_num = quad(f1_num, -Bfe / 2, 0)[0] + quad(f1_num, 0, Bfe / 2)[0]
    y1_den = quad(f1_den, -Bfe / 2, 0)[0] + quad(f1_den, 0, Bfe / 2)[0]
    y2_num = quad(f2_num, -Bfe / 2, 0)[0] + quad(f2_num, 0, Bfe / 2)[0]
    y2_den = quad(f2_den, -Bfe / 2, 0)[0] + quad(f2_den, 0, Bfe / 2)[0]

    sigma = 1 / Tc * np.sqrt(Bn * y1_num / ((2 * pi) ** 2 * C_N_linear * y1_den ** 2)) * np.sqrt(
        1 + y2_num / (T * C_N_linear * y2_den ** 2))

    # sigma in meters
    sigma *= Tc * SPEED_OF_LIGHT

    return sigma


def main():

    ########################
    #       BPSK(n)        #
    ########################
    n = 5
    fc = n * f0

    Tc = 1.0 / fc  # chip period [s]
    Bfe = 1.7E6  # typical value for L1-L2 signals
    Bn = 0.1
    psd_fun = lambda f: call_bpsk(f, n)
    T = 30E-3
    D = 1
    C_N = np.arange(10, 30, 1)

    sigma = [dll_measurement_error(psd_fun, Tc, Bfe, Bn, T, D, x) for x in C_N]
    plot(C_N, sigma, xlabel="C/N0 [dB-Hz]", ylabel="DLL Error [m]", title=f"Delay Lock Loop Error for BPSK({n})")


    ########################
    #    BOC(n_s,n_c)      #
    ########################

    n_s = 10
    n_c = 5
    fc = n_c * f0

    Tc = 1.0 / fc  # chip period [s]
    Bfe = 3E7  # typical value for L1-L2 signals
    Bn = 0.1
    psd_fun = lambda f: call_boc(f, False, n_s, n_c)
    T = 0.01
    D = 1/8
    C_N = np.arange(10, 30, 1)

    sigma = [dll_measurement_error(psd_fun, Tc, Bfe, Bn, T, D, x) for x in C_N]
    plot(C_N, sigma, xlabel="C/N0 [dB-Hz]", ylabel="DLL Error [m]", title=f"Delay Lock Loop Error for BOC({n_s},{n_c})")

    show()


if __name__ == '__main__':
    # unit_tests()

    main()
