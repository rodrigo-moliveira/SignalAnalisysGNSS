import numpy as np
from matplotlib import pyplot as plt
from scipy.integrate import quad

from signal_utils import call_boc, call_bpsk, f0, call_cboc, call_altboc

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

    # Receiver stats:
    Bfe_E1 = 16E6  # typical value for L1 and L2 signals
    Bfe_E5 = 50E6
    Bfe_E6 = 40E6
    Bn = 0.1
    T = 30E-3
    D = 1

    ###########################################################################
    #       E1 band (OS uses CBOC(6,1,1/11), PRS uses cos BOC(15,2.5))        #
    ###########################################################################

    # -> CBOC(6,1,1/11)
    n_s = 6
    n_c = 1
    ratio = 1/11

    fc = n_c * f0
    Tc = 1.0 / fc  # chip period [s]
    psd_fun = lambda f: call_cboc(f, n_s, n_c, ratio)
    C_N = np.arange(20, 45, 1)
    sigma = [dll_measurement_error(psd_fun, Tc, Bfe_E1, Bn, T, D, x) for x in C_N]
    plot(C_N, sigma, xlabel="C/N0 [dB-Hz]", ylabel="DLL Error [m]",
         title=f"Delay Lock Loop Error for CBOC(6,1,1/11) - E1 OS")

    # -> cos BOC(15,2.5)
    n_s = 15
    n_c = 2.5

    fc = n_c * f0
    Tc = 1.0 / fc  # chip period [s]
    psd_fun = lambda f: call_boc(f, True, n_s, n_c)
    C_N = np.arange(20, 45, 1)
    sigma = [dll_measurement_error(psd_fun, Tc, 80*f0, Bn, T, D, x) for x in C_N]
    plot(C_N, sigma, xlabel="C/N0 [dB-Hz]", ylabel="DLL Error [m]",
         title=f"Delay Lock Loop Error for cos BOC(15,2.5)) - E1 PRS")

    #

    ###########################################
    #       E5 band uses AltBOC(15,10)        #
    ###########################################
    n_s = 15
    n_c = 10

    fc = n_c * f0
    Tc = 1.0 / fc  # chip period [s]
    psd_fun = lambda f: call_altboc(f, n_s, n_c)
    C_N = np.arange(20, 45, 1)
    sigma = [dll_measurement_error(psd_fun, Tc, 70 * f0, Bn, T, D, x) for x in C_N]
    plot(C_N, sigma, xlabel="C/N0 [dB-Hz]", ylabel="DLL Error [m]",
         title=f"Delay Lock Loop Error for AltBOC(15,10) - E5 PRS")

    #

    #####################################################################
    #       E6 band (CS uses BPSK(5) and PRS uses cos BOC(10,5))        #
    #####################################################################
    # -> BPSK(5)
    n = 5
    fc = n * f0

    Tc = 1.0 / fc  # chip period [s]
    Bn = 0.1
    psd_fun = lambda f: call_bpsk(f, n)
    T = 30E-3
    D = 1
    C_N = np.arange(10, 45, 1)

    sigma = [dll_measurement_error(psd_fun, Tc, Bfe_E6, Bn, T, D, x) for x in C_N]
    plot(C_N, sigma, xlabel="C/N0 [dB-Hz]", ylabel="DLL Error [m]",
         title=f"Delay Lock Loop Error for BPSK(5) - E6 CS")

    # -> cos BOC(10,5)
    n_s = 10
    n_c = 5
    fc = n_c * f0

    Tc = 1.0 / fc  # chip period [s]
    Bn = 0.1
    psd_fun = lambda f: call_boc(f, True, n_s, n_c)
    T = 0.01
    D = 1 / 8
    C_N = np.arange(10, 45, 1)

    sigma = [dll_measurement_error(psd_fun, Tc, Bfe_E6, Bn, T, D, x) for x in C_N]
    plot(C_N, sigma, xlabel="C/N0 [dB-Hz]", ylabel="DLL Error [m]",
         title=f"Delay Lock Loop Error for cos BOC(10,5) - E6 PRS")

    #

    show()


if __name__ == '__main__':
    # unit_tests()

    main()
