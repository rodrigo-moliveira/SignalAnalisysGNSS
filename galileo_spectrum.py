from signal_utils import *

"""
Galileo Signals

    * E1 band (center frequency 1575.420 MHz)
        - E1 OS Data & Pilot with CBOC(6,1,1/11) modulation
        - E1 PRS with cosine BOC(15,2.5) modulation
        
    * E5 band (center frequency 1191.795 MHz)
        - E5a Data & Pilot with AltBOC(15,10) modulation
        - E5b Data & Pilot with AltBOC(15,10) modulation
        
    * E6 band (center frequency 1278.750 MHz)
        - E6 CS Data & Pilot with BPSK(5) modulation
        - E6 PRS with cosine BOC(10,5) modulation
        
"""


def plot(freqs, psd, label="", title="", ax=None):
    # plot data
    if ax is None:
        fig, ax = plt.subplots()
    ax.plot(freqs, psd, label=label)
    ax.set_ylim((-100, -50))
    ax.set_title(title)
    ax.set_ylabel("PSD [dBW/Hz]")
    ax.set_xlabel("Frequency Offset with respect to the carrier [MHz]")
    ax.legend()
    ax.grid(visible=True)

    return ax


def plot_E1():
    """
    E1 band (center frequency 1575.420 MHz)
        - E1 OS Data & Pilot with CBOC(6,1,1/11) modulation
        - E1 PRS with cosine BOC(15,2.5) modulation
    """
    cboc_f, cboc_psd = psd_cboc(6, 1, 1/11)
    boc_f, boc_psd = psd_boc(15, 2.5, cos=True)

    ax = plot(cboc_f, cboc_psd, label="E1 OS Data & Pilot with CBOC(6,1,1/11) modulation")
    plot(boc_f, boc_psd, label="E1 PRS with cosine BOC(15,2.5) modulation", title="Galileo E1 Band", ax=ax)


def plot_E6():
    """
    E6 band (center frequency 1278.750 MHz)
        - E6 CS Data & Pilot with BPSK(5) modulation
        - E6 PRS with cosine BOC(10,5) modulation
    """
    bpsk_f, bpsk_psd = psd_bpsk(5)
    boc_f, boc_psd = psd_boc(10, 5, cos=True)

    ax = plot(bpsk_f, bpsk_psd, label="E6 CS Data & Pilot with BPSK(5) modulation")
    plot(boc_f, boc_psd, label="E6 PRS with cosine BOC(10,5) modulation", title="Galileo E6 Band", ax=ax)


def plot_E5():
    """
    E5 band (center frequency 1191.795 MHz)
        - E5a Data & Pilot with AltBOC(15,10) modulation
        - E5b Data & Pilot with AltBOC(15,10) modulation
    """
    altboc_f, altboc_psd = psd_altboc(15, 10)
    plot(altboc_f, altboc_psd, label="E5a + E5b Data & Pilot with AltBOC(15,10) modulation",
         title="Galileo E5 Band")


def main():

    plot_E1()

    plot_E6()

    plot_E5()

    show()


if __name__ == '__main__':
    main()
