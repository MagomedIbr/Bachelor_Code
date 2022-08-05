from biosppy import storage
import scipy
from biosppy.signals import emg

# load raw ECG signal
signal, mdata = storage.load_txt('e07_002_001_0348.bdf')

# process it and plot
out = emg.emg(signal=signal, sampling_rate=1000., show=True)