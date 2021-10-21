from os import listdir, path, rename

import numpy

from backend.mass_spectrometry_tools import MassSpectrometryTools, rename_file, refactor_name

abs_path = r'sample_data/2021-10-18'

names = refactor_name(abs_path)

data = MassSpectrometryTools(names)

data.plot_ratio_raw_data()

print('done')
