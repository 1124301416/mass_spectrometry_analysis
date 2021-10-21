import numpy

from backend.mass_spectrometry_tools import to_number, is_float, rename_file

y = '15 helo'
print(next((x for x,val in enumerate(y) if val > 3)))