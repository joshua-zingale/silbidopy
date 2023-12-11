# silbidopy
*[silbido](https://github.com/MarineBioAcousticsRC/silbido)* is a MATLAB tool that allows for both automatic- and expert-generated tonal annotations that are exported in a binary format. This repository contains a Python package that allows these binary files to be read into Python and also for Python to generate these binary files. Also, there is functionality for rendering annotations.

# Files
- README.md | text | information about this repository
- silbidopy | folder | the Python package
   - \_\_init\_\_.py | python | boiler plate to create a package
   - readBinaries.py | python | code to read *silbido* files
   - render.py | python | code to render *silbido* annotations
   - sigproc.py | python | helper functions for signal processing
   - writeBinaries.py | python | code to write *silbido* files

# Use
The package has two files that can be imported directly into Ptyhon code, `readBinaries` has code to read the *silbido* format and `writeBinaries` has code to write binary files in the *silbido* format.

## Reading files
The follow code snippet loads the annotations from a *silbido* annotation file as an array containing the time-frequency information for each labeled tonal.
```python
from silbidopy.readBinaries import tonalReader

# Initialize the reader
tr = tonalReader("annotations.ann")

# Read the entire file, saving only the time-frequency information
contours = tr.getTimeFrequencyContours()

# Get the third node from the second tonal, which will be a
# (time-in-seconds, frequency-in-hz) tuple of floating point values,
# e.g. (75.025, 65.125)
node = contours[2][3]
```

Alternatively, if more information from each tonal is desired, the reader may be iterated through to receive one tonal at a time as a dictionary.
```python
from silbidopy.readBinaries import tonalReader

# Initialize the reader
tr = tonalReader("annotations.ann")

# Iterate through the reader to get each tonal
for tonal in tr:
 print(tonal)
```
One such output dictionary may be rendered thus:
```python
{'species': 'Balaenoptera musculus', 'call': 'D', 'graphId': 18446744073709551615, 'confidence': 0.0, 'score': 0.0, 'tfnodes': [{'time': 685.0, 'freq': 65.45902326742953, 'snr': None, 'phase': None, 'ridge': False}, {'time': 685.1, 'freq': 60.264037864371126, 'snr': None, 'phase': None, 'ridge': False}, {'time': 685.2, 'freq': 55.93661795036039, 'snr': None, 'phase': None, 'ridge': False}, {'time': 685.3, 'freq': 52.47676352540076, 'snr': None, 'phase': None, 'ridge': False}, {'time': 685.4, 'freq': 49.88447458948437, 'snr': None, 'phase': None, 'ridge': False}, {'time': 685.5, 'freq': 48.15975114261565, 'snr': None, 'phase': None, 'ridge': False}, {'time': 685.6, 'freq': 47.302593184794596, 'snr': None, 'phase': None, 'ridge': False}]}
```

## Writing files
The binary writer currently leaves all fields blank except for a dummy value that is used as the graphId and the times and frequencies for each tonal. The following code first reads in an annotation file. Then, all but the first three annotations are dropped and a new binary *silbido* file is written that contains only the these first three annotations.
```python
from silbidopy.readBinaries import tonalReader
from silbidopy.writeBinaries import writeTimeFrequencyBinary

# Initialize the reader
tr = tonalReader("annotations.ann")

# Read the entire file, saving only the time-frequency information
contours = tr.getTimeFrequencyContours()

# Keep only the first three tonals
trimmed_contours = contours[:3]

# Write these three tonals to a new annotation file
writeTimeFrequencyBinary("trimmed-annotations.ann", trimmed_contours)
```

## Displaying Spectrograms & Annotations
There are two functions in `render` for making displays.
These require [NumPy](https://pypi.org/project/numpy/) and and [wavio](https://pypi.org/project/wavio/).

To generate a spectrogram, use `render.getSpectrogram`.
This requires either the name of an audio file or a `wavio.Wav` object.
The function also accepts many optional parameters to specify how the spectrogram will be generated.
The output is a two-dimensional numpy array and each entry is a magnitude corresponding to a time and frequency,
where frequency varies along axis 0 and time varies along axis 1.
There is also a second value returned, which gives the actual length,
in miliseconds, that the spectrogram covers, which will always match the input end_time unless the end time
is after the file ends.
This can help with generating a corresponding annotation mask.

Here is an example program that generates a spectrogram:
```python
from silbidopy.render import getSpectrogram 
import wavio

wav_data = wavio.read('wav-file.wav')
 
spectrogram, _ = getSpectrogram(wav_data, start_time = 0, end_time = 8000)

print(spectrogram.shape)
# Output: (360, 4000)
```

To generate an annotation mask, use `render.getAnnotationMask`.
This requires a list of annotations such as is returned by `readBinaries.tonalReader.getTimeFrequencyContours`.
The output is a two-dimensional numpy array and each entry is either 1, meaning that there is tonal energy, or 0,
meaning that there is no tonal energy, corresponding to a time and frequency.
The frequency varies along axis 0 and time varies along axis 1.

Given analogous arguments, `render.getAnnotationMask` will generate a mask wherein each entry directly corresponds
to a spectrogram generated a `render.getSpectrogram`.

Here is an example of a program that generates a mask that corresponds directly to the spectrogram generated above:
```python
from silbidopy.readBinaries import tonalReader
from silbidopy.render import getAnnotationMask

# Get annotations that go with wave-file.wav
contours = tonalReader("wav-file-annotations.bin")

mask = getAnnotation(contours, start_time = 0, end_time = 8000)

print(mask.shape)
# Output: (360, 4000)
```

To view the spectrogram or the annotation mask, [matplotlib](https://pypi.org/project/matplotlib/) could be used,
for example with `pyplot.imshow`.
