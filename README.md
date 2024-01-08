# silbidopy
*[silbido](https://github.com/MarineBioAcousticsRC/silbido)* is a MATLAB tool that has abilities for both automatic- and expert-generated tonal
annotations to be exported in a binary format. This repository contains a Python package that allows these binary files to be read into
Python and also for Python to generate these binary files. Also, there is functionality for rendering annotations and for loading audio and annotations as a PyTorch datset.

# Files
- README.md | text | information about this repository
- silbidopy | folder | the Python package
   - \_\_init\_\_.py | python | boiler plate to create a package.
   - data.py | python | code to load spectrograms & annotations as a PyTorch dataset.
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
These require [NumPy](https://pypi.org/project/numpy/) and [wavio](https://pypi.org/project/wavio/).

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

## PyTorch Dataset
A spectrogram could be used as an input to a neural network and the ground truth, in a task similar to segmentation,
could be the corresponding annotation mask. `data` contains a PyTorch dataset class that can be used with the Pytorch DataLoader
to load in such spectrogram-annotation-mask pairs.
Importing `data` requires [h5py](https://pypi.org/project/h5py/), [NumPy](https://pypi.org/project/numpy/), [PyTorch](https://pypi.org/project/torch/), and [wavio](https://pypi.org/project/wavio/).

`data.AudioTonalDataset` works by reading from the audio and annotation files as needed, dynamically generating the spectrograms
and annotation masks as they are requested. Hence, the dataset requires constant access to specified wav and binary-annotation files.

```python
from silbidopy.data import AudioTonalDataset
from torch.utils.data import DataLoader

# Create a dataset that will dynamically load spectrograms and annotation masks
dataset = AudioTonalDataset(
 "path-to-folder-with-audio-files",
 "path-to-folder-with-annotation-files",
 ... # any spectrogram paremeters
)

# Access an item
# This creates the spectrogram and mask
spectrogram, mask = dataset[777]
```

A `data.AudioTonalDataset` is typically not balanced because there are usually more examples without tonal energy.
There is a method to generate a new dataset that will have an even number of data that have annotations with at least some tonal
energy, `get_balanced_dataset`.

```python
from silbidopy.data import AudioTonalDataset

# Basic dataset
dataset = AudioTonalDataset(...)

# Get a new dataset wherein 55% of the indices will correspond to
# a spectrogram with some tonal energy
dataset = dataset.get_balanced_dataset(positive_proportion = 0.55)
```

A `data.AudioTonalDataset` has a sizeable overhead when loading data because the spectrograms and annotation masks are dynamically created for each datum load.
Therefore, for a big speed increase in datum load time, the function `data.dataset_to_hdf5` may be used to export a `data.AudioTonalDataset` to an hdf5 file.
There is also another dataset, `data.Hdf5Dataset`, which may accesses an hdf5 file to serve data.

```python
from silbidopy.data import (
 AudioTonalDataset,
 dataset_to_hdf5,
 Hdf5Dataset)
 

# Basic dataset
dataset = AudioTonalDataset(...)

# Save dynamic dataset to static hdf5 file
dataset_to_hdf5(dataset, "static.hdf5")

# Create dataset from the hdf5 file
h5_dataset = Hdf5Dataset("static.hdf5")
```

