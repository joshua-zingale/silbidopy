from silbidopy.render import getSpectrogram, getAnnotationMask
from silbidopy.readBinaries import tonalReader
from torch.utils.data import Dataset
import numpy as np
import fnmatch
import wavio
import os

class AudioTonalDataset(Dataset):
    def __init__(self, audio_dir, annotation_dir, frame_time_span = 8, step_time_span = 2,
                 spec_clip_min = 0, spec_clip_max = 6, min_freq = 5000, max_freq = 50000,
                 time_patch_frames = 64, freq_patch_frames = 64, time_patch_advance = 64,
                 freq_patch_advance = 64, cache_wavs = True):
        '''
        A Dataset that pulls spectrogram's and tonal annotations from audio and annotation
        files respectively. Each datum is one patch from one spectrogram representation of one of the audio files.
        The dimensionality for each patch is (freq_patch_frames, time_patch_frames). The patches may overlap depending
        on the time and frequency patch advance parameters. Each datum's corresponding label is an array of equal size
        to the datum, two-dimensional, containing zeros where no whistle energy lies and ones where whistle energy does
        lie according to the annotation files.

        :param audio_dir: the file path where the audio files are stored
        :param annotation_dir: the file path where the annotation files corresponding to
                                    the audio files are stored. For each annotation file,
                                    there must be a corresponding audio file with the same
                                    name but with a different extension, namely .wav 
        :param frame_time_span: ms, length of time for one time window for dft
        :param step_time_span: ms, length of time step for spectrogram
        :param spec_clip_min: log magnitude spectrogram min-max normalization, minimum value
        :param spec_clip_max: log magnitude spectrogram min-max normalization, maximum value
        :param min_freq: Hz, lower bound of frequency for spectrogram
        :param max_freq: Hz, upper bound of frequency for spectrogram
        :param time_patch_frames: the number of time frames per ouput datum, i.e. the number per patch.

        :param cache_wavs: If True, all wav files are saved in memory;
                           else, get datum acces opens and closes a
                           wav file.

        :returns: A tuple with both the spectrogram and the time at which the
                    spectrogram ended in ms: (spectogram, end_time)
        '''
        
        ## COLLECT AUDIO AND ANNOTATIONS ##
        # colelct all .wav files
        wav_files = findfiles(audio_dir, "*.wav")
        
        # collect all .wav filenames
        wav_names = list(map(os.path.basename, wav_files))
    
        # map each file name to its full file path
        wav_file_dict = {wav_names[i] : wav_files[i] for i in range(len(wav_names))}
        
        # Get all annotation file paths
        bin_files = findfiles(annotation_dir, "*.bin")

        # find all .wav with corresponding .bin
        anno_wav_filenames = list(map(bin2wav_filename, bin_files))

        try:
            anno_wav_files = [wav_file_dict[filename] for filename in anno_wav_filenames]
        except KeyError as ex:
            raise Exception(f"Could not find audio file {str(ex)} corresponding to binary file.")
       
       ## SAVE DATASET VALUES ##
        freq_resolution = 1000 / frame_time_span
        self.patch_freq_length_hz = freq_resolution * freq_patch_frames
        self.freq_patch_advance_hz = freq_resolution * freq_patch_advance
        self.patch_time_length_ms = step_time_span * time_patch_frames
        self.time_patch_advance_ms = step_time_span * time_patch_advance

        self.frame_time_span = frame_time_span
        self.step_time_span = step_time_span
        self.spec_clip_min = spec_clip_min
        self.spec_clip_max = spec_clip_max
        self.min_freq = min_freq
        self.max_freq = max_freq
        self.time_patch_frames = time_patch_frames
        self.freq_patch_frames = freq_patch_frames
        self.time_patch_advance = time_patch_advance
        self.freq_patch_advance = freq_patch_advance
        
        self.cache_wavs = cache_wavs

        self.bin_files = bin_files
        self.anno_wav_files = anno_wav_files

        # Get length of dataset
        self.num_patches = []
        self.file_info = []
        for wav_file in (wavio.read(w) for w in anno_wav_files):
            num_samples = wav_file.data.shape[0]

            file_length_ms = num_samples * 1000 / wav_file.rate

            # determine & append number of patches in file
            num_freq_divisions = int((max_freq - self.patch_freq_length_hz) / self.freq_patch_advance_hz)

            num_time_divisions = int((file_length_ms - self.patch_time_length_ms - frame_time_span) / self.time_patch_advance_ms)

            self.num_patches.append(num_freq_divisions * num_time_divisions)

            if cache_wavs:
                self.file_info.append({
                    "num_time_divisions": num_time_divisions,
                    "wav_data": wav_file,
                    })
            else:
                self.file_info.append({
                    "num_time_divisions": num_time_divisions,
                    })



    def __len__(self):
        return sum(self.num_patches)

    def __getitem__(self, idx):
        ## Determine which file corresponds to idx ##
        if idx >= len(self):
            raise IndexError("Dataset index out of bounds.")
        patches_cumsum = np.cumsum(self.num_patches)
        file_idx = np.argmax(patches_cumsum > idx)

        # Adjust idx to be relative to the file index
        if file_idx != 0:
            idx -= patches_cumsum[file_idx - 1]
        
        # get starting time and frequency
        num_time_divisions = self.file_info[file_idx]["num_time_divisions"]
        start_time = (idx % num_time_divisions) * self.time_patch_advance_ms
        start_freq = (idx // num_time_divisions) * self.freq_patch_advance_hz + self.min_freq

        end_time = start_time + self.patch_time_length_ms
        end_freq = start_freq + self.patch_freq_length_hz
       
        # get wav file
        wav_data = None
        if self.cache_wavs:
            wav_data = self.file_info[file_idx]["wav_data"]
        else:
            wav_data = wavio.read(self.anno_wav_files[file_idx])

        # get datum (spectrogram) and label
        datum, _ = getSpectrogram(wav_data,
                frame_time_span = self.frame_time_span,
                step_time_span = self.step_time_span,
                spec_clip_min = self.spec_clip_min,
                spec_clip_max = self.spec_clip_max,
                min_freq = start_freq, max_freq = end_freq,
                start_time = start_time, end_time = end_time)

        contours = tonalReader(self.bin_files[file_idx]).getTimeFrequencyContours()
        
        label = getAnnotationMask(contours,
                frame_time_span = self.frame_time_span,
                step_time_span = self.step_time_span,
                min_freq = start_freq, max_freq = end_freq,
                start_time = start_time, end_time = end_time)

        return datum, label

# find file  with certain pattern.
def findfiles(path, fnmatchex='*.*'):
    result = []
    for root, dirs, files in os.walk(path):
        for filename in fnmatch.filter(files, fnmatchex):
            fullname = os.path.join(root, filename)
            result.append(fullname)
    return result

# Substitue postfix .bin to .wav.
def bin2wav_filename(bin_file):
    bin_filename = os.path.basename(bin_file)
    bin_name, ext = os.path.splitext(bin_filename)
    return bin_name + '.wav'
