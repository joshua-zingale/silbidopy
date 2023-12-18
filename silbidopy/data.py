from silbidopy.render import getSpectrogram, getAnnotationMask
from silbidopy.readBinaries import tonalReader
from torch.utils.data import Dataset, IterableDataset
import numpy as np
import fnmatch
import wavio
import os

class AudioTonalDataset(Dataset):
    def __init__(self, audio_dir, annotation_dir, frame_time_span = 8, step_time_span = 2,
                 spec_clip_min = 0, spec_clip_max = 6, min_freq = 5000, max_freq = 50000,
                 time_patch_frames = 50, freq_patch_frames = 50, time_patch_advance = None,
                 freq_patch_advance = None, cache_wavs = True,
                 cache_annotations = True):
        '''
        A Dataset that pulls spectrograms and tonal annotations from audio and annotation
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
        :param time_patch_advance: the number of time frames between successive patches.                                   Defaults to time_patch_frames (also when argument set to None)
        :param freq_patch_frames: the number of frequency frames per ouput datum, i.e. the number per patch.
        :param freq_patch_advance: the number offrequency frames between successive patches.                                   Defaults to freq_patch_frames (also when argument set to None)
        :param cache_wavs: If True, all wav files are saved in memory;
                           else, each datum access opens and closes a
                           wav file.
        :param cache_annotations: If True, all annotations are saved
                                  in memory; else, each datum access
                                  loads the relevant annotations.
                                  WARNING: setting to false leads to
                                  a significant slowdown.

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
        self.freq_patch_length_hz = freq_resolution * freq_patch_frames
        self.freq_patch_advance_hz = self.freq_patch_length_hz if freq_patch_advance == None else freq_resolution * freq_patch_advance
        self.time_patch_length_ms = step_time_span * time_patch_frames
        self.time_patch_advance_ms = self.time_patch_length_ms if time_patch_advance == None else step_time_span * time_patch_advance

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
        self.cache_annotations = cache_annotations

        self.bin_files = bin_files
        self.anno_wav_files = anno_wav_files

        # Get length of dataset
        self.num_patches = []
        self.file_info = []
        for i, wav_file in enumerate((wavio.read(w) for w in anno_wav_files)):
            num_samples = wav_file.data.shape[0]

            file_length_ms = num_samples * 1000 / wav_file.rate

            # determine & append number of patches in file
            num_freq_divisions = int((max_freq - self.freq_patch_length_hz) / self.freq_patch_advance_hz)

            num_time_divisions = int((file_length_ms - self.time_patch_length_ms - frame_time_span) / self.time_patch_advance_ms)

            self.num_patches.append(num_freq_divisions * num_time_divisions)

            self.file_info.append({
                "audio_file": anno_wav_files[i],
                "num_time_divisions": num_time_divisions,
                })
            if cache_wavs:
                self.file_info[i]["wav_data"] = wav_file
            if cache_annotations:
                self.file_info[i]["contours"] =  tonalReader(self.bin_files[i]).getTimeFrequencyContours()
    
    def get_balanced_dataset(self, positive_proportion = 0.5):
        '''
        Builds a new dataset that wraps around the current one. The new
        dataset will include entries of the current dataset
        such that a specific proportion of the entries will be positive.
        The generated dataset will randomly select entries from the current
        dataset to include.
        The generated dataset also depends on the current dataset's
        existence to function.

        :param positive_proportion: the proportion of entries that will
                                    be positive in the new dataset.
                                    Setting this to 0.5 results in a
                                    balanced dataset

        :returns: the newly balanced dataset
        '''
        return BalancedDataset(self, self.get_positive_indices(),
                positive_proportion = positive_proportion)

    def get_balanced_iterable(self, epoch_size = None):
        '''
        Returns an iterable PyTorch dataset, for which the output values
        alternate between positive and negative.

        :param epoch_size: the size of an epoch, i.e. of the iterator.
                           If None, which is default, epoch_size is set to
                           be twice the size of the minority class.
        '''
        return BalancedIterableDataset(self, self.get_positive_indices(),
                epoch_size = epoch_size)

    def get_positive_indices(self):
        '''
        Returns a set that contains all positive indices,
        i.e. all indices for which at least one pixel in the label has
        tonal energy
        '''

        # Each node can be in multiple patches.
        # Get the possible number of patches that could overlap with a node
        freq_patch_range = self.freq_patch_length_hz / self.freq_patch_advance_hz
        time_patch_range = self.time_patch_length_ms / self.time_patch_advance_ms
        
        patches_cumsum = np.cumsum(np.append([0], self.num_patches))
        positive_set = set()
       
        # Get the index of each patch that has at least one node in it
        for file_idx in range(len(self.bin_files)):
            contours = None
            if self.cache_annotations:
                contours = self.file_info[file_idx]["contours"]
            else:
                contours = tonalReader(self.bin_files[file_idx]).getTimeFrequencyContours()

            num_time_divisions = self.file_info[file_idx]["num_time_divisions"]
            for contour in contours:
                for time, freq in contour:
                    # skip out of range nodes
                    if freq < self.min_freq or freq >= self.max_freq:
                        continue
                    # seconds to miliseconds
                    time = time *1000

                    # Each node can be in multiple patches
                    # Start with the highest frequency and time patch
                    freq_patch = ((freq - self.min_freq) / self.freq_patch_advance_hz)
                    time_patch = (time / self.time_patch_advance_ms)

                    # The number of frequency and time frames that overlap
                    # with the node
                    freq_overlap = np.ceil(freq_patch_range - freq_patch % 1).astype(int)
                    time_overlap = np.ceil(time_patch_range - time_patch % 1).astype(int)
                    
                    # Add indices to positive set
                    for f_idx in range(int(freq_patch), max(int(freq_patch) - freq_overlap, -1), -1):
                        for t_idx in range(int(time_patch), max(int(time_patch) - time_overlap, -1), -1):
                            idx = t_idx + f_idx * num_time_divisions + patches_cumsum[file_idx]
                    
                            positive_set.add(idx)

        return positive_set
    
    def get_index_source(self, idx):
        ''' Finds the source audio file and timestamp for an index
        in this dataset. Returns a three item tuple,
        (file_name: str, start_time: float, start_frequency: float)'''
        
        ## Determine which file corresponds to idx ##
        if idx >= len(self):
            raise IndexError("Dataset index out of bounds.")
        patches_cumsum = np.cumsum(self.num_patches)
        file_idx = np.argmax(patches_cumsum > idx)

        audio_file = self.file_info[file_idx]["audio_file"] 

        # Adjust idx to be relative to the file index
        if file_idx != 0:
            idx -= patches_cumsum[file_idx - 1]
        
        # get starting time and frequency
        num_time_divisions = self.file_info[file_idx]["num_time_divisions"]
        start_time = (idx % num_time_divisions) * self.time_patch_advance_ms
        start_freq = (idx // num_time_divisions) * self.freq_patch_advance_hz + self.min_freq

        return audio_file, start_time, start_freq




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

        end_time = start_time + self.time_patch_length_ms
        end_freq = start_freq + self.freq_patch_length_hz
       
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
        
        # get contours
        contours = None
        if self.cache_annotations:
            contours = self.file_info[file_idx]["contours"]
        else:
            contours = tonalReader(self.bin_files[file_idx]).getTimeFrequencyContours()
        
        label = getAnnotationMask(contours,
                frame_time_span = self.frame_time_span,
                step_time_span = self.step_time_span,
                min_freq = start_freq, max_freq = end_freq,
                start_time = start_time, end_time = end_time)

        return datum, label

class BalancedDataset(Dataset):
    def __init__(self, dataset, positive_set,
            positive_proportion = 0.5):
        '''
        A dataset that rebalances another dataset such that a certain
        proportion of the indices correspond to positive labels. The
        size of this dataset will be as large as possible given the
        number of positive and negative labels in the original dataset
        and the desired proportion of labels that be positive. The order
        of this dataset is random, including which indices from the
        majority class are selected for inclusion.

        :param dataset: the dataset for which a balanced dataset will be
                        created
        :param positive_set: a Python set that contains all of the indices
                             in dataset that correspond to a positive
                             label
        :param positive_proportion: the proportion of indices in this
                                    new dataset that will correspond to
                                    positive labels

        '''
        

        positive_indices = np.array(list(positive_set))
        negative_indices = np.array(list(set(range(len(dataset))) - positive_set))
        np.random.shuffle(positive_indices)
        np.random.shuffle(negative_indices)

        # Get number of positive and negative indices to be included
        init_len_p = len(positive_indices)
        init_len_n = len(negative_indices)

        len_p = int(min(
                init_len_p,
                positive_proportion * init_len_n / (1-positive_proportion)
                ))
        len_n = int(
                (1 - positive_proportion) * len_p / positive_proportion
                )

        positive_indices = positive_indices[:len_p]
        negative_indices = negative_indices[:len_n]
        
        # Combine the positive and negative index arrays
        self.indices = np.append(positive_indices, negative_indices)
        np.random.shuffle(self.indices)

        self.dataset = dataset


    def get_index_source(self, idx):
        ''' Finds the source audio file and timestamp for an index
        in this dataset. Returns a three item tuple,
        (file_name: str, start_time: float, start_frequency: float)'''
        return self.dataset.get_index_source(self.indices[idx])

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        return self.dataset.__getitem__(self.indices[idx])



class BalancedIterableDataset(IterableDataset):
    def __init__(self, dataset, positive_set, epoch_size = None):
        '''
        An IterableDataset that samples from a map-style
        Dataset such that there is an equal chance of sampling
        a positive sample to that of a negative sample.
        The dataset is shuffled at the load of the iterable,
        i.e. at the start of each epoch.

        :param dataset: the map-style dataset
        :param positive_set: a Python set that contains the positive indices
                       for the map-style input dataset
        :param epoch_size: the size of an epoch, i.e. of the iterator.
                           If None, which is default, epoch_size is set to
                           be twice the size of the minority class.
        '''
        
        length = len(dataset)
        negative_set = set(range(length)) - positive_set

        self.positive_indices = np.array(list(positive_set))
        self.negative_indices = np.array(list(negative_set))

        self.dataset = dataset

        self.next_is_positive = True

        if epoch_size == None:
            self.epoch_size = 2 * min(len(self.positive_indices), len(self.negative_indices))
        else:
            self.epoch_size = epoch_size

    def __iter__(self):
        np.random.shuffle(self.positive_indices)
        np.random.shuffle(self.negative_indices)

        for i in range(self.epoch_size):
            if self.next_is_positive:
                self.next_is_positive = False
                idx = self.positive_indices[i // 2]
                yield self.dataset.__getitem__(idx)
            else:
                self.next_is_positive = True
                idx = self.negative_indices[i // 2]
                yield self.dataset.__getitem__(idx)

## HELPERS ##
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
