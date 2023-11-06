from readBinaries import tonalReader
import numpy as np
from sigproc import magspec, framesig
import wavio
import math

def getSpectrogram(audioFile, frame_time_span = 8, step_time_span = 2, spec_clip_min = 0,
                   spec_clip_max = 6, min_freq = 5000, max_freq = 50000,
                   start_time = 0, end_time=-1):
    '''
    Gets and returns a two-dimensional list in which the values encode a spectrogram.

    :param audioFile: the audio file in .wav format for which a spectrogram is generated
    :param frame_time_span: ms, length of time for one time window for dft
    :param step_time_span: ms, length of time step for spectrogram
    :param spec_clip_min: log magnitude spectrogram min-max normalization, minimum value
    :param spec_clip_max: log magnitude spectrogram min-max normalization, maximum value
    :param min_freq: Hz, lower bound of frequency for spectrogram
    :param max_freq: Hz, upper bound of frequency for spectrogram
    :param start_time: ms, the beginning of where the audioFile is read
    :param end_time: ms, the end of where the audioFile is read. -1 reads until the end

    :returns: the spectrogram
    '''

    freq_resolution = 1000 / frame_time_span

    # Load audio file
    wav_data = wavio.read(audioFile)

    # I copy this from Pu Li's DeepWhistle implementation
    # in wav2spec.py. I do not know why it is necessary
    if wav_data.sampwidth > 2:
            wav_data.sampwidth /= 2 ** (8 * (wav_data.sampwidth - 2))

    # #
    # Split the wave signal into overlapping frames
    # #

    start_frame = int(start_time / 1000 * wav_data.rate)
    end_frame = int((end_time / 1000 + 1 / freq_resolution - step_time_span / 1000)* wav_data.rate)

    frame_sample_span = int(math.floor(frame_time_span / 1000 * wav_data.rate) + 1)
    step_sample_span = int(math.floor(step_time_span / 1000 * wav_data.rate))
    # No frames if the audio file is too short
    print(start_frame, end_frame, wav_data.data[start_frame:end_frame].shape, frame_sample_span)
    print("B", end_time, freq_resolution, step_time_span, wav_data.rate)
    if wav_data.data[start_frame:end_frame].shape[0] < frame_sample_span:
        frames = []
    else:
        frames = framesig(wav_data.data.ravel()[start_frame:end_frame], frame_sample_span, step_sample_span)

    # #
    # Make spectrogram
    # #
    NFFT = len(frames[0])

    # Compute magnitude spectra
    singal_magspec = magspec(frames, NFFT)

    clip_bottom = int(min_freq // freq_resolution)
    clip_top = int(max_freq // freq_resolution + 1)
    spectogram = singal_magspec.T[clip_bottom:clip_top]
    spectogram = np.log10(spectogram)
    spectogram = normalize3(spectogram, spec_clip_min, spec_clip_max)

    # Flip spectrogram to match expectations for display
    # and scall to be 0-255
    spectrogram_flipped = spectogram[::-1, ] * 255
    return spectrogram_flipped




####### UTILITY #######

# Credit to Pu Li https://github.com/Paul-LiPu/DeepWhistle
# min-max normalization
def normalize3(mat, min_v, max_v):
    mat = np.clip(mat, min_v, max_v)
    return (mat - min_v) / (max_v - min_v)