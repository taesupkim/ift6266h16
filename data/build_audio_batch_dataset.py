import pafy
import os
import sys
import subprocess
from fuel.datasets.hdf5 import H5PYDataset
import scipy.io.wavfile as wav
import h5py

def extract_wav_youtube(save_dir, youtube_id, channels, sample_rate):
    # download file
    file_path = os.path.join(save_dir, '{}.m4a'.format(youtube_id))
    file_url  = 'https://www.youtube.com/watch?v={}'.format(youtube_id)
    video = pafy.new(file_url)
    audio = video.getbestaudio()
    audio.download(quiet=False, filepath=file_path)

    # extract audio
    wav_path = '{}.wav'.format(youtube_id)
    wav_path = os.path.join(save_dir, wav_path)
    subprocess.check_call(['ffmpeg', '-y', '-i', file_path, '-ac',
                           str(channels), '-ar', str(sample_rate), wav_path],
                          stdout=sys.stdout)

    return wav_path

def build_raw_hdf5_dataset(wav_name, hdf5_name, window_size):
    [rate, signal] = wav.read(wav_name)
    num_steps   = signal.shape[0]
    num_seqs    = num_steps-window_size
    output_path = '{}.hdf5'.format(hdf5_name)
    output_path = os.path.join(output_path)

    signal = signal.reshape(num_steps,1)
    with h5py.File(output_path, mode='w') as h5file:
        input_feature  = h5file.create_dataset(name='input_feature' , shape=(num_seqs, window_size, 1), dtype='int16')
        target_feature = h5file.create_dataset(name='target_feature', shape=(num_seqs, window_size, 1), dtype='int16')
        print ' num of sequences : {}'.format(num_seqs)
        for s in xrange(num_seqs):
            input_feature[s]  = signal[s:s+window_size]
            target_feature[s] = signal[(s+1):(s+1)+window_size]

        # label each dataset axis
        input_feature.dims[0].label = 'batch'
        input_feature.dims[1].label = 'time'
        input_feature.dims[2].label = 'feature'

        target_feature.dims[0].label = 'batch'
        target_feature.dims[1].label = 'time'
        target_feature.dims[2].label = 'feature'

        split_dict = {'train': {'input_feature' : ( 0,  num_seqs),
                                'target_feature': ( 0,  num_seqs)}}

        h5file.attrs['split'] = H5PYDataset.create_split_array(split_dict)

        h5file.flush()
        h5file.close()

    return num_seqs

if __name__=="__main__":
    directory  = '/data/lisatmp4/taesup/data/YouTubeAudio/'
    youtube_id = 'XqaJ2Ol5cC4'
    wavfile_path = extract_wav_youtube(directory, youtube_id, 1, 16000)
    # wavfile_path = directory + youtube_id + '.wav'
    window_size_list = [100, 1000]
    for window_size in window_size_list:
        hdf5_name = directory + 'XqaJ2Ol5cC4_{}'.format(window_size)
        build_raw_hdf5_dataset(wavfile_path, hdf5_name, window_size)
        print 'done window size : {}'.format(window_size)

