import pafy
import os
import sys
import subprocess
from fuel.datasets.hdf5 import H5PYDataset
import scipy.io.wavfile as wav
import h5py
from data.window import Window
from fuel.datasets.youtube_audio import YouTubeAudio

# dataset size
# (num_samples_for_dataset, num_secs , sec_length) (num_seqs, window_size, sec_length)
# sec_length =16,000

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

def build_raw_interval_hdf5_dataset(youtube_id, hdf5_name, interval_size, window_size):
    data_stream = YouTubeAudio(youtube_id).get_example_stream()

    data_stream = Window(offset=interval_size,
                         source_window=interval_size*window_size,
                         target_window=interval_size*window_size,
                         overlapping=True,
                         data_stream=data_stream)

    data_iterator = data_stream.get_epoch_iterator()

    num_sequences = 0
    for data in data_iterator:
        num_sequences = num_sequences + 1

    output_path = '{}.hdf5'.format(hdf5_name)
    output_path = os.path.join(output_path)
    print 'total num sequences : ', num_sequences
    with h5py.File(output_path, mode='w') as h5file:
        input_feature  = h5file.create_dataset(name='input_feature' , shape=(num_sequences, window_size, interval_size), dtype='int16')
        target_feature = h5file.create_dataset(name='target_feature', shape=(num_sequences, window_size, interval_size), dtype='int16')

        data_iterator = data_stream.get_epoch_iterator()
        # for each batch
        for s_idx, sequence_data in enumerate(data_iterator):
            # get data
            source_data = sequence_data[0]
            target_data = sequence_data[1]

            # save data
            input_feature[s_idx]  = source_data.reshape(window_size, interval_size)
            target_feature[s_idx]  = target_data.reshape(window_size, interval_size)

        # label each dataset axis
        input_feature.dims[0].label = 'batch'
        input_feature.dims[1].label = 'time'
        input_feature.dims[2].label = 'feature'

        target_feature.dims[0].label = 'batch'
        target_feature.dims[1].label = 'time'
        target_feature.dims[2].label = 'feature'

        num_trains = int(num_sequences*0.8)

        split_dict = {'train': {'input_feature' : ( 0,  num_trains),
                                'target_feature': ( 0,  num_trains)},
                      'valid': {'input_feature' : ( num_trains,  num_sequences),
                                'target_feature': ( num_trains,  num_sequences)},
                      }
        h5file.attrs['split'] = H5PYDataset.create_split_array(split_dict)

        h5file.flush()
        h5file.close()

    return num_sequences

if __name__=="__main__":

    youtube_id = 'XqaJ2Ol5cC4'
    interval_size_list = [0.1, 0.5, 1]
    window_size_list = [100, 1000]
    for interval_size in interval_size_list:
        for window_size in window_size_list:
            hdf5_name  = '/data/lisatmp4/taesup/data/YouTubeAudio/sec{}_step{}.hdf5'.format(interval_size, window_size)
            print hdf5_name
            print build_raw_interval_hdf5_dataset(youtube_id,
                                                  hdf5_name,
                                                  16000*interval_size,
                                                  window_size)
            print 'Done'

