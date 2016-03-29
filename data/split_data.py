__author__ = 'KimTS'

import h5py
import scipy.io.wavfile
from fuel.converters.base import fill_hdf5_file


directory  = '/data/lisatmp4/taesup/data/YouTubeAudio/'
youtube_id = 'XqaJ2Ol5cC4'

wav_file = directory+youtube_id+'.wav'
output_file = directory+youtube_id+'_split.hdf5'

_, data = scipy.io.wavfile.read(wav_file)
if data.ndim == 1:
    data = data[:, None]
data = data[None, :]

num_total  = data.shape[1]
num_trains = 160000000
num_valids = num_total-num_trains
train_data = data[:,0:num_trains, :]
valid_data = data[:,num_trains:, :]

with h5py.File(output_file, 'w') as h5file:
    data = (('train', 'features', train_data),
            ('valid', 'features', valid_data))
    fill_hdf5_file(h5file, data)
    h5file['features'].dims[0].label = 'batch'
    h5file['features'].dims[1].label = 'time'
    h5file['features'].dims[2].label = 'feature'