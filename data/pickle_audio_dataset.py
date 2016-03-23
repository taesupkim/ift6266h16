from scipy.io import wavfile
import cPickle as pickle
import numpy

train_idx  = 0
train_size = 160000000
test_idx   = train_size
test_size  = 14229537

# get signal data from wav file
def get_signal_data(file_full_path, dtype='int16'):
    # get sampling rate and signal data
    sampling_rate, raw_data = wavfile.read(file_full_path)

    # convert data type
    raw_data = numpy.asarray(raw_data, dtype=dtype)

    # full time length in seconds
    time_length = raw_data.shape[0]/sampling_rate

    # return raw signal data, sampling rate, total length
    return [raw_data, sampling_rate, time_length]

# calculate sequence length
def get_sequence_size(step_time_length=0.1,
                      total_time_length=10.0,
                      sampling_rate=160000.0):
    input_step_size = step_time_length*sampling_rate
    num_steps = total_time_length/step_time_length

    return [input_step_size, num_steps]

# convert raw data into a set of sequences
def make_sequence_data(raw_data,
                       offset_list,
                       total_sequence_length):
    total_seq_data = numpy.empty(shape=(0, total_sequence_length))
    for offset in offset_list:
        seq_data = raw_data[offset:]
        total_data_length = seq_data.shape[0]
        num_sequences = int(total_data_length/total_sequence_length)
        seq_data = seq_data[:(num_sequences*total_sequence_length)]
        seq_data = seq_data.reshape((num_sequences, total_sequence_length))
        total_seq_data = numpy.vstack([total_seq_data, seq_data])

    return total_seq_data

def get_subset_data(raw_data,
                    start_idx,
                    num_data):
    return raw_data[start_idx:start_idx+num_data]

def data_statics(raw_data):
    data_mean = numpy.mean(raw_data)
    data_var  = numpy.var(raw_data)
    data_min  = numpy.min(raw_data)
    data_max  = numpy.max(raw_data)
    return [data_mean, data_var, data_min, data_max]


def build_sequence_data(raw_data,
                        total_sequence_length,
                        offset_list,
                        data_set_path):
    [data_mean, data_var, data_min, data_max] = data_statics(raw_data)
    raw_data = make_sequence_data(raw_data,
                                  offset_list,
                                  total_sequence_length)

    print 'sequence data shape : ({}, {})'.format(raw_data.shape[0],
                                                  raw_data.shape[1])

    print 'pickle start'
    with open(data_set_path + '.pkl', "wb") as f:
        pickle.dump((raw_data, data_mean, data_var, data_min, data_max), f, pickle.HIGHEST_PROTOCOL )
    print 'pickle done'



if __name__=="__main__":
    file_path = '/data/lisatmp4/taesup/data/YouTubeAudio/XqaJ2Ol5cC4.wav'

    for t in [5, 10, 20]:
        total_time_length = t
        print 'start with training data construction : {} seconds'.format(t)
        [raw_data, sampling_rate, time_length] = get_signal_data(file_path)
        raw_data = get_subset_data(raw_data,
                                   train_idx,
                                   train_size)
        build_sequence_data(raw_data,
                            total_time_length*sampling_rate,
                            [0, sampling_rate/2],
                            '/data/lisatmp4/taesup/data/YouTubeAudio/XqaJ2Ol5cC4_train_{}s'.format(total_time_length))

    print 'start with testing data construction'
    [raw_data, sampling_rate, time_length] = get_signal_data(file_path)
    raw_data = get_subset_data(raw_data,
                               test_idx,
                               test_size)
    build_sequence_data(raw_data,
                        test_size,
                        [0,],
                        '/data/lisatmp4/taesup/data/YouTubeAudio/XqaJ2Ol5cC4_test')