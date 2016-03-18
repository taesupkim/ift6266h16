from scipy.io import wavfile
import cPickle as pickle
import numpy as np

def get_signal_data(file_path, file_name, dtype='float32'):
    # sampling_rate : number of time steps for each second
    sampling_rate, raw_data = wavfile.read(file_path + file_name)
    raw_data = np.asarray(raw_data, dtype=dtype)
    time_length = raw_data.shape[0]/sampling_rate
    return [raw_data, sampling_rate, time_length]

def get_data_size(single_time_length, sequence_time_length, sampling_rate):
    input_size      = single_time_length*sampling_rate
    sequence_length = sequence_time_length/single_time_length
    return [input_size, sequence_length]

def split_data(raw_data, train_ratio):
    data_size  = raw_data.shape[0]
    train_data = raw_data[:int(data_size*train_ratio)]
    valid_data = raw_data[int(data_size*train_ratio):]

    return [train_data, valid_data]

def data_statics(raw_data):
    data_mean = np.mean(raw_data)
    data_min  = np.min(raw_data)
    data_max  = np.max(raw_data)
    return [data_mean, data_min, data_max]

def normalize_data(raw_data):
    [data_mean, data_min, data_max] = data_statics(raw_data)
    raw_data = (raw_data-data_min)/(data_max-data_min)
    return [raw_data, data_min, data_max]

def build_sequence_data(raw_data_set,
                        input_size,
                        sequence_timesteps,
                        overlap,
                        data_set_path,
                        data_min_max):
    sequence_data_set = []
    sequence_length = input_size*sequence_timesteps
    for i in xrange(len(raw_data_set)):
        seq_start_idx = 0
        seq_list = []
        while True:
            seq_end_idx = seq_start_idx + sequence_length
            if seq_end_idx>=raw_data_set[i].shape[0]:
                break
            seq_data = raw_data_set[i][seq_start_idx:seq_end_idx]
            seq_list.append(seq_data)
            seq_start_idx += overlap
        sequence_data_set.append(np.asarray(seq_list, dtype="float32"))
        print 'sequence data set shape : ({}, {})'.format(sequence_data_set[-1].shape[0],
                                                          sequence_data_set[-1].shape[1])

    with open(data_set_path, "wb") as f:
        pickle.dump((sequence_data_set, data_min_max[0], data_min_max[1]), f, pickle.HIGHEST_PROTOCOL )

if __name__=="__main__":
    file_path = '/data/lisatmp4/taesup/data/YouTubeAudio/'
    file_name = 'XqaJ2Ol5cC4.wav'
    train_ratio = 0.7
    # read data
    [raw_data, sampling_rate, time_length] = get_signal_data(file_path, file_name)
    [raw_data, data_min, data_max] = normalize_data(raw_data)
    [train_raw_data, valid_raw_data] = split_data(raw_data, train_ratio)

    # dataset type
    single_time_length_list   = [0.01, 0.05] #sec
    sequence_time_length_list = [10, 50] #sec

    for single_time_length in single_time_length_list:
        for sequence_time_length in sequence_time_length_list:
            # get data size
            [input_size, sequence_length] = get_data_size(single_time_length,
                                                          sequence_time_length,
                                                          sampling_rate)

            data_set_path = file_path + 'audio_input{}seq{}.pkl'.format(input_size,
                                                                        sequence_length)
            # build dataset
            build_sequence_data([train_raw_data, valid_raw_data],
                                input_size,
                                sequence_length,
                                input_size/2,
                                data_set_path,
                                [data_min, data_max])