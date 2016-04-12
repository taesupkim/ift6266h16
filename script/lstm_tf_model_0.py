__author__ = 'KimTS'
import theano
import numpy
from theano import tensor
from data.window import Window
from util.utils import save_wavfile
from layer.activations import Tanh, Logistic, Relu
from layer.layers import LinearLayer, SingleLoopLstmLayer, SingleLstmLayer
from layer.layer_utils import (get_model_updates, get_model_gradients,
                               save_model_params)
from optimizer.rmsprop import RmsProp
from numpy.random import RandomState
from theano.sandbox.rng_mrg import MRG_RandomStreams
from scipy.io import wavfile
from utils.utils import merge_dicts
theano_rng = MRG_RandomStreams(42)
np_rng = RandomState(42)

floatX = theano.config.floatX
sampling_rate = 16000

def set_generator_rnn_model(input_size,
                            hidden_size):
    layers = []
    layers.append(SingleLoopLstmLayer(input_dim=input_size,
                                      hidden_dim=hidden_size,
                                      name='generator_rnn_model'))
    return layers

def set_tf_update_function(generator_rnn_model,
                           generator_optimizer,
                           generator_grad_clipping):

    # input sequence data (time_length * num_samples * input_dims)
    input_sequence  = tensor.tensor3(name='input_sequence',
                                     dtype=floatX)
    target_sequence  = tensor.tensor3(name='target_sequence',
                                      dtype=floatX)
    # set generator input data list
    generator_input_data_list = [input_sequence,]

    # get generator output data
    generator_output = generator_rnn_model[0].forward(generator_input_data_list, is_training=True)
    generator_sample = generator_output[0]
    generator_random = generator_output[-1]

    # get square error
    square_error = tensor.sqr(target_sequence-generator_sample).sum(axis=2)

    # set generator update
    tf_updates_cost = square_error.mean()
    tf_updates_dict = get_model_updates(layers=generator_rnn_model,
                                        cost=tf_updates_cost,
                                        optimizer=generator_optimizer)

    generator_gradient_dict  = get_model_gradients(layers=generator_rnn_model,
                                                   cost=tf_updates_cost)
    generator_gradient_norm  = 0.
    for grad in generator_gradient_dict:
        generator_gradient_norm += tensor.sum(grad**2)
    generator_gradient_norm  = tensor.sqrt(generator_gradient_norm)

    # set tf update inputs
    tf_updates_inputs  = [input_sequence,
                          target_sequence]

    # set tf update outputs
    tf_updates_outputs = [square_error,
                          generator_gradient_norm,]

    # set tf update function
    tf_updates_function = theano.function(inputs=tf_updates_inputs,
                                          outputs=tf_updates_outputs,
                                          updates=merge_dicts([tf_updates_dict, generator_random]),
                                          on_unused_input='ignore')

    return tf_updates_function

def set_evaluation_function(generator_rnn_model):

    # input sequence data (time_length * num_samples * input_dims)
    input_sequence  = tensor.tensor3(name='input_sequence',
                                     dtype=floatX)
    target_sequence  = tensor.tensor3(name='target_sequence',
                                    dtype=floatX)
    # set generator input data list
    generator_input_data_list = [input_sequence,]

    # get generator output data
    generator_output = generator_rnn_model[0].forward(generator_input_data_list, is_training=True)
    generator_sample = generator_output[0]
    generator_random = generator_output[-1]

    # get square error
    square_error = tensor.sqr(target_sequence-generator_sample).sum(axis=2)

    # set evaluation inputs
    evaluation_inputs  = [input_sequence,
                          target_sequence]

    # set evaluation outputs
    evaluation_outputs = [square_error,
                          generator_sample]

    # set evaluation function
    evaluation_function = theano.function(inputs=evaluation_inputs,
                                          outputs=evaluation_outputs,
                                          updates=generator_random,
                                          on_unused_input='ignore')

    return evaluation_function

def set_sample_function(generator_rnn_model):

    # seed input data (num_samples *input_dims)
    seed_input_data = tensor.matrix(name='seed_input_data',
                                    dtype=floatX)

    time_length = tensor.scalar(name='time_length',
                                dtype='int32')

    # set generator input data list
    generator_input_data_list = [seed_input_data,
                                 time_length]

    # get generator output data
    generator_output = generator_rnn_model[0].loop_forward(generator_input_data_list)
    generator_sample = generator_output[0]
    generator_random = generator_output[-1]

    # input data
    sample_function_inputs  = [seed_input_data,
                               time_length]
    sample_function_outputs = [generator_sample,]

    sample_function = theano.function(inputs=sample_function_inputs,
                                      outputs=sample_function_outputs,
                                      updates=generator_random,
                                      on_unused_input='ignore')
    return sample_function

def train_model(feature_size,
                hidden_size,
                init_window_size,
                generator_rnn_model,
                generator_tf_optimizer,
                num_epochs,
                model_name):

    print 'COMPILING TF UPDATE FUNCTION '
    tf_updater = set_tf_update_function(generator_rnn_model=generator_rnn_model,
                                        generator_optimizer=generator_tf_optimizer,
                                        generator_grad_clipping=.0)

    # evaluator
    print 'COMPILING EVALUATION FUNCTION '
    evaluator = set_evaluation_function(generator_rnn_model=generator_rnn_model)

    # sample generator
    print 'COMPILING SAMPLING FUNCTION '
    sample_generator = set_sample_function(generator_rnn_model=generator_rnn_model)

    print 'READ RAW WAV DATA'
    _, train_raw_data = wavfile.read('/data/lisatmp4/taesup/data/YouTubeAudio/XqaJ2Ol5cC4.wav')
    valid_raw_data  = train_raw_data[160000000:]
    train_raw_data  = train_raw_data[:160000000]
    train_raw_data  = train_raw_data[2000:]
    train_raw_data  = (train_raw_data/(1.15*2.**13)).astype(floatX)
    valid_raw_data  = (valid_raw_data/(1.15*2.**13)).astype(floatX)

    num_train_total_steps = train_raw_data.shape[0]
    num_valid_total_steps = valid_raw_data.shape[0]
    batch_size      = 64

    num_valid_sequences = num_valid_total_steps/(feature_size*init_window_size)-1
    valid_source_data = valid_raw_data[:num_valid_sequences*(feature_size*init_window_size)]
    valid_source_data = valid_source_data.reshape((num_valid_sequences, init_window_size, feature_size))
    valid_target_data = valid_raw_data[feature_size:feature_size+num_valid_sequences*(feature_size*init_window_size)]
    valid_target_data = valid_target_data.reshape((num_valid_sequences, init_window_size, feature_size))

    valid_raw_data = None
    num_seeds = 10
    valid_shuffle_idx = np_rng.permutation(num_valid_sequences)
    valid_source_data = valid_source_data[valid_shuffle_idx]
    valid_target_data = valid_target_data[valid_shuffle_idx]
    valid_seed_data   = valid_source_data[:num_seeds][0][:]
    valid_source_data = numpy.swapaxes(valid_source_data, axis1=0, axis2=1)
    valid_target_data = numpy.swapaxes(valid_target_data, axis1=0, axis2=1)
    num_valid_batches = num_valid_sequences/batch_size


    print 'NUM OF VALID BATCHES : ', num_valid_sequences/batch_size
    best_valid = 10000.

    print 'START TRAINING'
    # for each epoch
    tf_train_mse_list      = []
    tf_valid_mse_list      = []
    tf_generator_grad_list = []

    train_batch_count = 0
    for e in xrange(num_epochs):
        window_size        = init_window_size + 5*e
        sequence_size      = feature_size*window_size
        train_last_seq_idx = num_train_total_steps-(sequence_size+feature_size)
        train_seq_orders   = np_rng.permutation(train_last_seq_idx)
        train_seq_orders   = train_seq_orders[:train_last_seq_idx-train_last_seq_idx%batch_size]
        train_seq_orders   = train_seq_orders.reshape((-1, batch_size))

        print 'NUM OF TRAIN BATCHES : ', train_seq_orders.shape[0]
        # for each batch
        for batch_idx, batch_info in enumerate(train_seq_orders):
            # source data
            train_source_idx  = batch_info.reshape((batch_size, 1)) + numpy.repeat(numpy.arange(sequence_size).reshape((1, sequence_size)), batch_size, axis=0)
            train_source_data = train_raw_data[train_source_idx]
            train_source_data = train_source_data.reshape((batch_size, window_size, feature_size))
            train_source_data = numpy.swapaxes(train_source_data, axis1=0, axis2=1)

            # target data
            train_target_idx  = train_source_idx + feature_size
            train_target_data = train_raw_data[train_target_idx]
            train_target_data = train_target_data.reshape((batch_size, window_size, feature_size))
            train_target_data = numpy.swapaxes(train_target_data, axis1=0, axis2=1)

            # tf update
            tf_update_output = tf_updater(train_source_data,
                                          train_target_data)
            tf_train_mse           = tf_update_output[0].mean()
            tf_generator_grad_norm = tf_update_output[1]

            train_batch_count += 1

            tf_train_mse_list.append(tf_train_mse)
            tf_generator_grad_list.append(tf_generator_grad_norm)


            if train_batch_count%10==0:
                print '============{}_LENGTH{}============'.format(model_name, window_size)
                print 'epoch {}, batch_cnt {} => TF  train mse cost  {}'.format(e, train_batch_count, tf_train_mse_list[-1])
                print 'epoch {}, batch_cnt {} => TF  train grad norm {}'.format(e, train_batch_count, tf_generator_grad_list[-1])


            if train_batch_count%100==0:
                tf_valid_mse = 0.0
                valid_batch_count = 0
                for valid_idx in xrange(num_valid_batches):
                    start_idx = batch_size*valid_idx
                    end_idx   = batch_size*(valid_idx+1)
                    evaluation_outputs = evaluator(valid_source_data[:][start_idx:end_idx][:],
                                                   valid_target_data[:][start_idx:end_idx][:])
                    tf_valid_mse += evaluation_outputs[0].mean()
                    valid_batch_count += 1

                    if valid_idx==0:
                        recon_data = evaluation_outputs[1]
                        recon_data = numpy.swapaxes(recon_data, axis1=0, axis2=1)
                        recon_data = recon_data[:10]
                        recon_data = recon_data.reshape((10, -1))
                        recon_data = recon_data*(1.15*2.**13)
                        recon_data = recon_data.astype(numpy.int16)
                        save_wavfile(recon_data, model_name+'_recon')

                        orig_data = valid_target_data[:][start_idx:end_idx][:]
                        orig_data = numpy.swapaxes(orig_data, axis1=0, axis2=1)
                        orig_data = orig_data[:10]
                        orig_data = orig_data.reshape((10, -1))
                        orig_data = orig_data*(1.15*2.**13)
                        orig_data = orig_data.astype(numpy.int16)
                        save_wavfile(orig_data, model_name+'_orig')


                tf_valid_mse_list.append(tf_valid_mse/valid_batch_count)
                print '----------------------------------------------------------'
                print 'epoch {}, batch_cnt {} => TF  valid mse cost  {}'.format(e, train_batch_count, tf_valid_mse_list[-1])

                if best_valid>tf_valid_mse_list[-1]:
                    best_valid = tf_valid_mse_list[-1]

            if train_batch_count%1000==0:
                numpy.save(file=model_name+'tf_train_mse',
                           arr=numpy.asarray(tf_train_mse_list))
                numpy.save(file=model_name+'tf_gen_grad',
                           arr=numpy.asarray(tf_generator_grad_list))
                numpy.save(file=model_name+'tf_valid_mse',
                           arr=numpy.asarray(tf_valid_mse_list))

                num_sec = 100
                sampling_length = num_sec*sampling_rate/feature_size
                seed_input_data = valid_seed_data

                [generated_sequence, ] = sample_generator(seed_input_data,
                                                          sampling_length)

                sample_data = numpy.swapaxes(generated_sequence, axis1=0, axis2=1)
                sample_data = sample_data.reshape((num_seeds, -1))
                sample_data = sample_data*(1.15*2.**13)
                sample_data = sample_data.astype(numpy.int16)
                save_wavfile(sample_data, model_name+'_sample')

                if best_valid==tf_valid_mse_list[-1]:
                    save_model_params(generator_rnn_model, model_name+'_model.pkl')

if __name__=="__main__":
    feature_size  = 1600
    hidden_size   =  800
    lr=1e-4

    model_name = 'LSTM_TF_VALID' \
                + '_FEATURE{}'.format(int(feature_size)) \
                + '_HIDDEN{}'.format(int(hidden_size)) \

    # generator model
    generator_rnn_model    = set_generator_rnn_model(input_size=feature_size,
                                                     hidden_size=hidden_size)
    # set optimizer
    tf_generator_optimizer = RmsProp(learning_rate=0.001).update_params

    train_model(feature_size=feature_size,
                hidden_size=hidden_size,
                init_window_size=100,
                generator_rnn_model=generator_rnn_model,
                generator_tf_optimizer=tf_generator_optimizer,
                num_epochs=10,
                model_name=model_name)
