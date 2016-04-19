__author__ = 'KimTS'
import theano
import numpy
from theano import tensor
from time import time
from util.utils import save_wavfile
from layer.activations import Tanh, Logistic, Relu
from layer.layers import LinearLayer, SingleLstmGanForceLayer, LinearBatchNormalization
from layer.layer_utils import (get_tensor_output,
                               get_model_updates,
                               get_model_gradients,
                               save_model_params)
from optimizer.rmsprop import RmsProp
from optimizer.adagrad import AdaGrad
from numpy.random import RandomState
from theano.sandbox.rng_mrg import MRG_RandomStreams
from scipy.io import wavfile
from utils.utils import merge_dicts
theano_rng = MRG_RandomStreams(42)
np_rng = RandomState(42)

floatX = theano.config.floatX
sampling_rate = 16000

def set_generator_model(input_size,
                        hidden_size):
    layers = []
    layers.append(SingleLstmGanForceLayer(input_dim=input_size,
                                          hidden_dim=hidden_size,
                                          init_bias=1.0,
                                          output_activation='tanh',
                                          disconnect_sample=True,
                                          name='generator_model'))
    return layers

def set_updater_function(generator_model,
                         generator_optimizer,
                         generator_grad_clipping):
    # input sequence data (time_length * num_samples * input_dims)
    input_sequence  = tensor.tensor3(name='input_sequence',
                                     dtype=floatX)
    target_sequence  = tensor.tensor3(name='target_sequence',
                                      dtype=floatX)
    lambda_regularizer = tensor.scalar(name='lambda_regularizer',
                                       dtype=floatX)

    # set generator input data list
    generator_input_data_list = [input_sequence,]

    # get generator output data
    generator_output = generator_model[0].forward(generator_input_data_list, is_training=True)
    output_sequence  = generator_output[0]
    data_hidden      = generator_output[1]
    data_cell        = generator_output[2]
    model_hidden     = generator_output[3]
    model_cell       = generator_output[4]
    generator_random = generator_output[-1]

    # get square error
    sample_cost = tensor.sqr(target_sequence-output_sequence).sum(axis=2)

    # get positive phase hidden
    positive_hid = data_hidden[1:]

    # get negative phase hidden
    negative_hid = model_hidden[1:]

    # get phase diff cost
    regularizer_cost = tensor.sqr(positive_hid-negative_hid).sum(axis=2)

    # set generator update
    updater_cost = sample_cost.mean() + regularizer_cost.mean()*lambda_regularizer
    updater_dict = get_model_updates(layers=generator_model,
                                     cost=updater_cost,
                                     optimizer=generator_optimizer)

    # get generator gradient norm2
    generator_gradient_dict  = get_model_gradients(generator_model, updater_cost)
    generator_gradient_norm  = 0.
    for grad in generator_gradient_dict:
        generator_gradient_norm += tensor.sum(grad**2)
    generator_gradient_norm  = tensor.sqrt(generator_gradient_norm)

    # set updater inputs
    updater_inputs  = [input_sequence,
                       target_sequence,
                       lambda_regularizer]

    # set updater outputs
    updater_outputs = [sample_cost,
                       regularizer_cost,
                       generator_gradient_norm,]

    # set updater function
    updater_function = theano.function(inputs=updater_inputs,
                                       outputs=updater_outputs,
                                       updates=merge_dicts([updater_dict,
                                                            generator_random]),
                                       on_unused_input='ignore')

    return updater_function

def set_evaluation_function(generator_model):
    # input sequence data (time_length * num_samples * input_dims)
    input_sequence  = tensor.tensor3(name='input_sequence',
                                     dtype=floatX)
    target_sequence  = tensor.tensor3(name='target_sequence',
                                    dtype=floatX)
    # set generator input data list
    generator_input_data_list = [input_sequence,]

    # get generator output data
    generator_output = generator_model[0].forward(generator_input_data_list,
                                                  is_training=True)
    output_sequence  = generator_output[0]
    generator_random = generator_output[-1]

    # get square error
    sample_cost = tensor.sqr(target_sequence-output_sequence).sum(axis=2)

    # set evaluation inputs
    evaluation_inputs  = [input_sequence,
                          target_sequence]

    # set evaluation outputs
    evaluation_outputs = [sample_cost,
                          output_sequence]

    # set evaluation function
    evaluation_function = theano.function(inputs=evaluation_inputs,
                                          outputs=evaluation_outputs,
                                          updates=generator_random,
                                          on_unused_input='ignore')

    return evaluation_function


def set_sampling_function(generator_model):
    # seed input data (num_samples *input_dims)
    seed_input_data = tensor.matrix(name='seed_input_data',
                                    dtype=floatX)

    time_length = tensor.scalar(name='time_length',
                                dtype='int32')

    # set generator input data list
    generator_input_data_list = [seed_input_data,
                                 time_length]

    # get generator output data
    generator_output = generator_model[0].forward(generator_input_data_list,
                                                  is_training=False)
    generator_sample = generator_output[0]
    generator_random = generator_output[-1]

    # input data
    sampling_function_inputs  = [seed_input_data,
                                 time_length]
    sampling_function_outputs = [generator_sample,]

    sampling_function = theano.function(inputs=sampling_function_inputs,
                                        outputs=sampling_function_outputs,
                                        updates=generator_random,
                                        on_unused_input='ignore')
    return sampling_function

def train_model(feature_size,
                hidden_size,
                init_window_size,
                generator_model,
                generator_optimizer,
                num_epochs,
                model_name):

    # model updater
    print 'COMPILING UPDATER FUNCTION '
    t = time()
    updater_function = set_updater_function(generator_model=generator_model,
                                            generator_optimizer=generator_optimizer,
                                            generator_grad_clipping=.0)
    print '%.2f SEC '%(time()-t)

    # evaluator
    print 'COMPILING EVALUATION FUNCTION '
    t = time()
    evaluation_function = set_evaluation_function(generator_model=generator_model)
    print '%.2f SEC '%(time()-t)

    # sample generator
    print 'COMPILING SAMPLING FUNCTION '
    t = time()
    sampling_function = set_sampling_function(generator_model=generator_model)
    print '%.2f SEC '%(time()-t)

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
    train_sample_cost_list        = []
    train_regularizer_cost_list   = []
    train_gradient_norm_list      = []
    train_lambda_regularizer_list = []
    valid_sample_cost_list        = []


    train_batch_count = 0
    for e in xrange(num_epochs):
        window_size      = init_window_size + 5*e
        sequence_size    = feature_size*window_size
        last_seq_idx     = num_train_total_steps-(sequence_size+feature_size)
        train_seq_orders = np_rng.permutation(last_seq_idx)
        train_seq_orders = train_seq_orders[:last_seq_idx-last_seq_idx%batch_size]
        train_seq_orders = train_seq_orders.reshape((-1, batch_size))

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


            # update model
            lambda_regularizer = 0.1
            updater_outputs = updater_function(train_source_data,
                                               train_target_data,
                                               lambda_regularizer)
            train_sample_cost      = updater_outputs[0].mean()
            train_regularizer_cost = updater_outputs[1].mean()
            train_gradient_norm    = updater_outputs[2]

            train_batch_count += 1

            train_sample_cost_list.append(train_sample_cost)
            train_regularizer_cost_list.append(train_regularizer_cost)
            train_gradient_norm_list.append(train_gradient_norm)
            train_lambda_regularizer_list.append(lambda_regularizer)

            if train_batch_count%10==0:
                print '============{}_LENGTH{}============'.format(model_name, window_size)
                print 'epoch {}, batch_cnt {} => train sample      cost   {}'.format(e, train_batch_count, train_sample_cost_list[-1])
                print 'epoch {}, batch_cnt {} => train regularizer cost   {}'.format(e, train_batch_count, train_regularizer_cost_list[-1])
                print '----------------------------------------------------------'
                print 'epoch {}, batch_cnt {} => train gradient    norm   {}'.format(e, train_batch_count, train_gradient_norm_list[-1])
                print 'epoch {}, batch_cnt {} => train regularizer lambda {}'.format(e, train_batch_count, train_lambda_regularizer_list[-1])


            if train_batch_count%100==0:
                tf_valid_mse = 0.0
                valid_batch_count = 0
                for valid_idx in xrange(num_valid_batches):
                    start_idx = batch_size*valid_idx
                    end_idx   = batch_size*(valid_idx+1)
                    evaluation_outputs = evaluation_function(valid_source_data[:][start_idx:end_idx][:],
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

                valid_sample_cost_list.append(tf_valid_mse/valid_batch_count)
                print '----------------------------------------------------------'
                print 'epoch {}, batch_cnt {} => valid sample      cost   {}'.format(e, train_batch_count, valid_sample_cost_list[-1])

                if best_valid>valid_sample_cost_list[-1]:
                    best_valid = valid_sample_cost_list[-1]


            if train_batch_count%500==0:
                numpy.save(file=model_name+'_train_sample_cost',
                           arr=numpy.asarray(train_sample_cost_list))
                numpy.save(file=model_name+'_train_regularizer_cost',
                           arr=numpy.asarray(train_regularizer_cost_list))
                numpy.save(file=model_name+'_train_gradient_norm',
                           arr=numpy.asarray(train_gradient_norm_list))
                numpy.save(file=model_name+'_train_lambda_value',
                           arr=numpy.asarray(train_lambda_regularizer_list))
                numpy.save(file=model_name+'_valid_sample_cost',
                           arr=numpy.asarray(valid_sample_cost_list))

                num_sec = 100
                sampling_length = num_sec*sampling_rate/feature_size
                seed_input_data = valid_seed_data

                [generated_sequence, ] = sampling_function(seed_input_data,
                                                           sampling_length)

                sample_data = numpy.swapaxes(generated_sequence, axis1=0, axis2=1)
                sample_data = sample_data.reshape((num_seeds, -1))
                sample_data = sample_data*(1.15*2.**13)
                sample_data = sample_data.astype(numpy.int16)
                save_wavfile(sample_data, model_name+'_sample')

                # if best_valid==valid_sample_cost_list[-1]:
            save_model_params(generator_model, model_name+'_model.pkl')


if __name__=="__main__":
    feature_size  = 1600
    hidden_size   = 1600

    model_name = 'LSTM_REGULARIZER_LAMBDA' \
                + '_FEATURE{}'.format(int(feature_size)) \
                + '_HIDDEN{}'.format(int(hidden_size)) \

    # generator model
    generator_model = set_generator_model(input_size=feature_size,
                                          hidden_size=hidden_size)

    # set optimizer
    generator_optimizer = RmsProp(learning_rate=0.01, momentum=0.9).update_params

    train_model(feature_size=feature_size,
                hidden_size=hidden_size,
                init_window_size=100,
                generator_model=generator_model,
                generator_optimizer=generator_optimizer,
                num_epochs=10,
                model_name=model_name)
