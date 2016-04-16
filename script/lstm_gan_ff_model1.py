__author__ = 'KimTS'
import theano
import numpy
from theano import tensor
from time import time
from util.utils import save_wavfile
from layer.activations import Tanh, Logistic, Relu
from layer.layers import LinearLayer, SingleLstmGanForceLayer, SingleLstmLayer
from layer.layer_utils import get_tensor_output, get_model_updates, get_model_gradients, save_model_params
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
                                          disconnect_sample=True,
                                          name='generator_model'))
    return layers

def set_discriminator_feature_model(hidden_size,
                                    feature_size):
    layers = []
    layers.append(LinearLayer(input_dim=hidden_size,
                              output_dim=hidden_size/2,
                              name='discriminator_feature_linear0'))
    layers.append(Relu(name='discriminator_feature_relu0'))
    layers.append(LinearLayer(input_dim=hidden_size/2,
                              output_dim=feature_size,
                              name='discriminator_feature_linear1'))
    layers.append(Relu(name='discriminator_feature_relu1'))
    return layers

def set_discriminator_output_model(feature_size):
    layers = []
    layers.append(LinearLayer(input_dim=feature_size*2,
                              output_dim=feature_size,
                              name='discriminator_output_linear0'))
    layers.append(Relu(name='discriminator_output_relu0'))
    layers.append(LinearLayer(input_dim=feature_size,
                              output_dim=1,
                              name='discriminator_output_linear1'))
    layers.append(Logistic(name='discriminator_output_linear1'))
    return layers

def set_updater_function(generator_model,
                         discriminator_feature_model,
                         discriminator_output_model,
                         generator_optimizer,
                         discriminator_optimizer,
                         generator_grad_clipping,
                         discriminator_grad_clipping):

    # input sequence data (time_length * num_samples * input_dims)
    input_sequence  = tensor.tensor3(name='input_sequence',
                                     dtype=floatX)
    target_sequence  = tensor.tensor3(name='target_sequence',
                                      dtype=floatX)
    # set generator input data list
    generator_input_data_list = [input_sequence, ]

    # get generator output data
    generator_output = generator_model[0].forward(generator_input_data_list, is_training=True)
    output_sequence  = generator_output[0]
    data_hidden      = generator_output[1]
    model_hidden     = generator_output[3]
    generator_random = generator_output[-1]

    # get conditional hidden
    condition_hid     = data_hidden[:-1]
    condition_hid     = theano.gradient.disconnected_grad(condition_hid)
    condition_feature = get_tensor_output(condition_hid,
                                          discriminator_feature_model,
                                          is_training=True)

    # get positive hidden
    positive_hid     = data_hidden[1:]
    positive_feature = get_tensor_output(positive_hid,
                                         discriminator_feature_model,
                                         is_training=True)

    # get negative hidden
    negative_hid     = model_hidden[1:]
    negative_feature = get_tensor_output(negative_hid,
                                         discriminator_feature_model,
                                         is_training=True)

    # get positive/negative pairs
    positive_pair = tensor.concatenate([condition_feature, positive_feature], axis=2)
    negative_pair = tensor.concatenate([condition_feature, negative_feature], axis=2)

    # get positive score
    positive_score = get_tensor_output(positive_pair,
                                       discriminator_output_model,
                                       is_training=True)

    # get negative score
    negative_score = get_tensor_output(negative_pair,
                                       discriminator_output_model,
                                       is_training=True)

    # get generator gan cost
    generator_gan_cost = tensor.nnet.binary_crossentropy(output=negative_score,
                                                         target=tensor.ones_like(negative_score))

    # get discriminator gan cost
    discriminator_gan_cost = (tensor.nnet.binary_crossentropy(output=positive_score,
                                                              target=tensor.ones_like(positive_score)) +
                              tensor.nnet.binary_crossentropy(output=negative_score,
                                                              target=tensor.zeros_like(negative_score)))

    # get data square error
    square_error = tensor.sqr(target_sequence-output_sequence).sum(axis=2)
    hidden_error = tensor.sqr(positive_hid-negative_hid).sum(axis=2)

    # set generator update
    generator_updates_cost = generator_gan_cost.mean() + square_error.mean()
    generator_updates_dict = get_model_updates(layers=generator_model,
                                               cost=generator_updates_cost,
                                               optimizer=generator_optimizer,
                                               use_grad_clip=generator_grad_clipping)

    # get generator gradient
    generator_gradient_dict  = get_model_gradients(layers=generator_model,
                                                   cost=generator_updates_cost)
    generator_gradient_norm  = 0.
    for grad in generator_gradient_dict:
        generator_gradient_norm += tensor.sum(grad**2)
    generator_gradient_norm  = tensor.sqrt(generator_gradient_norm)

    # set discriminator update
    discriminator_updates_cost = discriminator_gan_cost.mean()
    discriminator_updates_dict = get_model_updates(layers=discriminator_feature_model+discriminator_output_model,
                                                   cost=discriminator_updates_cost,
                                                   optimizer=discriminator_optimizer,
                                                   use_grad_clip=discriminator_grad_clipping)

    # get discriminator gradient
    discriminator_gradient_dict  = get_model_gradients(layers=discriminator_feature_model+discriminator_output_model,
                                                       cost=discriminator_updates_cost)
    discriminator_gradient_norm  = 0.
    for grad in discriminator_gradient_dict:
        discriminator_gradient_norm += tensor.sum(grad**2)
    discriminator_gradient_norm  = tensor.sqrt(discriminator_gradient_norm)

    # set updater inputs
    updater_inputs  = [input_sequence,
                       target_sequence]

    # set updater outputs
    updater_outputs = [generator_gan_cost,
                       discriminator_gan_cost,
                       positive_score,
                       negative_score,
                       square_error,
                       hidden_error,
                       generator_gradient_norm,
                       discriminator_gradient_norm,]

    # set updater function
    updater_function = theano.function(inputs=updater_inputs,
                                       outputs=updater_outputs,
                                       updates=merge_dicts([generator_updates_dict,
                                                            discriminator_updates_dict,
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
    output_data_set  = generator_model[0].forward(generator_input_data_list, is_training=True)
    output_sequence  = output_data_set[0]
    generator_random = output_data_set[-1]

    # get square error
    square_error = tensor.sqr(target_sequence-output_sequence).sum(axis=2)

    # set evaluation inputs
    evaluation_inputs  = [input_sequence,
                          target_sequence]

    # set evaluation outputs
    evaluation_outputs = [square_error,
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
                                 time_length,]

    # get generator output data
    generator_output = generator_model[0].forward(generator_input_data_list, is_training=False)
    generator_sample = generator_output[0]
    generator_random = generator_output[-1]

    # input data
    sampling_function_inputs  = [seed_input_data,
                                 time_length,]
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
                discriminator_feature_model,
                discriminator_output_model,
                discriminator_optimizer,
                num_epochs,
                model_name):

    # updater function
    print 'COMPILING UPDATER FUNCTION '
    t = time()
    updater_function= set_updater_function(generator_model=generator_model,
                                           discriminator_feature_model=discriminator_feature_model,
                                           discriminator_output_model=discriminator_output_model,
                                           generator_optimizer=generator_optimizer,
                                           discriminator_optimizer=discriminator_optimizer,
                                           generator_grad_clipping=.0,
                                           discriminator_grad_clipping=.0)
    print '%.2f SEC '%(time()-t)

    # sample generator
    print 'COMPILING SAMPLING FUNCTION '
    t = time()
    sampling_generator = set_sampling_function(generator_model=generator_model)
    print '%.2f SEC '%(time()-t)

    # evaluator
    print 'COMPILING EVALUATION FUNCTION '
    t = time()
    evaluator = set_evaluation_function(generator_model=generator_model)
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
    train_mse_list                = []
    train_positive_score_list     = []
    train_negative_score_list     = []
    train_generator_cost_list     = []
    train_discriminator_cost_list = []
    train_generator_norm_list     = []
    train_discriminator_norm_list = []
    train_hidden_mse_list         = []
    valid_mse_list                = []


    train_batch_count = 0
    for e in xrange(num_epochs):
        window_size      = init_window_size + 5*e
        sequence_size    = feature_size*window_size
        last_seq_idx     = num_train_total_steps-(sequence_size+feature_size)
        train_seq_orders = np_rng.permutation(last_seq_idx)
        train_seq_orders = train_seq_orders[:last_seq_idx-last_seq_idx%batch_size]
        train_seq_orders = train_seq_orders.reshape((-1, batch_size))

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

            # update
            updater_outputs = updater_function(train_source_data,
                                               train_target_data)

            train_generator_cost     = updater_outputs[0].mean()
            train_discriminator_cost = updater_outputs[1].mean()
            train_positive_score     = updater_outputs[2].mean()
            train_negative_score     = updater_outputs[3].mean()
            train_mse                = updater_outputs[4].mean()
            train_hidden_mse         = updater_outputs[5].mean()
            train_generator_norm     = updater_outputs[6].mean()
            train_discriminator_norm = updater_outputs[7].mean()

            train_batch_count += 1

            train_mse_list.append(train_mse)
            train_positive_score_list.append(train_positive_score)
            train_negative_score_list.append(train_negative_score)
            train_generator_cost_list.append(train_generator_cost)
            train_discriminator_cost_list.append(train_discriminator_cost)
            train_generator_norm_list.append(train_generator_norm)
            train_discriminator_norm_list.append(train_discriminator_norm)
            train_hidden_mse_list.append(train_hidden_mse)

            if train_batch_count%10==0:
                print '============{}_LENGTH{}============'.format(model_name, window_size)
                print 'epoch {}, batch_cnt {} => TRAIN data   mse cost  {}'.format(e, train_batch_count, train_mse_list[-1])
                print 'epoch {}, batch_cnt {} => TRAIN hidden mse cost  {}'.format(e, train_batch_count, train_hidden_mse_list[-1])
                print '----------------------------------------------------------'
                print 'epoch {}, batch_cnt {} => TRAIN GAN generator     cost  {}'.format(e, train_batch_count, train_generator_cost_list[-1])
                print 'epoch {}, batch_cnt {} => TRAIN GAN discriminator cost  {}'.format(e, train_batch_count, train_discriminator_cost_list[-1])
                print '----------------------------------------------------------'
                print 'epoch {}, batch_cnt {} => TRAIN GAN input score         {}'.format(e, train_batch_count, train_positive_score_list[-1])
                print 'epoch {}, batch_cnt {} => TRAIN GAN sample score        {}'.format(e, train_batch_count, train_negative_score_list[-1])
                print '----------------------------------------------------------'
                print 'epoch {}, batch_cnt {} => TRAIN GAN discrim.  grad norm {}'.format(e, train_batch_count, train_discriminator_norm_list[-1])
                print 'epoch {}, batch_cnt {} => TRAIN GAN generator grad norm {}'.format(e, train_batch_count, train_generator_norm_list[-1])


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


                valid_mse_list.append(tf_valid_mse/valid_batch_count)
                print '----------------------------------------------------------'
                print 'epoch {}, batch_cnt {} => VALID data   mse cost  {}'.format(e, train_batch_count, valid_mse_list[-1])

                if best_valid>valid_mse_list[-1]:
                    best_valid = valid_mse_list[-1]

            if train_batch_count%1000==0:

                numpy.save(file=model_name+'_train_mse',
                           arr=numpy.asarray(train_mse_list))
                numpy.save(file=model_name+'_train_positive_score',
                           arr=numpy.asarray(train_positive_score_list))
                numpy.save(file=model_name+'_train_negative_score',
                           arr=numpy.asarray(train_negative_score_list))
                numpy.save(file=model_name+'_train_generator_cost',
                           arr=numpy.asarray(train_generator_cost_list))
                numpy.save(file=model_name+'_train_discriminator_cost',
                           arr=numpy.asarray(train_discriminator_cost_list))
                numpy.save(file=model_name+'_train_generator_norm',
                           arr=numpy.asarray(train_generator_norm_list))
                numpy.save(file=model_name+'_train_discriminator_norm',
                           arr=numpy.asarray(train_discriminator_norm_list))
                numpy.save(file=model_name+'_train_hidden_mse',
                           arr=numpy.asarray(train_hidden_mse_list))
                numpy.save(file=model_name+'_valid_mse',
                           arr=numpy.asarray(valid_mse_list))


                num_sec = 100
                sampling_length = num_sec*sampling_rate/feature_size
                seed_input_data = valid_seed_data

                [generated_sequence, ] = sampling_generator(seed_input_data,
                                                            sampling_length)

                sample_data = numpy.swapaxes(generated_sequence, axis1=0, axis2=1)
                sample_data = sample_data.reshape((num_seeds, -1))
                sample_data = sample_data*(1.15*2.**13)
                sample_data = sample_data.astype(numpy.int16)
                save_wavfile(sample_data, model_name+'_sample')


                if best_valid==valid_mse_list[-1]:
                    save_model_params(generator_model+discriminator_feature_model+discriminator_output_model, model_name+'_model.pkl')


if __name__=="__main__":
    feature_size  = 1600
    hidden_size   =  800
    lr=1e-4

    model_name = 'LSTM_GAN_HIDDEN_FF(SINGLE)' \
                + '_FEATURE{}'.format(int(feature_size)) \
                + '_HIDDEN{}'.format(int(hidden_size)) \

    # generator model
    generator_model = set_generator_model(input_size=feature_size,
                                          hidden_size=hidden_size)

    # discriminator model
    discriminator_feature_model = set_discriminator_feature_model(hidden_size=hidden_size,
                                                                  feature_size=200)
    discriminator_output_model = set_discriminator_output_model(feature_size=200)

    # set optimizer
    generator_optimizer     = RmsProp(learning_rate=0.001).update_params
    discriminator_optimizer = RmsProp(learning_rate=0.0001).update_params


    train_model(feature_size=feature_size,
                hidden_size=hidden_size,
                init_window_size=100,
                generator_model=generator_model,
                generator_optimizer=generator_optimizer,
                discriminator_feature_model=discriminator_feature_model,
                discriminator_output_model=discriminator_output_model,
                discriminator_optimizer=discriminator_optimizer,
                num_epochs=10,
                model_name=model_name)
