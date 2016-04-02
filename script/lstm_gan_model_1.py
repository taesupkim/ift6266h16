__author__ = 'KimTS'
import theano
import numpy
from theano import tensor
from data.window import Window
from util.utils import save_wavfile
from layer.activations import Tanh, Logistic, Relu
from layer.layers import LinearLayer, LstmStackLayer, SingleLstmGanForceLayer
from layer.layer_utils import get_tensor_output, get_model_updates, get_lstm_outputs, get_model_gradients
from optimizer.rmsprop import RmsProp
from numpy.random import RandomState
from theano.sandbox.rng_mrg import MRG_RandomStreams
from utils.display import plot_learning_curve
from utils.utils import merge_dicts
theano_rng = MRG_RandomStreams(42)
np_rng = RandomState(42)

floatX = theano.config.floatX
sampling_rate = 16000

from fuel.datasets.hdf5 import H5PYDataset
from fuel.utils import find_in_data_path

class YouTubeAudio(H5PYDataset):
    def __init__(self, youtube_id, **kwargs):
        super(YouTubeAudio, self).__init__(
            file_or_path='/data/lisatmp4/taesup/data/YouTubeAudio/'+youtube_id+'.hdf5',
            which_sets=('train',), **kwargs
        )

def set_train_datastream(feature_size=16000,
                         window_size=100,
                         youtube_id='XqaJ2Ol5cC4_train'):
    data_stream = YouTubeAudio(youtube_id).get_example_stream()
    data_stream = Window(offset=feature_size,
                         source_window=window_size*feature_size,
                         target_window=window_size*feature_size,
                         overlapping=True,
                         data_stream=data_stream)
    return data_stream

def set_valid_datastream(feature_size=16000,
                         window_size=100,
                         youtube_id='XqaJ2Ol5cC4_valid'):
    data_stream = YouTubeAudio(youtube_id).get_example_stream()
    data_stream = Window(offset=feature_size,
                         source_window=window_size*feature_size,
                         target_window=window_size*feature_size,
                         overlapping=False,
                         data_stream=data_stream)
    return data_stream

def set_generator_model(input_size,
                        hidden_size):
    layers = []
    layers.append(SingleLstmGanForceLayer(input_dim=input_size,
                                          hidden_dim=hidden_size,
                                          name='generator_model'))
    return layers

def set_discriminator_model(total_hidden_size):
    layers = []

    layers.append(LinearLayer(input_dim=total_hidden_size,
                              output_dim=total_hidden_size/2,
                              name='discriminator_model_linear0'))
    layers.append(Relu(name='discriminator_model_relu0'))
    layers.append(LinearLayer(input_dim=total_hidden_size/2,
                              output_dim=total_hidden_size/2,
                              name='discriminator_model_linear1'))
    layers.append(Relu(name='discriminator_model_relu1'))
    layers.append(LinearLayer(input_dim=total_hidden_size/2,
                              output_dim=1,
                              name='discriminator_model_linear2'))
    layers.append(Logistic(name='discriminator_model_output'))
    return layers

def set_gan_update_function(generator_model,
                            discriminator_model,
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
    generator_input_data_list = [input_sequence,]

    # get generator output data
    output_data_set = generator_model[0].forward(generator_input_data_list, is_training=True)
    output_sequence = output_data_set[0]
    data_hidden     = output_data_set[1]
    data_cell       = output_data_set[2]
    model_hidden    = output_data_set[3]
    model_cell      = output_data_set[4]

    condition_hidden = data_hidden[:-1]
    condition_cell   = data_cell[:-1]

    condition_hidden = theano.gradient.disconnected_grad(condition_hidden)
    condition_cell   = theano.gradient.disconnected_grad(condition_cell)

    true_hidden = data_hidden[1:]
    true_cell   = data_cell[1:]

    false_hidden = model_hidden[1:]
    false_cell   = model_cell[1:]

    true_pair_hidden = tensor.concatenate([condition_hidden, true_hidden], axis=2)
    true_pair_cell   = tensor.concatenate([condition_cell, true_cell], axis=2)

    false_pair_hidden = tensor.concatenate([condition_hidden, false_hidden], axis=2)
    false_pair_cell   = tensor.concatenate([condition_cell, false_cell], axis=2)

    discriminator_true_score  = get_tensor_output(true_pair_hidden, discriminator_model, is_training=True)
    discriminator_false_score = get_tensor_output(false_pair_hidden, discriminator_model, is_training=True)


    generator_gan_cost = tensor.nnet.binary_crossentropy(output=discriminator_false_score,
                                                         target=tensor.ones_like(discriminator_false_score))

    discriminator_gan_cost = (tensor.nnet.binary_crossentropy(output=discriminator_true_score,
                                                              target=tensor.ones_like(discriminator_true_score)) +
                              tensor.nnet.binary_crossentropy(output=discriminator_false_score,
                                                              target=tensor.zeros_like(discriminator_false_score)))

    # set generator update
    generator_updates_cost = generator_gan_cost.mean()
    generator_updates_dict = get_model_updates(layers=generator_model,
                                               cost=generator_updates_cost,
                                               optimizer=generator_optimizer,
                                               use_grad_clip=generator_grad_clipping)

    generator_gradient_dict  = get_model_gradients(generator_model, generator_updates_cost)
    generator_gradient_norm  = 0.
    for grad in generator_gradient_dict:
        generator_gradient_norm += tensor.sum(grad**2)
    generator_gradient_norm  = tensor.sqrt(generator_gradient_norm)

    # set discriminator update
    discriminator_updates_cost = discriminator_gan_cost.mean()
    discriminator_updates_dict = get_model_updates(layers=discriminator_model,
                                                   cost=discriminator_updates_cost,
                                                   optimizer=discriminator_optimizer,
                                                   use_grad_clip=discriminator_grad_clipping)

    discriminator_gradient_dict  = get_model_gradients(discriminator_model, discriminator_updates_cost)
    discriminator_gradient_norm  = 0.
    for grad in discriminator_gradient_dict:
        discriminator_gradient_norm += tensor.sum(grad**2)
    discriminator_gradient_norm  = tensor.sqrt(discriminator_gradient_norm)

    square_error = tensor.sqr(target_sequence-output_sequence).sum(axis=2)

    # set gan update inputs
    gan_updates_inputs  = [input_sequence,
                           target_sequence]

    # set gan update outputs
    gan_updates_outputs = [generator_gan_cost,
                           discriminator_gan_cost,
                           discriminator_true_score,
                           discriminator_false_score,
                           square_error,
                           generator_gradient_norm,
                           discriminator_gradient_norm,]

    # set gan update function
    gan_updates_function = theano.function(inputs=gan_updates_inputs,
                                           outputs=gan_updates_outputs,
                                           updates=merge_dicts([generator_updates_dict, discriminator_updates_dict]),
                                           on_unused_input='ignore')

    return gan_updates_function

def set_teacher_force_update_function(generator_model,
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
    output_data_set = generator_model[0].forward(generator_input_data_list, is_training=True)
    output_sequence = output_data_set[0]

    # get square error
    square_error = tensor.sqr(target_sequence-output_sequence).sum(axis=2)

    # set generator update
    tf_updates_cost = square_error.mean()
    tf_updates_dict = get_model_updates(layers=generator_model,
                                        cost=tf_updates_cost,
                                        optimizer=generator_optimizer)

    generator_gradient_dict  = get_model_gradients(generator_model, tf_updates_cost)
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
                                          updates=tf_updates_dict,
                                          on_unused_input='ignore')

    return tf_updates_function

def set_evaluation_function(generator_model):

    # input sequence data (time_length * num_samples * input_dims)
    input_sequence  = tensor.tensor3(name='input_sequence',
                                     dtype=floatX)
    target_sequence  = tensor.tensor3(name='target_sequence',
                                    dtype=floatX)
    # set generator input data list
    generator_input_data_list = [input_sequence,]

    # get generator output data
    output_data_set = generator_model[0].forward(generator_input_data_list, is_training=True)
    output_sequence = output_data_set[0]

    # get square error
    square_error = tensor.sqr(target_sequence-output_sequence)

    # set evaluation inputs
    evaluation_inputs  = [input_sequence,
                          target_sequence]

    # set evaluation outputs
    evaluation_outputs = [square_error,]

    # set evaluation function
    evaluation_function = theano.function(inputs=evaluation_inputs,
                                          outputs=evaluation_outputs,
                                          on_unused_input='ignore')

    return evaluation_function


def set_sample_function(generator_model):

    # init input data (num_samples *input_dims)
    init_input_data = tensor.matrix(name='init_input_data',
                                    dtype=floatX)

    # init hidden data (num_layers * num_samples *input_dims)
    init_hidden_data = tensor.tensor3(name='init_hidden_data',
                                      dtype=floatX)

    # init cell data (num_layers * num_samples *input_dims)
    init_cell_data = tensor.tensor3(name='init_cell_data',
                                    dtype=floatX)

    # set generator input data list
    generator_input_data_list = [init_input_data,
                                 init_hidden_data,
                                 init_cell_data]

    # get generator output data
    output_data_set = generator_model[0].forward(generator_input_data_list, is_training=False)
    sample_data = output_data_set[0]
    hidden_data = output_data_set[1]
    cell_data   = output_data_set[2]

    # input data
    sample_function_inputs  = [init_input_data,
                               init_hidden_data,
                               init_cell_data]
    sample_function_outputs = [sample_data,
                               hidden_data,
                               cell_data]

    sample_function = theano.function(inputs=sample_function_inputs,
                                      outputs=sample_function_outputs,
                                      on_unused_input='ignore')
    return sample_function

def train_model(feature_size,
                hidden_size,
                generator_model,
                generator_gan_optimizer,
                generator_tf_optimizer,
                discriminator_model,
                discriminator_optimizer,
                num_epochs,
                model_name):

    # generator updater
    print 'COMPILING TEACHER FORCE UPDATE FUNCTION '
    tf_generator_updater = set_teacher_force_update_function(generator_model=generator_model,
                                                             generator_optimizer=generator_tf_optimizer,
                                                             generator_grad_clipping=0.0)

    # print 'COMPILING GAN UPDATE FUNCTION '
    # gan_generator_updater = set_gan_update_function(generator_model=generator_model,
    #                                                 discriminator_model=discriminator_model,
    #                                                 generator_optimizer=generator_gan_optimizer,
    #                                                 discriminator_optimizer=discriminator_optimizer,
    #                                                 generator_grad_clipping=0.0,
    #                                                 discriminator_grad_clipping=0.0)

    # evaluator
    print 'COMPILING EVALUATION FUNCTION '
    evaluator = set_evaluation_function(generator_model=generator_model)

    # sample generator
    print 'COMPILING SAMPLING FUNCTION '
    sample_generator = set_sample_function(generator_model=generator_model)


    print 'START TRAINING'
    # for each epoch
    tf_generator_train_cost_list = []
    tf_generator_valid_cost_list = []

    gan_generator_cost_list     = []
    gan_discriminator_cost_list = []

    tf_generator_grad_norm_mean = 0.0
    gan_generator_grad_norm_mean = 0.0
    gan_discriminator_grad_norm_mean = 0.0

    init_window_size = 100
    for e in xrange(num_epochs):
        window_size = init_window_size + 5*e

        # set train data stream with proper length (window size)
        train_data_stream = set_train_datastream(feature_size=feature_size,
                                                 window_size=window_size)
        # get train data iterator
        train_data_iterator = train_data_stream.get_epoch_iterator()

        # for each batch
        train_batch_count = 0
        train_batch_size = 0
        train_source_data = []
        train_target_data = []
        for batch_idx, batch_data in enumerate(train_data_iterator):
            # skip the beginning part
            if batch_idx<10000:
                continue

            # init train batch data
            if train_batch_size==0:
                train_source_data = []
                train_target_data = []

            # save source data
            single_data = batch_data[0]
            single_data = single_data.reshape(single_data.shape[0]/feature_size, feature_size)
            train_source_data.append(single_data)

            # save target data
            single_data = batch_data[1]
            single_data = single_data.reshape(single_data.shape[0]/feature_size, feature_size)
            train_target_data.append(single_data)
            train_batch_size += 1


            if train_batch_size<128:
                continue
            else:
                # source data
                train_source_data = numpy.asarray(train_source_data, dtype=floatX)
                train_source_data = numpy.swapaxes(train_source_data, axis1=0, axis2=1)
                # target data
                train_target_data = numpy.asarray(train_target_data, dtype=floatX)
                train_target_data = numpy.swapaxes(train_target_data, axis1=0, axis2=1)
                train_batch_size = 0

            # normalize
            train_source_data = (train_source_data/(1.15*2.**13)).astype(floatX)
            train_target_data = (train_target_data/(1.15*2.**13)).astype(floatX)

            # teacher force update
            tf_update_output = tf_generator_updater(train_source_data, train_target_data)
            tf_square_error = tf_update_output[0].mean()
            tf_generator_grad_norm_mean += tf_update_output[1]

            # gan update
            # gan_update_output = gan_generator_updater(train_source_data, train_target_data)
            # generator_gan_cost               = gan_update_output[0].mean()
            # discriminator_gan_cost           = gan_update_output[1].mean()
            # discriminator_true_score         = gan_update_output[2].mean()
            # discriminator_false_score        = gan_update_output[3].mean()
            # gan_square_error                 = gan_update_output[4].mean()
            # gan_generator_grad_norm_mean    += gan_update_output[5]
            # gan_discriminator_grad_norm_mean+= gan_update_output[6]

            train_batch_count += 1

            print '=============sample length {}============================='.format(window_size)
            # print 'epoch {}, batch_cnt {} => GAN generator cost          {}'.format(e, train_batch_count, generator_gan_cost)
            # print 'epoch {}, batch_cnt {} => GAN discriminator cost      {}'.format(e, train_batch_count, discriminator_gan_cost)
            # print 'epoch {}, batch_cnt {} => GAN input score             {}'.format(e, train_batch_count, discriminator_true_score)
            # print 'epoch {}, batch_cnt {} => GAN sample score            {}'.format(e, train_batch_count, discriminator_false_score)
            print 'epoch {}, batch_cnt {} => TF generator grad norm      {}'.format(e, train_batch_count, tf_generator_grad_norm_mean/train_batch_count)
            # print 'epoch {}, batch_cnt {} => GAN generator grad norm     {}'.format(e, train_batch_count, gan_generator_grad_norm_mean/train_batch_count)
            # print 'epoch {}, batch_cnt {} => GAN discriminator grad norm {}'.format(e, train_batch_count, gan_discriminator_grad_norm_mean/train_batch_count)
            print 'epoch {}, batch_cnt {} => TF generator train mse cost {}'.format(e, train_batch_count, tf_square_error)
            # print 'epoch {}, batch_cnt {} => GAN generator train mse cost{}'.format(e, train_batch_count, gan_square_error)

            sampling_seed_data = []
            # if train_batch_count%10==0:
            if 0:
                # set valid data stream with proper length (window size)
                valid_window_size = window_size
                valid_data_stream = set_valid_datastream(feature_size=feature_size,
                                                         window_size=valid_window_size)
                # get train data iterator
                valid_data_iterator = valid_data_stream.get_epoch_iterator()

                # for each batch
                valid_batch_count = 0
                valid_batch_size  = 0
                valid_source_data = []
                valid_target_data = []
                valid_cost_mean = 0.0
                for batch_idx, batch_data in enumerate(valid_data_iterator):
                    if valid_batch_size==0:
                        valid_source_data = []
                        valid_target_data = []

                    # source data
                    single_data = batch_data[0]
                    single_data = single_data.reshape(single_data.shape[0]/feature_size, feature_size)
                    valid_source_data.append(single_data)

                    # target data
                    single_data = batch_data[1]
                    single_data = single_data.reshape(single_data.shape[0]/feature_size, feature_size)
                    valid_target_data.append(single_data)

                    valid_batch_size += 1

                    if valid_batch_size<128:
                        continue
                    else:
                        # source data
                        valid_source_data = numpy.asarray(valid_source_data, dtype=floatX)
                        valid_source_data = numpy.swapaxes(valid_source_data, axis1=0, axis2=1)
                        # target data
                        valid_target_data = numpy.asarray(valid_target_data, dtype=floatX)
                        valid_target_data = numpy.swapaxes(valid_target_data, axis1=0, axis2=1)
                        valid_batch_size = 0

                    # normalize
                    valid_source_data = (valid_source_data/(1.15*2.**13)).astype(floatX)
                    valid_target_data = (valid_target_data/(1.15*2.**13)).astype(floatX)

                    generator_evaluator_output = evaluator(valid_source_data, valid_target_data)
                    generator_valid_cost = generator_evaluator_output[0].mean()

                    valid_cost_mean += generator_valid_cost
                    valid_batch_count += 1

                    if valid_batch_count>1000:
                        sampling_seed_data = valid_source_data
                        break

                valid_cost_mean = valid_cost_mean/valid_batch_count

                print 'epoch {}, batch_cnt {} => generator valid mse cost    {}'.format(e, train_batch_count, valid_cost_mean)

            #     generator_train_cost_list.append(generator_train_cost)
            #     generator_valid_cost_list.append(valid_cost_mean)
            #
            #     plot_learning_curve(cost_values=[generator_train_cost_list, generator_valid_cost_list],
            #                         cost_names=['Train Cost', 'Valid Cost'],
            #                         save_as=model_name+'_model_cost.png',
            #                         legend_pos='upper left')
            #
            # if train_batch_count%100==0:
            #     num_samples = 10
            #     num_sec     = 10
            #     sampling_length = num_sec*sampling_rate/feature_size
            #
            #     curr_input_data  = sampling_seed_data[0][:num_samples]
            #     prev_hidden_data = np_rng.normal(size=(num_layers, num_samples, hidden_size)).astype(floatX)
            #     prev_hidden_data = numpy.tanh(prev_hidden_data)
            #     output_data      = numpy.zeros(shape=(sampling_length, num_samples, feature_size))
            #     for s in xrange(sampling_length):
            #
            #
            #         generator_input = [curr_input_data,
            #                            prev_hidden_data,]
            #
            #         [curr_input_data, prev_hidden_data] = generator_sampler(*generator_input)
            #
            #         output_data[s] = curr_input_data
            #     sample_data = numpy.swapaxes(output_data, axis1=0, axis2=1)
            #     sample_data = sample_data.reshape((num_samples, -1))
            #     sample_data = sample_data*(1.15*2.**13)
            #     sample_data = sample_data.astype(numpy.int16)
            #     save_wavfile(sample_data, model_name+'_sample')

if __name__=="__main__":
    feature_size  = 160
    hidden_size   = 100

    model_name = 'lstm_gan' \
                 + '_FEATURE{}'.format(int(feature_size)) \
                 + '_HIDDEN{}'.format(int(hidden_size)) \

    # generator model
    generator_model = set_generator_model(input_size=feature_size,
                                          hidden_size=hidden_size)

    # discriminator model
    discriminator_model = set_discriminator_model(total_hidden_size=hidden_size*2)

    # set optimizer
    tf_generator_optimizer      = RmsProp(learning_rate=0.001).update_params
    gan_generator_optimizer     = RmsProp(learning_rate=0.000).update_params
    gan_discriminator_optimizer = RmsProp(learning_rate=0.000).update_params


    train_model(feature_size=feature_size,
                hidden_size=hidden_size,
                generator_model=generator_model,
                generator_gan_optimizer=gan_generator_optimizer,
                generator_tf_optimizer=tf_generator_optimizer,
                discriminator_model=discriminator_model,
                discriminator_optimizer=gan_discriminator_optimizer,
                num_epochs=10,
                model_name=model_name)
