__author__ = 'KimTS'
import theano
import numpy
from theano import tensor
from data.window import Window
from util.utils import save_wavfile
from layer.activations import Tanh, Logistic, Relu
from layer.layers import LinearLayer, SingleLstmGanForceLayer, SingleLstmLayer
from layer.layer_utils import get_tensor_output, get_model_updates, get_model_gradients
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

def set_generator_rnn_model(input_size,
                            hidden_size):
    layers = []
    layers.append(SingleLstmLayer(input_dim=input_size,
                                  hidden_dim=hidden_size,
                                  name='generator_rnn_model'))
    return layers

def set_generator_output_model(hidden_size,
                               input_size):
    layers = []
    layers.append(LinearLayer(input_dim=hidden_size,
                              output_dim=input_size,
                              name='generator_output_model_linear'))
    layers.append(Tanh(name='generator_output_model_tanh'))
    return layers

def set_discriminator_rnn_model(input_size,
                                hidden_size):
    layers = []
    layers.append(SingleLstmLayer(input_dim=input_size,
                                  hidden_dim=hidden_size,
                                  name='discriminator_rnn_model'))
    return layers

def set_discriminator_output_model(hidden_size):
    layers = []
    layers.append(LinearLayer(input_dim=hidden_size,
                              output_dim=hidden_size/2,
                              name='discriminator_output_model_linear0'))
    layers.append(Tanh(name='discriminator_output_model_tanh1'))
    layers.append(LinearLayer(input_dim=hidden_size/2,
                              output_dim=1,
                              name='discriminator_output_model_linear1'))
    layers.append(Logistic(name='discriminator_output_model_prob1'))
    return layers

def set_gan_update_function(generator_rnn_model,
                            generator_output_model,
                            discriminator_rnn_model,
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
    generator_input_data_list = [input_sequence,]

    # get generator output data
    generator_output = generator_rnn_model[0].forward(generator_input_data_list, is_training=True)
    generator_hidden = generator_output[0]
    generator_cell   = generator_output[1]

    generator_sample = get_tensor_output(generator_hidden, generator_output_model, is_training=True)

    generator_hidden = theano.gradient.disconnected_grad(generator_hidden)

    positive_pair = tensor.concatenate([generator_hidden, target_sequence], axis=2)
    negative_pair = tensor.concatenate([generator_hidden, generator_sample], axis=2)

    # set generator input data list
    discriminator_input_data_list = [positive_pair,]
    discriminator_output = discriminator_rnn_model[0].forward(discriminator_input_data_list, is_training=True)
    positive_hidden = discriminator_output[0]
    positive_cell   = discriminator_output[1]
    positive_score  = get_tensor_output(positive_hidden, discriminator_output_model, is_training=True)

    discriminator_input_data_list = [negative_pair,]
    discriminator_output = discriminator_rnn_model[0].forward(discriminator_input_data_list, is_training=True)
    negative_hidden = discriminator_output[0]
    negative_cell   = discriminator_output[1]
    negative_score  = get_tensor_output(negative_hidden, discriminator_output_model, is_training=True)


    generator_gan_cost = tensor.nnet.binary_crossentropy(output=negative_score,
                                                         target=tensor.ones_like(negative_score))

    discriminator_gan_cost = (tensor.nnet.binary_crossentropy(output=positive_score,
                                                              target=tensor.ones_like(positive_score)) +
                              tensor.nnet.binary_crossentropy(output=negative_score,
                                                              target=tensor.zeros_like(negative_score)))

    # set generator update
    generator_updates_cost = generator_gan_cost.mean()
    generator_updates_dict = get_model_updates(layers=generator_rnn_model+generator_output_model,
                                               cost=generator_updates_cost,
                                               optimizer=generator_optimizer,
                                               use_grad_clip=generator_grad_clipping)

    generator_gradient_dict  = get_model_gradients(generator_rnn_model+generator_output_model, generator_updates_cost)
    generator_gradient_norm  = 0.
    for grad in generator_gradient_dict:
        generator_gradient_norm += tensor.sum(grad**2)
    generator_gradient_norm  = tensor.sqrt(generator_gradient_norm)

    # set discriminator update
    discriminator_updates_cost = discriminator_gan_cost.mean()
    discriminator_updates_dict = get_model_updates(layers=discriminator_rnn_model+discriminator_output_model,
                                                   cost=discriminator_updates_cost,
                                                   optimizer=discriminator_optimizer,
                                                   use_grad_clip=discriminator_grad_clipping)

    discriminator_gradient_dict  = get_model_gradients(discriminator_rnn_model+discriminator_output_model, discriminator_updates_cost)
    discriminator_gradient_norm  = 0.
    for grad in discriminator_gradient_dict:
        discriminator_gradient_norm += tensor.sum(grad**2)
    discriminator_gradient_norm  = tensor.sqrt(discriminator_gradient_norm)

    square_error = tensor.sqr(target_sequence-generator_sample).sum(axis=2)

    # set gan update inputs
    gan_updates_inputs  = [input_sequence,
                           target_sequence]

    # set gan update outputs
    gan_updates_outputs = [generator_gan_cost,
                           discriminator_gan_cost,
                           positive_score,
                           negative_score,
                           square_error,
                           generator_gradient_norm,
                           discriminator_gradient_norm,]

    # set gan update function
    gan_updates_function = theano.function(inputs=gan_updates_inputs,
                                           outputs=gan_updates_outputs,
                                           updates=merge_dicts([generator_updates_dict, discriminator_updates_dict]),
                                           on_unused_input='ignore')

    return gan_updates_function

def set_tf_update_function(generator_rnn_model,
                           generator_output_model,
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
    generator_hidden = generator_output[0]
    generator_cell   = generator_output[1]

    generator_sample = get_tensor_output(generator_hidden, generator_output_model, is_training=True)

    # get square error
    square_error = tensor.sqr(target_sequence-generator_sample).sum(axis=2)

    # set generator update
    tf_updates_cost = square_error.mean()
    tf_updates_dict = get_model_updates(layers=generator_rnn_model+generator_output_model,
                                        cost=tf_updates_cost,
                                        optimizer=generator_optimizer)

    generator_gradient_dict  = get_model_gradients(generator_rnn_model+generator_output_model, tf_updates_cost)
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

def set_evaluation_function(generator_rnn_model,
                            generator_output_model):

    # input sequence data (time_length * num_samples * input_dims)
    input_sequence  = tensor.tensor3(name='input_sequence',
                                     dtype=floatX)
    target_sequence  = tensor.tensor3(name='target_sequence',
                                    dtype=floatX)
    # set generator input data list
    generator_input_data_list = [input_sequence,]

    # get generator output data
    generator_output = generator_rnn_model[0].forward(generator_input_data_list, is_training=True)
    generator_hidden = generator_output[0]
    generator_cell   = generator_output[1]

    generator_sample = get_tensor_output(generator_hidden, generator_output_model, is_training=True)

    # get square error
    square_error = tensor.sqr(target_sequence-generator_sample).sum(axis=2)

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


def set_sample_function(generator_rnn_model,
                        generator_output_model):

    # init input data (num_samples *input_dims)
    init_input_data = tensor.matrix(name='init_input_data',
                                    dtype=floatX)

    # init hidden data (num_samples *input_dims)
    init_hidden_data = tensor.matrix(name='init_hidden_data',
                                     dtype=floatX)

    # init cell data (num_samples *input_dims)
    init_cell_data = tensor.matrix(name='init_cell_data',
                                   dtype=floatX)

    # set generator input data list
    generator_input_data_list = [init_input_data,
                                 init_hidden_data,
                                 init_cell_data]

    # get generator output data
    generator_output = generator_rnn_model[0].forward(generator_input_data_list, is_training=False)
    generator_hidden = generator_output[0]
    generator_cell   = generator_output[1]

    generator_sample = get_tensor_output(generator_hidden, generator_output_model, is_training=True)
    # input data
    sample_function_inputs  = [init_input_data,
                               init_hidden_data,
                               init_cell_data]
    sample_function_outputs = [generator_sample,
                               generator_hidden,
                               generator_cell]

    sample_function = theano.function(inputs=sample_function_inputs,
                                      outputs=sample_function_outputs,
                                      on_unused_input='ignore')
    return sample_function

def train_model(feature_size,
                hidden_size,
                generator_rnn_model,
                generator_output_model,
                generator_gan_optimizer,
                generator_tf_optimizer,
                discriminator_rnn_model,
                discriminator_output_model,
                discriminator_optimizer,
                num_epochs,
                model_name):

    # generator updater
    print 'COMPILING GAN UPDATE FUNCTION '
    gan_updater = set_gan_update_function(generator_rnn_model=generator_rnn_model,
                                          generator_output_model=generator_output_model,
                                          discriminator_rnn_model=discriminator_rnn_model,
                                          discriminator_output_model=discriminator_output_model,
                                          generator_optimizer=generator_gan_optimizer,
                                          discriminator_optimizer=discriminator_optimizer,
                                          generator_grad_clipping=.0,
                                          discriminator_grad_clipping=.0)

    print 'COMPILING TF UPDATE FUNCTION '
    tf_updater = set_tf_update_function(generator_rnn_model=generator_rnn_model,
                                        generator_output_model=generator_output_model,
                                        generator_optimizer=generator_tf_optimizer,
                                        generator_grad_clipping=.0)

    # evaluator
    print 'COMPILING EVALUATION FUNCTION '
    evaluator = set_evaluation_function(generator_rnn_model=generator_rnn_model,
                                        generator_output_model=generator_output_model)

    # sample generator
    print 'COMPILING SAMPLING FUNCTION '
    sample_generator = set_sample_function(generator_rnn_model=generator_rnn_model,
                                           generator_output_model=generator_output_model)


    print 'START TRAINING'
    # for each epoch
    tf_mse_list                = []
    tf_generator_grad_list     = []

    gan_generator_grad_list     = []
    gan_generator_cost_list     = []
    gan_discriminator_grad_list = []
    gan_discriminator_cost_list = []
    gan_true_score_list         = []
    gan_false_score_list        = []
    gan_mse_list                = []

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

            # tf update
            tf_update_output = tf_updater(train_source_data,
                                          train_target_data)
            tf_square_error        = tf_update_output[0].mean()
            tf_generator_grad_norm = tf_update_output[1]

            # gan update
            gan_update_output = gan_updater(train_source_data,
                                            train_target_data)
            generator_gan_cost               = gan_update_output[0].mean()
            discriminator_gan_cost           = gan_update_output[1].mean()
            discriminator_true_score         = gan_update_output[2].mean()
            discriminator_false_score        = gan_update_output[3].mean()
            gan_square_error                 = gan_update_output[4].mean()
            gan_generator_grad_norm          = gan_update_output[5]
            gan_discriminator_grad_norm      = gan_update_output[6]

            train_batch_count += 1

            tf_generator_grad_list.append(tf_generator_grad_norm)
            tf_mse_list.append(tf_square_error)

            gan_generator_grad_list.append(gan_generator_grad_norm)
            gan_generator_cost_list.append(generator_gan_cost)

            gan_discriminator_grad_list.append(gan_discriminator_grad_norm)
            gan_discriminator_cost_list.append(discriminator_gan_cost)

            gan_true_score_list.append(discriminator_true_score)
            gan_false_score_list.append(discriminator_false_score)

            gan_mse_list.append(gan_square_error)

            if train_batch_count%10==0:
                print '=============sample length {}============================='.format(window_size)
                print 'epoch {}, batch_cnt {} => TF  generator mse cost  {}'.format(e, train_batch_count, tf_mse_list[-1])
                print 'epoch {}, batch_cnt {} => GAN generator mse cost  {}'.format(e, train_batch_count, gan_mse_list[-1])
                print '----------------------------------------------------------'
                print 'epoch {}, batch_cnt {} => GAN generator     cost  {}'.format(e, train_batch_count, gan_generator_cost_list[-1])
                print 'epoch {}, batch_cnt {} => GAN discriminator cost  {}'.format(e, train_batch_count, gan_discriminator_cost_list[-1])
                print '----------------------------------------------------------'
                print 'epoch {}, batch_cnt {} => GAN input score         {}'.format(e, train_batch_count, gan_true_score_list[-1])
                print 'epoch {}, batch_cnt {} => GAN sample score        {}'.format(e, train_batch_count, gan_false_score_list[-1])
                print '----------------------------------------------------------'
                print 'epoch {}, batch_cnt {} => GAN discrim.  grad norm {}'.format(e, train_batch_count, gan_discriminator_grad_list[-1])
                print 'epoch {}, batch_cnt {} => GAN generator grad norm {}'.format(e, train_batch_count, gan_generator_grad_list[-1])
                print '----------------------------------------------------------'
                print 'epoch {}, batch_cnt {} => TF  generator grad norm {}'.format(e, train_batch_count, tf_generator_grad_list[-1])


            if train_batch_count%100==0:
                numpy.save(file=model_name+'tf_mse',
                           arr=numpy.asarray(tf_mse_list))
                numpy.save(file=model_name+'tf_gen_grad',
                           arr=numpy.asarray(tf_generator_grad_list))
                numpy.save(file=model_name+'gan_mse',
                           arr=numpy.asarray(gan_mse_list))
                numpy.save(file=model_name+'gan_gen_cost',
                           arr=numpy.asarray(gan_generator_cost_list))
                numpy.save(file=model_name+'gan_disc_cost',
                           arr=numpy.asarray(gan_true_score_list))
                numpy.save(file=model_name+'gan_input_score',
                           arr=numpy.asarray(gan_true_score_list))
                numpy.save(file=model_name+'gan_sample_score',
                           arr=numpy.asarray(gan_false_score_list))
                numpy.save(file=model_name+'gan_gen_grad',
                           arr=numpy.asarray(gan_generator_grad_list))
                numpy.save(file=model_name+'gan_disc_grad',
                           arr=numpy.asarray(gan_discriminator_grad_list))

            num_samples = 10
            if train_batch_count%100==0:
                valid_data_stream = set_valid_datastream(feature_size=feature_size,
                                                         window_size=1)
                # get train data iterator
                valid_data_iterator = valid_data_stream.get_epoch_iterator()

                # for each batch
                valid_batch_size  = 0
                sampling_seed_data = []
                for batch_idx, batch_data in enumerate(valid_data_iterator):
                    # source data
                    single_data = batch_data[0]
                    single_data = single_data.reshape(single_data.shape[0]/feature_size, feature_size)
                    sampling_seed_data.append(single_data)

                    valid_batch_size += 1

                    if valid_batch_size<num_samples:
                        continue
                    else:
                        # source data
                        sampling_seed_data = numpy.asarray(sampling_seed_data, dtype=floatX)

                    # normalize
                    sampling_seed_data = (sampling_seed_data/(1.15*2.**13)).astype(floatX)
                    break

                num_sec     = 10
                sampling_length = num_sec*sampling_rate/feature_size

                curr_input_data  = sampling_seed_data.reshape(num_samples, feature_size)
                prev_hidden_data = np_rng.normal(size=(num_samples, hidden_size)).astype(floatX)
                prev_hidden_data = numpy.tanh(prev_hidden_data)
                prev_cell_data   = np_rng.normal(size=(num_samples, hidden_size)).astype(floatX)
                output_data      = numpy.zeros(shape=(sampling_length, num_samples, feature_size))
                for s in xrange(sampling_length):
                    generator_input = [curr_input_data,
                                       prev_hidden_data,
                                       prev_cell_data]

                    [curr_input_data, prev_hidden_data, prev_cell_data] = sample_generator(*generator_input)
                    output_data[s] = curr_input_data
                sample_data = numpy.swapaxes(output_data, axis1=0, axis2=1)
                sample_data = sample_data.reshape((num_samples, -1))
                sample_data = sample_data*(1.15*2.**13)
                sample_data = sample_data.astype(numpy.int16)
                save_wavfile(sample_data, model_name+'_sample')


if __name__=="__main__":
    feature_size  = 1600
    hidden_size   = 1000
    lr=1e-5

    model_name = 'lstm_gan' \
                + '_FEATURE{}'.format(int(feature_size)) \
                 + '_HIDDEN{}'.format(int(hidden_size)) \

    # generator model
    generator_rnn_model    = set_generator_rnn_model(input_size=feature_size,
                                                     hidden_size=hidden_size)
    generator_output_model = set_generator_output_model(input_size=feature_size,
                                                        hidden_size=hidden_size)
    # discriminator model
    discriminator_rnn_model = set_discriminator_rnn_model(input_size=feature_size+hidden_size,
                                                          hidden_size=hidden_size)
    discriminator_output_model = set_discriminator_output_model(hidden_size=hidden_size)

    # set optimizer
    tf_generator_optimizer      = RmsProp(learning_rate=lr).update_params
    gan_generator_optimizer     = RmsProp(learning_rate=lr).update_params
    gan_discriminator_optimizer = RmsProp(learning_rate=lr).update_params


    train_model(feature_size=feature_size,
                hidden_size=hidden_size,
                generator_rnn_model=generator_rnn_model,
                generator_output_model=generator_output_model,
                generator_gan_optimizer=gan_generator_optimizer,
                generator_tf_optimizer=tf_generator_optimizer,
                discriminator_rnn_model=discriminator_rnn_model,
                discriminator_output_model=discriminator_output_model,
                discriminator_optimizer=gan_discriminator_optimizer,
                num_epochs=10,
                model_name=model_name)
