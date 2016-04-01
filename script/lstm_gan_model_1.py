__author__ = 'KimTS'
import theano
import numpy
from theano import tensor
from data.window import Window
from util.utils import save_wavfile
from layer.activations import Tanh, Logistic, Relu
from layer.layers import LinearLayer, LstmStackLayer, LstmLoopStackLayer
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

def set_datastream(feature_size=16000,
                   window_size=100,
                   youtube_id='XqaJ2Ol5cC4'):
    from fuel.datasets.youtube_audio import YouTubeAudio
    data_stream = YouTubeAudio(youtube_id).get_example_stream()
    data_stream = Window(offset=feature_size,
                         source_window=window_size*feature_size,
                         target_window=window_size*feature_size,
                         overlapping=False,
                         data_stream=data_stream)
    return data_stream

def set_generator_recurrent_model(input_size,
                                  hidden_size,
                                  num_layers):
    layers = []
    layers.append(LstmLoopStackLayer(input_dim=input_size,
                                     hidden_dim=hidden_size,
                                     num_rnn_layers=num_layers,
                                     num_lin_layers=num_layers-1,
                                     name='generator_rnn_model'))
    return layers

def set_discriminator_recurrent_model(input_size,
                                      hidden_size,
                                      num_layers):
    layers = []
    layers.append(LstmStackLayer(input_dim=input_size,
                                 hidden_dim=hidden_size,
                                 num_layers=num_layers,
                                 name='discriminator_rnn_model'))
    return layers

def set_discriminator_output_model(input_size):
    layers = []

    layers.append(LinearLayer(input_dim=input_size,
                              output_dim=input_size/2,
                              name='discriminator_output_model_linear0'))
    layers.append(Relu(name='discriminator_output_model_relu0'))

    layers.append(LinearLayer(input_dim=input_size/2,
                              output_dim=1,
                              name='discriminator_output_model_linear1'))
    layers.append(Logistic(name='discriminator_output_model_logistic2'))
    return layers

def set_generator_update_function(generator_rnn_model,
                                  discriminator_rnn_model,
                                  discriminator_output_model,
                                  generator_optimizer,
                                  grad_clipping):
    # init input data (num_samples *input_dims)
    init_input_data = tensor.matrix(name='init_input_data',
                                    dtype=floatX)

    # init hidden data (num_layers * num_samples *input_dims)
    init_hidden_data = tensor.tensor3(name='init_hidden_data',
                                      dtype=floatX)

    # init cell data (num_layers * num_samples *input_dims)
    init_cell_data = tensor.tensor3(name='init_cell_data',
                                    dtype=floatX)

    # sampling length
    sampling_length = tensor.scalar(name='sampling_length',
                                    dtype='int32')

    # set generator input data list
    generator_input_data_list = [init_input_data,
                                 init_hidden_data,
                                 init_cell_data,
                                 sampling_length]

    # get generator output data
    output_data_set = generator_rnn_model[0].forward(generator_input_data_list, is_training=True)
    sample_data = output_data_set[0]

    # set discriminator input data list
    discriminator_sample_data_list = [sample_data,]

    # get discriminator hidden data
    discriminator_sample_hidden_data = discriminator_rnn_model[0].forward(discriminator_sample_data_list, is_training=True)[0]
    discriminator_sample_hidden_data = discriminator_sample_hidden_data.dimshuffle(0, 2, 1, 3)
    discriminator_sample_hidden_data = discriminator_sample_hidden_data.flatten(ndim=3)

    # get discriminator output data
    discriminator_sample_score_data = get_tensor_output(input=discriminator_sample_hidden_data,
                                                        layers=discriminator_output_model,
                                                        is_training=True)[-1]

    # get cost based on discriminator (binary cross-entropy over all data)
    # sum over generator cost over time_length and output_dims, then mean over samples
    generator_cost = tensor.nnet.binary_crossentropy(output=discriminator_sample_score_data,
                                                     target=tensor.ones_like(discriminator_sample_score_data))

    # set generator update
    generator_updates_cost = generator_cost.mean()
    generator_updates_dict = get_model_updates(layers=generator_rnn_model,
                                               cost=generator_updates_cost,
                                               optimizer=generator_optimizer,
                                               use_grad_clip=grad_clipping)

    gradient_dict  = get_model_gradients(generator_rnn_model, generator_updates_cost)
    gradient_norm  = 0.
    for grad in gradient_dict:
        gradient_norm += tensor.sum(grad**2)
        gradient_norm  = tensor.sqrt(gradient_norm)

    # set generator update inputs
    generator_updates_inputs  = [init_input_data,
                                 init_hidden_data,
                                 init_cell_data,
                                 sampling_length]

    # set generator update outputs
    generator_updates_outputs = [discriminator_sample_score_data,
                                 generator_cost,
                                 gradient_norm]

    # set generator update function
    generator_updates_function = theano.function(inputs=generator_updates_inputs,
                                                 outputs=generator_updates_outputs,
                                                 updates=generator_updates_dict,
                                                 on_unused_input='ignore')

    return generator_updates_function

def set_discriminator_update_function(generator_rnn_model,
                                      discriminator_rnn_model,
                                      discriminator_output_model,
                                      discriminator_optimizer,
                                      grad_clipping):

    # input data (time_length * num_samples *input_dims)
    input_data = tensor.tensor3(name='input_data',
                                dtype=floatX)

    # init input data (num_samples *input_dims)
    init_input_data = tensor.matrix(name='init_input_data',
                                    dtype=floatX)

    # init hidden data (num_layers * num_samples *input_dims)
    init_hidden_data = tensor.tensor3(name='init_hidden_data',
                                      dtype=floatX)

    # init cell data (num_layers * num_samples *input_dims)
    init_cell_data = tensor.tensor3(name='init_cell_data',
                                    dtype=floatX)

    # sampling length
    sampling_length = input_data.shape[0]

    # set generator input data list
    generator_input_data_list = [init_input_data,
                                 init_hidden_data,
                                 init_cell_data,
                                 sampling_length]

    # get generator sampled output data
    output_data_set = generator_rnn_model[0].forward(generator_input_data_list, is_training=True)
    sample_data = output_data_set[0]

    # set discriminator real input data list
    discriminator_input_data_list = [input_data,]

    # set discriminator sample input data list
    discriminator_sample_data_list = [sample_data,]

    # get discriminator input cost data
    discriminator_input_hidden_data = discriminator_rnn_model[0].forward(discriminator_input_data_list, is_training=True)[0]
    discriminator_input_hidden_data = discriminator_input_hidden_data.dimshuffle(0, 2, 1, 3)
    discriminator_input_hidden_data = discriminator_input_hidden_data.flatten(ndim=3)
    discriminator_input_score_data  = get_tensor_output(input=discriminator_input_hidden_data,
                                                        layers=discriminator_output_model,
                                                        is_training=True)[-1]

    # get discriminator sample cost data
    discriminator_sample_hidden_data = discriminator_rnn_model[0].forward(discriminator_sample_data_list, is_training=True)[0]
    discriminator_sample_hidden_data = discriminator_sample_hidden_data.dimshuffle(0, 2, 1, 3)
    discriminator_sample_hidden_data = discriminator_sample_hidden_data.flatten(ndim=3)
    discriminator_sample_score_data  = get_tensor_output(input=discriminator_sample_hidden_data,
                                                         layers=discriminator_output_model,
                                                         is_training=True)[-1]


    # get cost based on discriminator (binary cross-entropy over all data)
    # sum over discriminator cost over time_length and output_dims, then mean over samples
    discriminator_cost = (tensor.nnet.binary_crossentropy(output=discriminator_input_score_data,
                                                          target=tensor.ones_like(discriminator_input_score_data)) +
                          tensor.nnet.binary_crossentropy(output=discriminator_sample_score_data,
                                                          target=tensor.zeros_like(discriminator_sample_score_data)))

    # set discriminator update
    discriminator_updates_cost = discriminator_cost.mean()
    discriminator_updates_dict = get_model_updates(layers=discriminator_rnn_model+discriminator_output_model,
                                                   cost=discriminator_updates_cost,
                                                   optimizer=discriminator_optimizer,
                                                   use_grad_clip=grad_clipping)

    gradient_dict  = get_model_gradients(discriminator_rnn_model+discriminator_output_model, discriminator_updates_cost)
    gradient_norm  = 0.
    for grad in gradient_dict:
        gradient_norm += tensor.sum(grad**2)
        gradient_norm  = tensor.sqrt(gradient_norm)

    # set discriminator update inputs
    discriminator_updates_inputs  = [input_data,
                                     init_input_data,
                                     init_hidden_data,
                                     init_cell_data]

    # set discriminator update outputs
    discriminator_updates_outputs = [discriminator_input_score_data,
                                     discriminator_sample_score_data,
                                     discriminator_cost,
                                     gradient_norm]

    # set discriminator update function
    discriminator_updates_function = theano.function(inputs=discriminator_updates_inputs,
                                                     outputs=discriminator_updates_outputs,
                                                     updates=discriminator_updates_dict,
                                                     on_unused_input='ignore')

    return discriminator_updates_function

def set_sample_generation_function(generator_rnn_model):

    # init input data (num_samples *input_dims)
    init_input_data = tensor.matrix(name='init_input_data',
                                    dtype=floatX)

    # init hidden data (num_layers * num_samples *input_dims)
    init_hidden_data = tensor.tensor3(name='init_hidden_data',
                                      dtype=floatX)

    # init cell data (num_layers * num_samples *input_dims)
    init_cell_data = tensor.tensor3(name='init_cell_data',
                                    dtype=floatX)

    # sampling length
    sampling_length = tensor.scalar(name='sampling_length',
                                    dtype='int32')
    # set generator input data list
    generator_input_data_list = [init_input_data,
                                 init_hidden_data,
                                 init_cell_data,
                                 sampling_length]

    # get generator output data
    output_data_set = generator_rnn_model[0].forward(generator_input_data_list, is_training=True)
    sample_data = output_data_set[0]

    # input data
    generation_function_inputs  = [init_input_data,
                                   init_hidden_data,
                                   init_cell_data,
                                   sampling_length]
    generation_function_outputs = [sample_data, ]

    generation_function = theano.function(inputs=generation_function_inputs,
                                          outputs=generation_function_outputs,
                                          on_unused_input='ignore')
    return generation_function

def train_model(feature_size,
                hidden_size,
                num_layers,
                generator_rnn_model,
                generator_optimizer,
                discriminator_rnn_model,
                discriminator_output_model,
                discriminator_optimizer,
                num_epochs,
                model_name):

    # generator updater
    print 'DEBUGGING GENERATOR UPDATE FUNCTION '
    generator_updater = set_generator_update_function(generator_rnn_model=generator_rnn_model,
                                                      discriminator_rnn_model=discriminator_rnn_model,
                                                      discriminator_output_model=discriminator_output_model,
                                                      generator_optimizer=generator_optimizer,
                                                      grad_clipping=0.0)

    # discriminator updater
    print 'DEBUGGING DISCRIMINATOR UPDATE FUNCTION '
    discriminator_updater = set_discriminator_update_function(generator_rnn_model=generator_rnn_model,
                                                              discriminator_rnn_model=discriminator_rnn_model,
                                                              discriminator_output_model=discriminator_output_model,
                                                              discriminator_optimizer=discriminator_optimizer,
                                                              grad_clipping=0.0)

    # sample generator
    print 'DEBUGGING SAMPLE GENERATOR FUNCTION '
    sample_generator = set_sample_generation_function(generator_rnn_model=generator_rnn_model)



    print 'START TRAINING'
    # for each epoch
    generator_cost_list = []
    discriminator_cost_list = []

    generator_grad_norm_mean     = 0.0
    discriminator_grad_norm_mean = 0.0

    init_window_size = 50
    for e in xrange(num_epochs):
        window_size = init_window_size + 5*e

        # set data stream with proper length (window size)
        data_stream = set_datastream(feature_size=feature_size,
                                     window_size=window_size)
        # get data iterator
        data_iterator = data_stream.get_epoch_iterator()

        # for each batch
        batch_count = 0
        batch_size = 0
        source_data = []
        for batch_idx, batch_data in enumerate(data_iterator):
            if batch_size==0:
                source_data = []
            # source data
            single_data = batch_data[0]
            single_data = single_data.reshape(window_size, feature_size)
            source_data.append(single_data)
            batch_size += 1

            if batch_size<128:
                continue
            else:
                source_data = numpy.asarray(source_data, dtype=floatX)
                source_data = numpy.swapaxes(source_data, axis1=0, axis2=1)
                batch_size = 0

            # normalize
            source_data = (source_data/(2.**15)).astype(floatX)

            # set generator initial values
            init_input_data  = source_data[0]
            init_hidden_data = np_rng.normal(size=(num_layers, source_data.shape[1], hidden_size)).astype(floatX)
            init_hidden_data = numpy.tanh(init_hidden_data)
            init_cell_data   = np_rng.normal(size=(num_layers, source_data.shape[1], hidden_size)).astype(floatX)

            # update generator
            generator_updater_input = [init_input_data,
                                       init_hidden_data,
                                       init_cell_data,
                                       window_size]

            generator_updater_output = generator_updater(*generator_updater_input)
            generator_cost = generator_updater_output[1].mean()
            generator_grad_norm = generator_updater_output[-1]

            # update discriminator
            init_input_data  = source_data[0]
            init_hidden_data = np_rng.normal(size=(num_layers, source_data.shape[1], hidden_size)).astype(floatX)
            init_hidden_data = numpy.tanh(init_hidden_data)
            init_cell_data   = np_rng.normal(size=(num_layers, source_data.shape[1], hidden_size)).astype(floatX)

            discriminator_updater_input = [source_data,
                                           init_input_data,
                                           init_hidden_data,
                                           init_cell_data]

            discriminator_updater_output = discriminator_updater(*discriminator_updater_input)
            input_cost_data    = discriminator_updater_output[0]
            sample_cost_data   = discriminator_updater_output[1]
            discriminator_cost = discriminator_updater_output[2].mean()
            discriminator_grad_norm = discriminator_updater_output[-1]

            generator_grad_norm_mean     += generator_grad_norm
            discriminator_grad_norm_mean += discriminator_grad_norm

            batch_count += 1

            if batch_count%10==0:
                print '=============sample length {}============================='.format(window_size)
                print 'epoch {}, batch_cnt {} => generator     cost {}'.format(e, batch_count, generator_cost)
                print 'epoch {}, batch_cnt {} => discriminator cost {}'.format(e, batch_count, discriminator_cost)
                print 'epoch {}, batch_cnt {} => input data    cost {}'.format(e, batch_count, input_cost_data.mean())
                print 'epoch {}, batch_cnt {} => sample data   cost {}'.format(e, batch_count, sample_cost_data.mean())
                print 'epoch {}, batch_cnt {} => generator gradnorm {}'.format(e, batch_count, generator_grad_norm_mean/batch_count)
                print 'epoch {}, batch_cnt {} => discriminator      {}'.format(e, batch_count, generator_grad_norm_mean/batch_count)

                generator_cost_list.append(generator_cost)
                discriminator_cost_list.append(discriminator_cost)
                plot_learning_curve(cost_values=[generator_cost_list, discriminator_cost_list],
                                    cost_names=['Generator Cost', 'Discriminator Cost'],
                                    save_as=model_name+'_model_cost.png',
                                    legend_pos='upper left')

                # plot_learning_curve(cost_values=[input_cost_data.mean(axis=(1, 2)), sample_cost_data.mean(axis=(1, 2))],
                #                     cost_names=['Data Distribution', 'Model Distribution'],
                #                     save_as=model_name+'_seq_cost{}.png'.format(batch_count),
                #                     legend_pos='upper left')


            if batch_count%100==0:
                num_samples = 10
                num_sec     = 10
                sampling_length = num_sec*sampling_rate/feature_size
                # set generator initial values
                init_input_data  = source_data[1][:num_samples].reshape(num_samples, feature_size)
                init_hidden_data = np_rng.normal(size=(num_layers, num_samples, hidden_size)).astype(floatX)
                init_hidden_data = numpy.tanh(init_hidden_data)
                init_cell_data   = np_rng.normal(size=(num_layers, num_samples, hidden_size)).astype(floatX)

                generator_input = [init_input_data,
                                   init_hidden_data,
                                   init_cell_data,
                                   sampling_length]

                sample_data = sample_generator(*generator_input)[0]

                sample_data = numpy.swapaxes(sample_data, axis1=0, axis2=1)
                sample_data = sample_data.reshape((num_samples, -1))
                sample_data = sample_data*(2.**15)
                sample_data = sample_data.astype(numpy.int16)
                save_wavfile(sample_data, model_name+'_sample')

if __name__=="__main__":
    feature_size  = 160
    hidden_size   = 240
    learning_rate = 1e-4
    num_layers    = 3

    model_name = 'lstm_gan_last_cost' \
                 + '_FEATURE{}'.format(int(feature_size)) \
                 + '_HIDDEN{}'.format(int(hidden_size)) \
                 + '_LAYERS{}'.format(int(num_layers)) \

    # generator model
    generator_rnn_model = set_generator_recurrent_model(input_size=feature_size,
                                                        hidden_size=hidden_size,
                                                        num_layers=num_layers)

    # discriminator model
    discriminator_rnn_model = set_discriminator_recurrent_model(input_size=feature_size,
                                                                hidden_size=hidden_size,
                                                                num_layers=num_layers)
    discriminator_output_model = set_discriminator_output_model(input_size=hidden_size*num_layers)

    # set optimizer
    generator_optimizer     = RmsProp(learning_rate=0.01).update_params
    discriminator_optimizer = RmsProp(learning_rate=0.001).update_params


    train_model(feature_size=feature_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                generator_rnn_model=generator_rnn_model,
                generator_optimizer=generator_optimizer,
                discriminator_rnn_model=discriminator_rnn_model,
                discriminator_output_model=discriminator_output_model,
                discriminator_optimizer=discriminator_optimizer,
                num_epochs=10,
                model_name=model_name)
