__author__ = 'KimTS'
import theano
import numpy
from theano import tensor
from data.window import Window
from util.utils import save_wavfile
from layer.activations import Tanh, Logistic
from layer.layers import LinearLayer, LstmLayer, LstmLoopLayer
from layer.layer_utils import get_tensor_output, get_model_updates, get_lstm_outputs
from optimizer.rmsprop import RmsProp
from numpy.random import RandomState
from theano.sandbox.rng_mrg import MRG_RandomStreams
from utils.display import plot_learning_curve
theano_rng = MRG_RandomStreams(42)
np_rng = RandomState(42)

floatX = theano.config.floatX
sampling_rate = 160000

def set_datastream(feature_size=16000,
                   window_size=100,
                   youtube_id='XqaJ2Ol5cC4'):
    from fuel.datasets.youtube_audio import YouTubeAudio
    data_stream = YouTubeAudio(youtube_id).get_example_stream()
    data_stream = Window(offset=feature_size,
                         source_window=window_size*feature_size,
                         target_window=window_size*feature_size,
                         overlapping=True,
                         data_stream=data_stream)
    return data_stream

def set_generator_recurrent_model(input_size,
                                  hidden_size,
                                  max_length):
    layers = []
    layers.append(LstmLoopLayer(input_dim=input_size,
                                hidden_dim=hidden_size,
                                max_time=max_length,
                                name='generator_rnn_model'))
    return layers

def set_discriminator_recurrent_model(input_size,
                                      hidden_size,
                                      num_layers):
    layers = []
    for l in xrange(num_layers):
        layers.append(LstmLayer(input_dim=input_size if l is 0 else hidden_size,
                                hidden_dim=hidden_size,
                                name='discriminator_rnn_layer{}'.format(l)))
    return layers

def set_discriminator_output_model(input_size):
    layers = []
    layers.append(LinearLayer(input_dim=input_size,
                              output_dim=1,
                              name='discriminator_output_layer'))
    layers.append(Logistic(name='discriminator_output_prob'))
    return layers

def set_generator_update_function(generator_rnn_model,
                                  discriminator_rnn_model,
                                  discriminator_output_model,
                                  generator_optimizer,
                                  grad_clipping):
    # init input data (num_samples *input_dims)
    init_input_data = tensor.matrix(name='init_input_data',
                                    dtype=floatX)

    # init hidden data (num_samples *input_dims)
    init_hidden_data = tensor.matrix(name='init_hidden_data',
                                     dtype=floatX)

    # init cell data (num_samples *input_dims)
    init_cell_data = tensor.matrix(name='init_hidden_data',
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
    output_data = generator_rnn_model[0].forward(generator_input_data_list, is_training=True)[0]

    # set discriminator input data list
    discriminator_input_data_list = [output_data,]

    # get discriminator hidden data
    discriminator_hidden_data = get_lstm_outputs(input_list=discriminator_input_data_list,
                                                 layers=discriminator_rnn_model,
                                                 is_training=True)[-1]

    # get discriminator output data
    cost_data = get_tensor_output(input=discriminator_hidden_data,
                                  layers=discriminator_output_model,
                                  is_training=True)

    # get cost based on discriminator (binary cross-entropy over all data)
    generator_cost = tensor.nnet.binary_crossentropy(output=cost_data,
                                                     target=tensor.ones_like(cost_data))

    # sum over generator cost over time_length and output_dims, then mean over samples
    generator_cost = generator_cost.sum(axis=(0, 2))

    # set generator update
    generator_updates_cost = generator_cost.mean()
    generator_updates_dict = get_model_updates(layers=generator_rnn_model,
                                               cost=generator_updates_cost,
                                               optimizer=generator_optimizer,
                                               use_grad_clip=grad_clipping)

    # set generator update inputs
    generator_updates_inputs  = [init_input_data,
                                 init_hidden_data,
                                 init_cell_data,
                                 sampling_length]

    # set generator update outputs
    generator_updates_outputs = [cost_data, generator_cost]

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

    # init hidden data (num_samples *input_dims)
    init_hidden_data = tensor.matrix(name='init_hidden_data',
                                     dtype=floatX)

    # init cell data (num_samples *input_dims)
    init_cell_data = tensor.matrix(name='init_hidden_data',
                                   dtype=floatX)

    # sampling length
    sampling_length = input_data.shape[0]

    # set generator input data list
    generator_input_data_list = [init_input_data,
                                 init_hidden_data,
                                 init_cell_data,
                                 sampling_length]

    # get generator sampled output data
    sample_data = generator_rnn_model[0].forward(generator_input_data_list, is_training=True)[0]


    # set discriminator real input data list
    discriminator_input_data_list = [input_data,]

    # set discriminator sample input data list
    discriminator_sample_data_list = [sample_data,]

    # get discriminator input cost data
    input_cost_data = get_tensor_output(input=get_lstm_outputs(input_list=discriminator_input_data_list,
                                                               layers=discriminator_rnn_model,
                                                               is_training=True)[-1],
                                        layers=discriminator_output_model,
                                        is_training=True)

    # get discriminator sample cost data
    sample_cost_data = get_tensor_output(input=get_lstm_outputs(input_list=discriminator_sample_data_list,
                                                                layers=discriminator_rnn_model,
                                                                is_training=True)[-1],
                                        layers=discriminator_output_model,
                                        is_training=True)

    # get cost based on discriminator (binary cross-entropy over all data)
    discriminator_cost = (tensor.nnet.binary_crossentropy(output=input_cost_data,
                                                          target=tensor.ones_like(input_cost_data)) +
                          tensor.nnet.binary_crossentropy(output=sample_cost_data,
                                                          target=tensor.ones_like(sample_cost_data)))

    # sum over discriminator cost over time_length and output_dims, then mean over samples
    discriminator_cost = discriminator_cost.sum(axis=(0, 2))

    # set discriminator update
    discriminator_updates_cost = discriminator_cost.mean()
    discriminator_updates_dict = get_model_updates(layers=discriminator_rnn_model+discriminator_output_model,
                                                   cost=discriminator_updates_cost,
                                                   optimizer=discriminator_optimizer,
                                                   use_grad_clip=grad_clipping)

    # set discriminator update inputs
    discriminator_updates_inputs  = [input_data,
                                     init_input_data,
                                     init_hidden_data,
                                     init_cell_data]

    # set discriminator update outputs
    discriminator_updates_outputs = [input_cost_data, sample_cost_data, discriminator_cost]

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

    # init hidden data (num_samples *input_dims)
    init_hidden_data = tensor.matrix(name='init_hidden_data',
                                     dtype=floatX)

    # init cell data (num_samples *input_dims)
    init_cell_data = tensor.matrix(name='init_hidden_data',
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
    output_data = generator_rnn_model[0].forward(generator_input_data_list, is_training=True)[0]


    # input data
    generation_function_inputs  = [init_input_data,
                                   init_hidden_data,
                                   init_cell_data,
                                   sampling_length]
    generation_function_outputs = [output_data, ]

    generation_function = theano.function(inputs=generation_function_inputs,
                                          outputs=generation_function_outputs,
                                          on_unused_input='ignore')
    return generation_function

def train_model(feature_size,
                hidden_size,
                generator_rnn_model,
                generator_optimizer,
                discriminator_rnn_model,
                discriminator_output_model,
                discriminator_optimizer,
                num_epochs,
                model_name):

    # generator updater
    generator_updater = set_generator_update_function(generator_rnn_model=generator_rnn_model,
                                                      discriminator_rnn_model=discriminator_rnn_model,
                                                      discriminator_output_model=discriminator_output_model,
                                                      generator_optimizer=generator_optimizer,
                                                      grad_clipping=1.0)

    # discriminator updater
    discriminator_updater = set_discriminator_update_function(generator_rnn_model=generator_rnn_model,
                                                              discriminator_rnn_model=discriminator_rnn_model,
                                                              discriminator_output_model=discriminator_output_model,
                                                              discriminator_optimizer=discriminator_optimizer,
                                                              grad_clipping=1.0)

    # sample generator
    sample_generator = set_sample_generation_function(generator_rnn_model=generator_rnn_model)

    # for each epoch
    generator_cost_list = []
    discriminator_cost_list = []
    for e in xrange(num_epochs):
        window_size = 100

        # set data stream with proper length (window size)
        data_stream = set_datastream(feature_size=feature_size,
                                     window_size=window_size)
        # get data iterator
        data_iterator = data_stream.get_epoch_iterator()

        # for each batch
        for batch_idx, batch_data in enumerate(data_iterator):
            # source data
            source_data = batch_data[0]
            source_data = source_data.reshape(window_size, feature_size)
            source_data = numpy.expand_dims(source_data, axis=0)
            source_data = numpy.swapaxes(source_data, axis1=0, axis2=1)

            # normalize
            source_data = (source_data/(2.**15)).astype(floatX)

            # set generator initial values
            init_input_data  = np_rng.uniform(low=-1.0, high=1.0, size=(source_data[1], feature_size)).astype(floatX)
            init_hidden_data = np_rng.uniform(low=-1.0, high=1.0, size=(source_data[1], hidden_size)).astype(floatX)
            init_cell_data   = np_rng.normal(size=(source_data[1], hidden_size)).astype(floatX)

            # update generator
            generator_updater_input = [init_input_data,
                                       init_hidden_data,
                                       init_cell_data,
                                       window_size]

            generator_updater_output = generator_updater(*generator_updater_input)
            generator_cost = generator_updater_output[1].mean()

            # update discriminator
            discriminator_updater_input = [source_data,
                                           init_input_data,
                                           init_hidden_data,
                                           init_cell_data]

            discriminator_updater_output = discriminator_updater(*discriminator_updater_input)
            input_cost_data    = discriminator_updater_output[0]
            sample_cost_data   = discriminator_updater_output[1]
            discriminator_cost = discriminator_updater_output[2].mean()



            if (batch_idx+1)%100==0:
                print 'epoch {}, batch_idx {} => generator     cost {}'.format(e, batch_idx, generator_cost)
                print '                       => discriminator cost {}'.format(e, batch_idx, discriminator_cost)
                generator_cost_list.append(generator_cost)
                discriminator_cost_list.append(discriminator_cost)
                plot_learning_curve(cost_values=[generator_cost_list, discriminator_cost_list],
                                    cost_names=['Generator Cost', 'Discriminator Cost'],
                                    save_as=model_name+'_model_cost.png',
                                    legend_pos='upper left')

                plot_learning_curve(cost_values=[input_cost_data.mean(axis=(1, 2)), sample_cost_data.mean(axis=(1, 2))],
                                    cost_names=['Data Distribution', 'Model Distribution'],
                                    save_as=model_name+'_seq_cost{}.png'.format(batch_idx),
                                    legend_pos='upper left')


            if (batch_idx+1)%1000==0:
                num_samples = 10
                num_sec     = 10
                sampling_length = num_sec*sampling_rate/feature_size
                # set generator initial values
                init_input_data  = np_rng.uniform(low=-1.0, high=1.0, size=(num_samples, feature_size)).astype(floatX)
                init_hidden_data = np_rng.uniform(low=-1.0, high=1.0, size=(num_samples, hidden_size)).astype(floatX)
                init_cell_data   = np_rng.normal(size=(num_samples, hidden_size)).astype(floatX)

                generator_input = [init_input_data,
                                   init_hidden_data,
                                   init_cell_data,
                                   sampling_length]

                sample_data = sample_generator(*generator_input)[0]

                sample_data = numpy.swapaxes(sample_data, axis1=0, axis2=1)
                sample_data = sample_data.reshape((num_samples, -1))
                num_samples = num_samples*(2.**15)
                num_samples = num_samples.astype(numpy.int16)
                save_wavfile(num_samples, model_name+'_sample')

if __name__=="__main__":
    feature_size  = 160
    hidden_size   = 1000
    learning_rate = 1e-3
    num_layers    = 2

    model_name = 'lstm_gan' \
                 + '_FEATURE{}'.format(int(feature_size)) \
                 + '_HIDDEN{}'.format(int(hidden_size)) \
                 + '_LR{}'.format(int(-numpy.log10(learning_rate))) \

    # generator model
    generator_rnn_model = set_generator_recurrent_model(input_size=feature_size,
                                                        hidden_size=hidden_size,
                                                        max_length=1000)

    # discriminator model
    discriminator_rnn_model = set_discriminator_recurrent_model(input_size=feature_size,
                                                                hidden_size=hidden_size,
                                                                num_layers=2)
    discriminator_output_model = set_discriminator_output_model(input_size=hidden_size)

    # set optimizer
    generator_optimizer     = RmsProp(learning_rate=learning_rate).update_params
    discriminator_optimizer = RmsProp(learning_rate=learning_rate).update_params


    train_model(feature_size=feature_size,
                hidden_size=hidden_size,
                generator_rnn_model=generator_rnn_model,
                generator_optimizer=generator_optimizer,
                discriminator_rnn_model=discriminator_rnn_model,
                discriminator_output_model=discriminator_output_model,
                discriminator_optimizer=discriminator_optimizer,
                num_epochs=10,
                model_name=model_name)
