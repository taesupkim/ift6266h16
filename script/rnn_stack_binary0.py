__author__ = 'KimTS'
import theano
import numpy
from theano import tensor
from data.window import Window
from util.utils import save_wavfile
from layer.activations import Tanh, Logistic, Relu
from layer.layers import LinearLayer, RecurrentStackLayer
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
    layers.append(RecurrentStackLayer(input_dim=input_size,
                                      hidden_dim=hidden_size,
                                      num_layers=num_layers,
                                      name='generator_rnn_model'))
    return layers

def set_generator_output_model(hidden_size,
                               output_size,
                               num_outputs,
                               num_layers):
    models = []
    for i in xrange(num_outputs):
        generator_layers = []
        for l in xrange(num_layers):
            input_dim = hidden_size
            output_dim = output_size if l is (num_layers-1) else hidden_size
            generator_layers.append(LinearLayer(input_dim=input_dim,
                                                output_dim=output_dim,
                                                name='generator_output_linear_model{}_layer{}'.format(i, l)))

            if l is (num_layers-1):
                generator_layers.append(Logistic(name='generator_output_clip_model{}_layer{}'.format(i, l)))
            else:
                generator_layers.append(Tanh(name='generator_output_clip_model{}_layer{}'.format(i, l)))

        models.append(generator_layers)

    return models

def set_generator_update_function(generator_rnn_model,
                                  generator_output_models,
                                  generator_optimizer,
                                  grad_clipping):
    # set source data (time_length * num_samples * input_dims)
    source_data  = tensor.tensor3(name='source_data',
                                  dtype=floatX)

    # set target data (time_length * num_samples * input_dims)
    target_data  = tensor.bmatrix(name='target_data')

    # set generator input data list
    generator_input_data_list = [source_data,]

    # get generator output data
    hidden_data = generator_rnn_model[0].forward(generator_input_data_list, is_training=True)[0]

    # for each rnn layer
    output_data_list = []
    for l, output_model in enumerate(generator_output_models):
        output_data = get_tensor_output(input=hidden_data[l],
                                        layers=output_model,
                                        is_training=True)
        output_data_list.append(output_data)

    output_data = tensor.concatenate(output_data_list[::-1], axis=1)

    output_sign_data  = output_data[:,0]
    output_sign_data  = 2.0*output_sign_data-tensor.ones_like(output_sign_data)
    output_value_data = output_data[:,1:]
    output_value_data = output_value_data*tensor.pow(2.0, tensor.arange(output_value_data.shape[1]))
    output_value_data = output_sign_data*output_value_data.sum(axis=1)

    target_sign_data  = target_data[:,0]
    target_sign_data  = 2.0*target_sign_data-tensor.ones_like(target_sign_data)
    target_value_data = target_data[:,1:]
    target_value_data = target_value_data*tensor.pow(2.0, tensor.arange(target_value_data.shape[1]))
    target_value_data = target_sign_data*target_value_data.sum(axis=1)

    mse_cost = tensor.sqr(output_value_data, target_value_data)
    bce_cost = tensor.nnet.binary_crossentropy(output_data, target_data).sum(axis=1)

    # set generator update
    generator_updates_cost = generator_cost.mean()
    generator_updates_dict = get_model_updates(layers=generator_rnn_model+,
                                               cost=generator_updates_cost,
                                               optimizer=generator_optimizer,
                                               use_grad_clip=grad_clipping)

    # gradient_dict  = get_model_gradients(generator_rnn_model, generator_updates_cost)
    # gradient_norm  = 0.
    # for grad in gradient_dict:
    #     gradient_norm += tensor.sum(grad**2)
    #     gradient_norm  = tensor.sqrt(gradient_norm)

    # set generator update inputs
    generator_updates_inputs  = [init_input_data,
                                 init_hidden_data,
                                 init_cell_data,
                                 sampling_length]

    # set generator update outputs
    generator_updates_outputs = [sample_cost_data, generator_cost, ]#gradient_norm]

    # set generator update function
    generator_updates_function = theano.function(inputs=generator_updates_inputs,
                                                 outputs=generator_updates_outputs,
                                                 updates=merge_dicts([generator_updates_dict, update_data]),
                                                 on_unused_input='ignore')

    return generator_updates_function

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
    # output_data = generator_rnn_model[0].forward(generator_input_data_list, is_training=True)[0]
    output_data_set = generator_rnn_model[0].forward(generator_input_data_list, is_training=True)
    sample_data = output_data_set[0]
    update_data = output_data_set[-1]

    # input data
    generation_function_inputs  = [init_input_data,
                                   init_hidden_data,
                                   init_cell_data,
                                   sampling_length]
    generation_function_outputs = [sample_data, ]

    generation_function = theano.function(inputs=generation_function_inputs,
                                          outputs=generation_function_outputs,
                                          updates=update_data,
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
                                                      grad_clipping=3.6)

    # discriminator updater
    print 'DEBUGGING DISCRIMINATOR UPDATE FUNCTION '
    discriminator_updater = set_discriminator_update_function(generator_rnn_model=generator_rnn_model,
                                                              discriminator_rnn_model=discriminator_rnn_model,
                                                              discriminator_output_model=discriminator_output_model,
                                                              discriminator_optimizer=discriminator_optimizer,
                                                              grad_clipping=1.8)

    # sample generator
    print 'DEBUGGING SAMPLE GENERATOR FUNCTION '
    sample_generator = set_sample_generation_function(generator_rnn_model=generator_rnn_model)



    print 'START TRAINING'
    # for each epoch
    generator_cost_list = []
    discriminator_cost_list = []

    generator_grad_norm_mean     = 0.0
    discriminator_grad_norm_mean = 0.0

    init_window_size = 20
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
            init_input_data  = np_rng.normal(size=(source_data.shape[1], feature_size)).astype(floatX)
            init_input_data  = numpy.clip(init_input_data, -1., 1.)
            # init_hidden_data = np_rng.normal(size=(num_layers, source_data.shape[1], hidden_size)).astype(floatX)
            # init_hidden_data = numpy.clip(init_hidden_data, -1., 1.)
            # init_cell_data   = np_rng.normal(size=(num_layers, source_data.shape[1], hidden_size)).astype(floatX)
            init_hidden_data = numpy.zeros(shape=(num_layers, source_data.shape[1], hidden_size), dtype=floatX)
            init_cell_data   = numpy.zeros(shape=(num_layers, source_data.shape[1], hidden_size), dtype=floatX)

            # update generator
            generator_updater_input = [init_input_data,
                                       init_hidden_data,
                                       init_cell_data,
                                       window_size]

            generator_updater_output = generator_updater(*generator_updater_input)
            generator_cost = generator_updater_output[1].mean()
            # generator_grad_norm = generator_updater_output[-1]

            # update discriminator
            init_input_data  = np_rng.normal(size=(source_data.shape[1], feature_size)).astype(floatX)
            init_input_data  = numpy.clip(init_input_data, -1., 1.)
            # init_hidden_data = np_rng.normal(size=(num_layers, source_data.shape[1], hidden_size)).astype(floatX)
            # init_hidden_data = numpy.clip(init_hidden_data, -1., 1.)
            # init_cell_data   = np_rng.normal(size=(num_layers, source_data.shape[1], hidden_size)).astype(floatX)
            init_hidden_data = numpy.zeros(shape=(num_layers, source_data.shape[1], hidden_size), dtype=floatX)
            init_cell_data   = numpy.zeros(shape=(num_layers, source_data.shape[1], hidden_size), dtype=floatX)

            discriminator_updater_input = [source_data,
                                           init_input_data,
                                           init_hidden_data,
                                           init_cell_data]

            discriminator_updater_output = discriminator_updater(*discriminator_updater_input)
            input_cost_data    = discriminator_updater_output[0]
            sample_cost_data   = discriminator_updater_output[1]
            discriminator_cost = discriminator_updater_output[2].mean()
            # discriminator_grad_norm = discriminator_updater_output[-1]

            # generator_grad_norm_mean     += generator_grad_norm
            # discriminator_grad_norm_mean += discriminator_grad_norm

            batch_count += 1

            if batch_count%500==0:
                print '=============sample length {}============================='.format(window_size)
                print 'epoch {}, batch_cnt {} => generator     cost {}'.format(e, batch_count, generator_cost)
                print 'epoch {}, batch_cnt {} => discriminator cost {}'.format(e, batch_count, discriminator_cost)
                print 'epoch {}, batch_cnt {} => input data    cost {}'.format(e, batch_count, input_cost_data.mean())
                print 'epoch {}, batch_cnt {} => sample data   cost {}'.format(e, batch_count, sample_cost_data.mean())
                # print 'epoch {}, batch_cnt {} => generator     grad norm{}'.format(e, batch_count, generator_grad_norm_mean/batch_count)
                # print 'epoch {}, batch_cnt {} => discriminator grad norm{}'.format(e, batch_count, discriminator_grad_norm_mean/batch_count)

                generator_cost_list.append(generator_cost)
                discriminator_cost_list.append(discriminator_cost)
                plot_learning_curve(cost_values=[generator_cost_list, discriminator_cost_list],
                                    cost_names=['Generator Cost', 'Discriminator Cost'],
                                    save_as=model_name+'_model_cost.png',
                                    legend_pos='upper left')

                plot_learning_curve(cost_values=[input_cost_data.mean(axis=(1, 2)), sample_cost_data.mean(axis=(1, 2))],
                                    cost_names=['Data Distribution', 'Model Distribution'],
                                    save_as=model_name+'_seq_cost{}.png'.format(batch_count),
                                    legend_pos='upper left')


            if batch_count%5000==0:
                num_samples = 10
                num_sec     = 10
                sampling_length = num_sec*sampling_rate/feature_size
                # set generator initial values
                init_input_data  = np_rng.normal(size=(num_samples, feature_size)).astype(floatX)
                init_input_data  = numpy.clip(init_input_data, -1., 1.)
                # init_hidden_data = np_rng.normal(size=(num_layers, num_samples, hidden_size)).astype(floatX)
                # init_hidden_data = numpy.clip(init_hidden_data, -1., 1.)
                # init_cell_data   = np_rng.normal(size=(num_layers, num_samples, hidden_size)).astype(floatX)
                init_hidden_data = numpy.zeros(shape=(num_layers, num_samples, hidden_size), dtype=floatX)
                init_cell_data   = numpy.zeros(shape=(num_layers, num_samples, hidden_size), dtype=floatX)

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
    learning_rate = 1e-3
    num_layers    = 2

    model_name = 'lstm_gan' \
                 + '_FEATURE{}'.format(int(feature_size)) \
                 + '_HIDDEN{}'.format(int(hidden_size)) \
                 + '_LAYERS{}'.format(int(num_layers)) \
                 + '_LR{}'.format(int(-numpy.log10(learning_rate))) \

    # generator model
    generator_rnn_model = set_generator_recurrent_model(input_size=feature_size,
                                                        hidden_size=hidden_size,
                                                        num_layers=num_layers)

    # discriminator model
    discriminator_rnn_model = set_discriminator_recurrent_model(input_size=feature_size,
                                                                hidden_size=hidden_size,
                                                                num_layers=num_layers+1)
    discriminator_output_model = set_discriminator_output_model(input_size=hidden_size,
                                                                num_layers=num_layers)

    # set optimizer
    generator_optimizer     = RmsProp(learning_rate=learning_rate).update_params
    discriminator_optimizer = RmsProp(learning_rate=learning_rate).update_params


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
