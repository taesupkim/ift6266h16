__author__ = 'KimTS'
import theano
import numpy
from theano import tensor
from data.window import Window
from util.utils import save_wavfile
from layer.activations import Tanh, Logistic, Relu, Softplus
from layer.layers import LinearLayer, GateFeedRecurrentLayer
from layer.layer_utils import get_tensor_output, get_model_updates, get_lstm_outputs, get_model_gradients
from optimizer.rmsprop import RmsProp
from optimizer.adagrad import AdaGrad
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

def set_generator_recurrent_model(input_size,
                                  hidden_size,
                                  num_layers):
    layers = []
    layers.append(GateFeedRecurrentLayer(input_dim=input_size,
                                         hidden_dim=hidden_size,
                                         num_layers=num_layers,
                                         name='generator_rnn_model'))
    return layers

def set_generator_mean_model(hidden_size,
                             output_size,
                             num_layers):
    layers = []

    for l in xrange(num_layers-1):
        layers.append(LinearLayer(input_dim=hidden_size,
                                  output_dim=hidden_size,
                                  name='generator_mean_linear_layer{}'.format(l)))
        layers.append(Relu(name='generator_mean_relu_layer{}'.format(l)))

    layers.append(LinearLayer(input_dim=hidden_size,
                              output_dim=output_size,
                              name='generator_mean_linear_output'))
    return layers

def set_generator_std_model(hidden_size,
                            output_size,
                            num_layers):
    layers = []

    for l in xrange(num_layers-1):
        layers.append(LinearLayer(input_dim=hidden_size,
                                  output_dim=hidden_size,
                                  name='generator_var_linear_layer{}'.format(l)))
        layers.append(Relu(name='generator_var_relu_layer{}'.format(l)))

    layers.append(LinearLayer(input_dim=hidden_size,
                              output_dim=output_size,
                              name='generator_var_linear_output'))
    layers.append(Softplus(name='generator_var_relu_output'))
    return layers


def set_generator_update_function(generator_rnn_model,
                                  generator_mean_model,
                                  generator_std_model,
                                  generator_optimizer,
                                  grad_clipping):

    # input data (time length * num_samples * input_dims)
    source_data = tensor.tensor3(name='source_data',
                                 dtype=floatX)

    target_data = tensor.tensor3(name='target_data',
                                 dtype=floatX)

    # set generator input data list
    generator_input_data_list = [source_data,]

    # get generator hidden data
    hidden_data = generator_rnn_model[0].forward(generator_input_data_list, is_training=True)[0]

    # get generator output data
    output_mean_data = get_tensor_output(input=hidden_data,
                                         layers=generator_mean_model,
                                         is_training=True)
    output_std_data = get_tensor_output(input=hidden_data,
                                        layers=generator_std_model,
                                        is_training=True)

    generator_cost  = -0.5*tensor.inv(2.0*tensor.sqr(output_std_data))*tensor.sqr(output_mean_data-target_data)
    generator_cost += -0.5*tensor.log(2.0*tensor.sqr(output_std_data)*numpy.pi)
    generator_cost  = tensor.sum(generator_cost, axis=1)

    # set generator update
    generator_updates_cost = generator_cost.mean()
    generator_updates_dict = get_model_updates(layers=generator_rnn_model+generator_mean_model+generator_std_model,
                                               cost=generator_updates_cost,
                                               optimizer=generator_optimizer,
                                               use_grad_clip=grad_clipping)

    gradient_dict  = get_model_gradients(generator_rnn_model+generator_mean_model+generator_std_model, generator_updates_cost)
    gradient_norm  = 0.
    for grad in gradient_dict:
        gradient_norm += tensor.sum(grad**2)
        gradient_norm  = tensor.sqrt(gradient_norm)

    # set generator update inputs
    generator_updates_inputs  = [source_data,
                                 target_data,]

    # set generator update outputs
    generator_updates_outputs = [generator_cost, gradient_norm]

    # set generator update function
    generator_updates_function = theano.function(inputs=generator_updates_inputs,
                                                 outputs=generator_updates_outputs,
                                                 updates=generator_updates_dict,
                                                 on_unused_input='ignore')

    return generator_updates_function

def set_generator_evaluation_function(generator_rnn_model,
                                      generator_mean_model,
                                      generator_std_model):

    # input data (time length * num_samples * input_dims)
    source_data = tensor.tensor3(name='source_data',
                                 dtype=floatX)

    target_data = tensor.tensor3(name='target_data',
                                 dtype=floatX)

    # set generator input data list
    generator_input_data_list = [source_data,]

    # get generator hidden data
    hidden_data = generator_rnn_model[0].forward(generator_input_data_list, is_training=True)[0]

    # get generator output data
    output_mean_data = get_tensor_output(input=hidden_data,
                                         layers=generator_mean_model,
                                         is_training=True)
    output_std_data = get_tensor_output(input=hidden_data,
                                        layers=generator_std_model,
                                        is_training=True)

    generator_cost  = -0.5*tensor.inv(2.0*tensor.sqr(output_std_data))*tensor.sqr(output_mean_data-target_data)
    generator_cost += -0.5*tensor.log(2.0*tensor.sqr(output_std_data)*numpy.pi)
    generator_cost  = tensor.sum(generator_cost, axis=1)

    # set generator evaluate inputs
    generator_evaluate_inputs  = [source_data,
                                  target_data,]

    # set generator evaluate outputs
    generator_evaluate_outputs = [generator_cost, ]

    # set generator evaluate function
    generator_evaluate_function = theano.function(inputs=generator_evaluate_inputs,
                                                  outputs=generator_evaluate_outputs,
                                                  on_unused_input='ignore')

    return generator_evaluate_function

def set_generator_sampling_function(generator_rnn_model,
                                    generator_mean_model,
                                    generator_std_model):

    # input data (num_samples *input_dims)
    cur_input_data = tensor.matrix(name='cur_input_data',
                                   dtype=floatX)

    # prev hidden data (num_samples * (num_layers * input_dims))
    prev_hidden_data = tensor.matrix(name='prev_hidden_data',
                                      dtype=floatX)

    generator_input_data_list = [cur_input_data, prev_hidden_data]
    cur_hidden_data = generator_rnn_model[0].forward(generator_input_data_list, is_training=False)[0]


    # get generator output data
    output_mean_data = get_tensor_output(input=cur_hidden_data,
                                         layers=generator_mean_model,
                                         is_training=True)
    output_std_data = get_tensor_output(input=cur_hidden_data,
                                        layers=generator_std_model,
                                        is_training=True)

    output_data = output_mean_data + output_std_data*theano_rng.normal(size=output_std_data.shape, dtype=floatX)

    # input data
    generation_sampling_inputs  = [cur_input_data,
                                   prev_hidden_data]
    generation_sampling_outputs = [output_data,
                                   cur_hidden_data]

    generation_sampling_function = theano.function(inputs=generation_sampling_inputs,
                                                   outputs=generation_sampling_outputs,
                                                   on_unused_input='ignore')
    return generation_sampling_function

def train_model(feature_size,
                hidden_size,
                num_layers,
                generator_rnn_model,
                generator_mean_model,
                generator_std_model,
                generator_optimizer,
                num_epochs,
                model_name):

    # generator updater
    print 'DEBUGGING GENERATOR UPDATE FUNCTION '
    generator_updater = set_generator_update_function(generator_rnn_model=generator_rnn_model,
                                                      generator_mean_model=generator_mean_model,
                                                      generator_std_model=generator_std_model,
                                                      generator_optimizer=generator_optimizer,
                                                      grad_clipping=0.0)

    # generator evaluator
    print 'DEBUGGING GENERATOR EVALUATION FUNCTION '
    generator_evaluator = set_generator_evaluation_function(generator_rnn_model=generator_rnn_model,
                                                            generator_mean_model=generator_mean_model,
                                                            generator_std_model=generator_std_model)

    # generator sampler
    print 'DEBUGGING GENERATOR SAMPLING FUNCTION '
    generator_sampler = set_generator_sampling_function(generator_rnn_model=generator_rnn_model,
                                                        generator_mean_model=generator_mean_model,
                                                        generator_std_model=generator_std_model)

    print 'START TRAINING'
    # for each epoch
    generator_cost_list = []

    generator_grad_norm_mean = 0.0

    init_window_size = 20
    for e in xrange(num_epochs):
        window_size = init_window_size + 5*e

        # set train data stream with proper length (window size)
        train_data_stream = set_train_datastream(feature_size=feature_size,
                                                 window_size=window_size)
        # get train data iterator
        train_data_iterator = train_data_stream.get_epoch_iterator()

        # for each batch
        batch_count = 0
        batch_size = 0
        source_data = []
        target_data = []
        for batch_idx, batch_data in enumerate(train_data_iterator):
            if batch_size==0:
                source_data = []
                target_data = []

            # source data
            single_data = batch_data[0]
            single_data = single_data.reshape(window_size, feature_size)
            source_data.append(single_data)

            # target data
            single_data = batch_data[1]
            single_data = single_data.reshape(window_size, feature_size)
            target_data.append(single_data)

            batch_size += 1

            if batch_size<128:
                continue
            else:
                # source data
                source_data = numpy.asarray(source_data, dtype=floatX)
                source_data = numpy.swapaxes(source_data, axis1=0, axis2=1)
                # target data
                target_data = numpy.asarray(target_data, dtype=floatX)
                target_data = numpy.swapaxes(target_data, axis1=0, axis2=1)
                batch_size = 0

            # normalize
            source_data = (source_data/(2.**15)).astype(floatX)
            target_data = (target_data/(2.**15)).astype(floatX)

            # update generator
            generator_updater_input = [source_data,
                                       target_data]

            generator_updater_output = generator_updater(*generator_updater_input)
            generator_cost      = generator_updater_output[0].mean()
            generator_grad_norm = generator_updater_output[1]

            generator_grad_norm_mean += generator_grad_norm
            batch_count += 1

            if batch_count%500==0:
                print '=============sample length {}============================='.format(window_size)
                print 'epoch {}, batch_cnt {} => generator cost      {}'.format(e, batch_count, generator_cost)
                print 'epoch {}, batch_cnt {} => generator grad norm {}'.format(e, batch_count, generator_grad_norm_mean/batch_count)

                generator_cost_list.append(generator_cost)
                plot_learning_curve(cost_values=[generator_cost_list,],
                                    cost_names=['Generator Cost', ],
                                    save_as=model_name+'_model_cost.png',
                                    legend_pos='upper left')

            # if batch_count%5000==0:
            #     num_samples = 10
            #     num_sec     = 10
            #     sampling_length = num_sec*sampling_rate/feature_size
            #
            #
            #     for s in xrange(sampling_length):
            #
            #     # set generator initial values
            #
            #     # init_hidden_data = np_rng.normal(size=(num_layers, num_samples, hidden_size)).astype(floatX)
            #     # init_hidden_data = numpy.clip(init_hidden_data, -1., 1.)
            #     # init_cell_data   = np_rng.normal(size=(num_layers, num_samples, hidden_size)).astype(floatX)
            #     init_hidden_data = numpy.zeros(shape=(num_layers, num_samples, hidden_size), dtype=floatX)
            #     init_cell_data   = numpy.zeros(shape=(num_layers, num_samples, hidden_size), dtype=floatX)
            #
            #     generator_input = [init_input_data,
            #                        init_hidden_data,
            #                        init_cell_data,
            #                        sampling_length]
            #
            #     sample_data = generator_sampler(*generator_input)[0]
            #
            #     sample_data = numpy.swapaxes(sample_data, axis1=0, axis2=1)
            #     sample_data = sample_data.reshape((num_samples, -1))
            #     sample_data = sample_data*(2.**15)
            #     sample_data = sample_data.astype(numpy.int16)
            #     save_wavfile(sample_data, model_name+'_sample')

if __name__=="__main__":
    feature_size  = 160
    hidden_size   = 240
    learning_rate = 1e-4
    num_layers    = 3

    model_name = 'gf_rnn_normal' \
                 + '_FEATURE{}'.format(int(feature_size)) \
                 + '_HIDDEN{}'.format(int(hidden_size)) \
                 + '_LAYERS{}'.format(int(num_layers)) \
                 + '_LR{}'.format(int(-numpy.log10(learning_rate))) \

    # generator model
    generator_rnn_model = set_generator_recurrent_model(input_size=feature_size,
                                                        hidden_size=hidden_size,
                                                        num_layers=num_layers)
    generator_mean_model = set_generator_mean_model(hidden_size=hidden_size*num_layers,
                                                    output_size=feature_size,
                                                    num_layers=2)
    generator_std_model  = set_generator_std_model(hidden_size=hidden_size*num_layers,
                                                   output_size=feature_size,
                                                   num_layers=2)

    # set optimizer
    generator_optimizer     = RmsProp(learning_rate=learning_rate).update_params

    train_model(feature_size=feature_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                generator_rnn_model=generator_rnn_model,
                generator_mean_model=generator_mean_model,
                generator_std_model=generator_std_model,
                generator_optimizer=generator_optimizer,
                num_epochs=10,
                model_name=model_name)
