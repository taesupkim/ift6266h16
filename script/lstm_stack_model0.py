__author__ = 'KimTS'
import theano
import numpy
from theano import tensor
from data.window import Window
from util.utils import save_wavfile
from layer.activations import Tanh, Logistic, Relu, Softplus
from layer.layers import LinearLayer, LstmStackLayer
from layer.layer_utils import get_tensor_output, get_model_updates, get_lstm_outputs, get_model_gradients
from optimizer.rmsprop import RmsProp
from optimizer.adagrad import AdaGrad
from numpy.random import RandomState
from theano.sandbox.rng_mrg import MRG_RandomStreams
from utils.display import plot_learning_curve
from time import time
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
    layers.append(LstmStackLayer(input_dim=input_size,
                                 hidden_dim=hidden_size,
                                 num_layers=num_layers,
                                 name='generator_rnn_model'))
    return layers

def set_generator_mean_model(hidden_size,
                             output_size,
                             num_layers):
    layers = []
    layers.append(LinearLayer(input_dim=hidden_size*num_layers,
                              output_dim=hidden_size*num_layers/2,
                              name='generator_mean_linear_layer0'))
    layers.append(Relu(name='generator_mean_relu_layer0'))

    layers.append(LinearLayer(input_dim=hidden_size*num_layers/2,
                              output_dim=output_size,
                              name='generator_mean_linear_output'))
    layers.append(Tanh(name='generator_mean_tanh_output'))
    return layers

def set_generator_std_model(hidden_size,
                            output_size,
                            num_layers):
    layers = []

    layers.append(LinearLayer(input_dim=hidden_size*num_layers,
                              output_dim=hidden_size*num_layers/2,
                              name='generator_var_linear_layer0'))
    layers.append(Relu(name='generator_var_relu_layer0'))

    layers.append(LinearLayer(input_dim=hidden_size*num_layers/2,
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
    hidden_data = hidden_data.dimshuffle(0, 2, 1, 3).flatten(3)

    # get generator output data
    output_mean_data = get_tensor_output(input=hidden_data,
                                         layers=generator_mean_model,
                                         is_training=True)
    output_std_data = get_tensor_output(input=hidden_data,
                                        layers=generator_std_model,
                                        is_training=True)

    # get generator cost (time_length x num_samples x hidden_size)
    generator_cost  = -0.5*tensor.inv(2.0*tensor.sqr(output_std_data))*tensor.sqr(output_mean_data-target_data)
    generator_cost += -0.5*tensor.log(2.0*tensor.sqr(output_std_data)*numpy.pi)

    # set generator update
    generator_updates_cost = tensor.sum(generator_cost, axis=2).mean()
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
    generator_updates_outputs = [generator_cost,
                                 gradient_norm]

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
    hidden_data = hidden_data.dimshuffle(0, 2, 1, 3).flatten(3)

    # get generator output data
    output_mean_data = get_tensor_output(input=hidden_data,
                                         layers=generator_mean_model,
                                         is_training=True)
    output_std_data = get_tensor_output(input=hidden_data,
                                        layers=generator_std_model,
                                        is_training=True)

    # get generator cost (time_length x num_samples x hidden_size)
    generator_cost  = -0.5*tensor.inv(2.0*tensor.sqr(output_std_data))*tensor.sqr(output_mean_data-target_data)
    generator_cost += -0.5*tensor.log(2.0*tensor.sqr(output_std_data)*numpy.pi)

    # set generator evaluate inputs
    generator_evaluate_inputs  = [source_data,
                                  target_data,]

    # set generator evaluate outputs
    generator_evaluate_outputs = [generator_cost,]

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

    # prev hidden data (num_layers * num_samples * input_dims))
    prev_hidden_data = tensor.tensor3(name='prev_hidden_data',
                                      dtype=floatX)


    # get current hidden data
    generator_input_data_list = [cur_input_data,
                                 prev_hidden_data]
    cur_hidden_data = generator_rnn_model[0].forward(generator_input_data_list, is_training=False)[0]

    # get generator output data
    output_mean_data = get_tensor_output(input=cur_hidden_data.dimshuffle(1, 0, 2).flatten(2),
                                         layers=generator_mean_model,
                                         is_training=False)
    output_data = output_mean_data

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
    t = time()
    generator_updater = set_generator_update_function(generator_rnn_model=generator_rnn_model,
                                                      generator_mean_model=generator_mean_model,
                                                      generator_std_model=generator_std_model,
                                                      generator_optimizer=generator_optimizer,
                                                      grad_clipping=0.0)
    print '{}.sec'.format(time()-t)

    # generator evaluator
    print 'DEBUGGING GENERATOR EVALUATION FUNCTION '
    t = time()
    generator_evaluator = set_generator_evaluation_function(generator_rnn_model=generator_rnn_model,
                                                            generator_mean_model=generator_mean_model,
                                                            generator_std_model=generator_std_model)
    print '{}.sec'.format(time()-t)

    # generator sampler
    print 'DEBUGGING GENERATOR SAMPLING FUNCTION '
    t = time()
    generator_sampler = set_generator_sampling_function(generator_rnn_model=generator_rnn_model,
                                                        generator_mean_model=generator_mean_model,
                                                        generator_std_model=generator_std_model)
    print '{}.sec'.format(time()-t)

    print 'START TRAINING'
    # for each epoch
    generator_train_cost_list = []
    generator_valid_cost_list = []

    generator_grad_norm_mean = 0.0

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
            if train_batch_size==0:
                train_source_data = []
                train_target_data = []

            # source data
            single_data = batch_data[0]
            single_data = single_data.reshape(single_data.shape[0]/feature_size, feature_size)
            train_source_data.append(single_data)

            # target data
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
            train_source_data = (train_source_data/(2.**15)).astype(floatX)
            train_target_data = (train_target_data/(2.**15)).astype(floatX)

            # update generator
            generator_updater_input = [train_source_data,
                                       train_target_data]

            generator_updater_output = generator_updater(*generator_updater_input)
            generator_train_cost = generator_updater_output[0].mean()
            generator_grad_norm  = generator_updater_output[1]

            generator_grad_norm_mean += generator_grad_norm
            train_batch_count += 1


            sampling_seed_data = []
            if train_batch_count%10==0:
                # set valid data stream with proper length (window size)
                valid_window_size = 100
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
                    valid_source_data = (valid_source_data/(2.**15)).astype(floatX)
                    valid_target_data = (valid_target_data/(2.**15)).astype(floatX)

                    generator_evaluator_input = [valid_source_data,
                                                 valid_target_data]

                    generator_evaluator_output = generator_evaluator(*generator_evaluator_input)
                    generator_valid_cost  = generator_evaluator_output[0].mean()

                    valid_cost_mean += generator_valid_cost
                    valid_batch_count += 1

                    if valid_batch_count>100:
                        sampling_seed_data = valid_source_data
                        break

                valid_cost_mean = valid_cost_mean/valid_batch_count

                print '=============sample length {}============================='.format(window_size)
                print 'epoch {}, batch_cnt {} => generator train cost {}'.format(e, train_batch_count, generator_train_cost)
                print 'epoch {}, batch_cnt {} => generator valid cost {}'.format(e, train_batch_count, valid_cost_mean)
                print 'epoch {}, batch_cnt {} => generator grad norm  {}'.format(e, train_batch_count, generator_grad_norm_mean/train_batch_count)

                generator_train_cost_list.append(generator_train_cost)
                generator_valid_cost_list.append(valid_cost_mean)

                plot_learning_curve(cost_values=[generator_train_cost_list, generator_valid_cost_list],
                                    cost_names=['Train Cost', 'Valid Cost'],
                                    save_as=model_name+'_model_cost.png',
                                    legend_pos='upper left')

            if train_batch_count%100==0:
                num_samples = 10
                num_sec     = 10
                sampling_length = num_sec*sampling_rate/feature_size

                curr_input_data  = sampling_seed_data[0][:num_samples]
                prev_hidden_data = np_rng.normal(size=(num_layers, num_samples, hidden_size)).astype(floatX)
                prev_hidden_data = numpy.tanh(prev_hidden_data)
                output_data      = numpy.zeros(shape=(sampling_length, num_samples, feature_size))
                for s in xrange(sampling_length):


                    generator_input = [curr_input_data,
                                       prev_hidden_data,]

                    [curr_input_data, prev_hidden_data] = generator_sampler(*generator_input)

                    output_data[s] = curr_input_data
                sample_data = numpy.swapaxes(output_data, axis1=0, axis2=1)
                sample_data = sample_data.reshape((num_samples, -1))
                sample_data = sample_data*(2.**15)
                sample_data = sample_data.astype(numpy.int16)
                save_wavfile(sample_data, model_name+'_sample')

if __name__=="__main__":
    feature_size  = 160
    hidden_size   = 160
    learning_rate = 1e-2
    num_layers    = 4

    model_name = 'lstm_stack_model' \
                 + '_FEATURE{}'.format(int(feature_size)) \
                 + '_HIDDEN{}'.format(int(hidden_size)) \
                 + '_LAYERS{}'.format(int(num_layers)) \
                 + '_LR{}'.format(int(-numpy.log10(learning_rate))) \

    # generator model
    generator_rnn_model = set_generator_recurrent_model(input_size=feature_size,
                                                        hidden_size=hidden_size,
                                                        num_layers=num_layers)
    generator_mean_model = set_generator_mean_model(hidden_size=hidden_size,
                                                    output_size=feature_size,
                                                    num_layers=num_layers)
    generator_std_model  = set_generator_std_model(hidden_size=hidden_size,
                                                   output_size=feature_size,
                                                   num_layers=num_layers)

    # set optimizer
    generator_optimizer = RmsProp(learning_rate=learning_rate).update_params

    train_model(feature_size=feature_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                generator_rnn_model=generator_rnn_model,
                generator_mean_model=generator_mean_model,
                generator_std_model=generator_std_model,
                generator_optimizer=generator_optimizer,
                num_epochs=10,
                model_name=model_name)
