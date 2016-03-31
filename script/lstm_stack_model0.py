__author__ = 'KimTS'
import theano
import numpy
from theano import tensor
from data.window import Window
from util.utils import save_wavfile
from layer.activations import Tanh
from layer.layers import LinearLayer, LstmLayer, LstmStackLayer
from layer.layer_utils import get_tensor_output, get_model_updates, get_model_gradients
from optimizer.rmsprop import RmsProp
from numpy.random import RandomState
from theano.sandbox.rng_mrg import MRG_RandomStreams
from utils.display import plot_learning_curve
from fuel.datasets.hdf5 import H5PYDataset
from time import time
theano_rng = MRG_RandomStreams(42)
np_rng = RandomState(42)
sampling_rate = 16000

floatX = theano.config.floatX
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

def set_recurrent_model(input_size, hidden_size, num_layers):
    layers = []
    layers.append(LstmStackLayer(input_dim=input_size,
                                 hidden_dim=hidden_size,
                                 num_layers=num_layers,
                                 name='recurrent_model'))
    return layers

def set_output_model(num_layers, hidden_size, input_size):
    layers = []
    layers.append(LinearLayer(input_dim=num_layers*hidden_size,
                              output_dim=num_layers*hidden_size/2,
                              name='output_layer0'))
    layers.append(Tanh(name='output_squeeze_layer0'))
    layers.append(LinearLayer(input_dim=num_layers*hidden_size/2,
                              output_dim=input_size,
                              name='output_layer0'))
    layers.append(Tanh(name='output_squeeze_layer'))
    return layers

def set_update_function(recurrent_model,
                        output_model,
                        optimizer,
                        grad_clip=0.0):
    # set source data (time_length * num_samples * input_dims)
    source_data = tensor.tensor3(name='source_data',
                                 dtype=floatX)

    # set target data (time_length * num_samples * output_dims)
    target_data = tensor.tensor3(name='target_data',
                                 dtype=floatX)

    # get hidden data
    input_list  = [source_data,]
    hidden_data = recurrent_model[-1].forward(input_list, is_training=True)[0]
    hidden_data = hidden_data.dimshuffle(0, 2, 1, 3).flatten(3)

    # get prediction data
    output_data = get_tensor_output(input=hidden_data,
                                    layers=output_model,
                                    is_training=True)

    # get cost (sum over feature, and time)
    sample_cost = tensor.sqr(output_data-target_data)

    # get model updates
    model_cost         = sample_cost.mean()
    model_updates_dict = get_model_updates(layers=recurrent_model+output_model,
                                           cost=model_cost,
                                           optimizer=optimizer,
                                           use_grad_clip=grad_clip)


    gradient_dict  = get_model_gradients(recurrent_model+output_model, model_cost)
    gradient_norm  = 0.
    for grad in gradient_dict:
        gradient_norm += tensor.sum(grad**2)
        gradient_norm  = tensor.sqrt(gradient_norm)


    update_function_inputs  = [source_data,
                               target_data,]
    update_function_outputs = [output_data,
                               sample_cost,
                               gradient_norm]

    update_function = theano.function(inputs=update_function_inputs,
                                      outputs=update_function_outputs,
                                      updates=model_updates_dict,
                                      on_unused_input='ignore')

    return update_function

def set_evaluation_function(recurrent_model,
                            output_model):
    # set source data (time_length * num_samples * input_dims)
    source_data = tensor.tensor3(name='source_data',
                                 dtype=floatX)

    # set target data (time_length * num_samples * output_dims)
    target_data = tensor.tensor3(name='target_data',
                                 dtype=floatX)

    # get hidden data
    input_list  = [source_data,]
    hidden_data = recurrent_model[-1].forward(input_list, is_training=True)[0]
    hidden_data = hidden_data.dimshuffle(0, 2, 1, 3).flatten(3)

    # get prediction data
    output_data = get_tensor_output(input=hidden_data,
                                    layers=output_model,
                                    is_training=True)

    # get cost (sum over feature, and time)
    sample_cost = tensor.sqr(output_data-target_data)

    evaluation_function_inputs  = [source_data,
                                   target_data,]
    evaluation_function_outputs = [output_data,
                                   sample_cost]

    evaluation_function = theano.function(inputs=evaluation_function_inputs,
                                          outputs=evaluation_function_outputs,
                                          on_unused_input='ignore')

    return evaluation_function

def set_generation_function(recurrent_model,
                            output_model):

    # set input data (num_samples,features)
    input_data = tensor.matrix(name='input_data',
                               dtype=floatX)

    # set hidden/cell data (num_layer, num_samples, features)
    hidden_data = tensor.tensor3(name='hidden_data',
                                 dtype=floatX)
    cell_data = tensor.tensor3(name='cell_data',
                               dtype=floatX)

    # set input list
    input_list = [input_data, hidden_data, cell_data]
    # get output data
    [new_hidden_data, new_cell_data] = recurrent_model[-1].forward(input_list, is_training=False)

    # get prediction data
    new_input_data = get_tensor_output(input=new_hidden_data.dimshuffle(1, 0, 2).flatten(2),
                                       layers=output_model,
                                       is_training=False)

    generation_function_inputs  = [input_data,
                                   hidden_data,
                                   cell_data]
    generation_function_outputs = [new_input_data,
                                   new_hidden_data,
                                   new_cell_data]

    generation_function = theano.function(inputs=generation_function_inputs,
                                          outputs=generation_function_outputs,
                                          on_unused_input='ignore')
    return generation_function

def train_model(feature_size,
                hidden_size,
                num_layers,
                init_window_size,
                recurrent_model,
                output_model,
                model_optimizer,
                num_epochs,
                model_name):

    print 'COMPILING GENERATOR UPDATE FUNCTION '
    t=time()
    update_function = set_update_function(recurrent_model=recurrent_model,
                                          output_model=output_model,
                                          optimizer=model_optimizer,
                                          grad_clip=0.0)
    print '%.2f SEC '%(time()-t)

    print 'COMPILING GENERATOR EVALUATION FUNCTION '
    t=time()
    evaluation_function = set_evaluation_function(recurrent_model=recurrent_model,
                                                output_model=output_model)
    print '%.2f SEC '%(time()-t)

    print 'COMPILING GENERATOR SAMPLING FUNCTION '
    t=time()
    generation_function = set_generation_function(recurrent_model=recurrent_model,
                                                  output_model=output_model)
    print '%.2f SEC '%(time()-t)


    print 'START TRAINING'
    # for each epoch
    generator_train_cost_list = []
    generator_valid_cost_list = []

    generator_grad_norm_mean = 0.0

    for e in xrange(num_epochs):
        window_size = init_window_size + 10*e

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
            single_data = single_data.reshape(window_size, feature_size)
            train_source_data.append(single_data)

            # target data
            single_data = batch_data[1]
            single_data = single_data.reshape(window_size, feature_size)
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

            generator_updater_output = update_function(*generator_updater_input)
            generator_train_cost = generator_updater_output[1].mean()
            generator_grad_norm  = generator_updater_output[2]

            generator_grad_norm_mean += generator_grad_norm
            train_batch_count += 1


            sampling_seed_data = []
            if train_batch_count%1==0:
                # set valid data stream with proper length (window size)
                num_sec           = 10
                sampling_length   = num_sec*sampling_rate/feature_size
                valid_data_stream = set_valid_datastream(feature_size=feature_size,
                                                         window_size=sampling_length)
                # get train data iterator
                valid_data_iterator = valid_data_stream.get_epoch_iterator()

                # for each batch
                valid_batch_count = 0
                valid_batch_size = 0
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

                    if valid_batch_size<10:
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

                    evaluation_output = evaluation_function(*generator_evaluator_input)
                    valid_output_data  = evaluation_output[0]
                    generator_valid_cost = evaluation_output[1].mean()

                    valid_cost_mean += generator_valid_cost
                    valid_batch_count += 1

                    valid_target_data = numpy.swapaxes(valid_target_data, axis1=0, axis2=1)
                    valid_target_data = valid_target_data.reshape((valid_target_data.shape[0], -1))
                    valid_target_data = valid_target_data*(2.**15)
                    valid_target_data = valid_target_data.astype(numpy.int16)

                    valid_output_data = numpy.swapaxes(valid_output_data, axis1=0, axis2=1)
                    valid_output_data = valid_output_data.reshape((valid_target_data.shape[0], -1))
                    valid_output_data = valid_output_data*(2.**15)
                    valid_output_data = valid_output_data.astype(numpy.int16)

                    save_wavfile(valid_target_data, model_name+'_original')
                    save_wavfile(valid_output_data, model_name+'_reconstr')

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

            if train_batch_count%1==0:
                num_samples = 10
                num_sec     = 10
                sampling_length = num_sec*sampling_rate/feature_size

                curr_input_data  = sampling_seed_data[0][:num_samples]
                prev_hidden_data = np_rng.normal(size=(num_layers, num_samples, hidden_size)).astype(floatX)
                prev_hidden_data = numpy.clip(prev_hidden_data, -1.0, 1.0)
                prev_cell_data   = np_rng.normal(size=(num_layers, num_samples, hidden_size)).astype(floatX)
                output_data      = numpy.zeros(shape=(sampling_length, num_samples, feature_size))
                for s in xrange(sampling_length):


                    generator_input = [curr_input_data,
                                       prev_hidden_data,
                                       prev_cell_data]

                    [curr_input_data, prev_hidden_data, prev_cell_data] = generation_function(*generator_input)

                    output_data[s] = curr_input_data

                sample_data = numpy.swapaxes(output_data, axis1=0, axis2=1)
                sample_data = sample_data.reshape((num_samples, -1))
                sample_data = sample_data*(2.**15)
                sample_data = sample_data.astype(numpy.int16)
                save_wavfile(sample_data, model_name+'_sample')

if __name__=="__main__":
    feature_size  = 16
    hidden_size   = 64
    learning_rate = 1e-4
    num_layers    = 3
    init_window   = 100

    model_name = 'lstm_stack_layer' \
                 + '_LAYER{}'.format(int(num_layers)) \
                 + '_FEATURE{}'.format(int(feature_size)) \
                 + '_HIDDEN{}'.format(int(hidden_size)) \
                 + '_LR{}'.format(int(-numpy.log10(learning_rate))) \

    # set model
    recurrent_model = set_recurrent_model(input_size=feature_size,
                                          hidden_size=hidden_size,
                                          num_layers=num_layers)

    output_model    = set_output_model(num_layers=num_layers,
                                       hidden_size=hidden_size,
                                       input_size=feature_size)

    # set optimizer
    optimizer = RmsProp(learning_rate=learning_rate).update_params

    # train model
    train_model(feature_size=feature_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                init_window_size=init_window,
                recurrent_model=recurrent_model,
                output_model=output_model,
                model_optimizer=optimizer,
                num_epochs=10,
                model_name=model_name)
