__author__ = 'KimTS'
import theano
import numpy
from theano import tensor
from fuel.datasets.hdf5 import H5PYDataset
from fuel.streams import DataStream
from fuel.schemes import ShuffledScheme
from util.utils import save_wavfile
from layer.activations import Tanh
from layer.layers import LinearLayer, LstmLayer
from layer.layer_utils import get_tensor_output, get_model_updates
from optimizer.rmsprop import RmsProp
from numpy.random import RandomState
from theano.sandbox.rng_mrg import MRG_RandomStreams
from utils.display import plot_learning_curve
theano_rng = MRG_RandomStreams(42)
np_rng = RandomState(42)

floatX = theano.config.floatX
input_feature_size = 1
hidden_size = 100

def set_recurrent_model(input_size, hidden_size):
    layers = []
    layers.append(LstmLayer(input_dim=input_size,
                            hidden_dim=hidden_size,
                            name='lstm_layer'))

    return layers

def set_output_model(input_size, output_size):
    layers = []
    layers.append(LinearLayer(input_dim=input_size,
                              output_dim=output_size,
                              name='output_layer'))
    layers.append(Tanh(name='output_squeeze_layer'))
    return layers

def set_datastream(data_path, batch_size):
    dataset = H5PYDataset(file_or_path=data_path,
                          which_sets=('train',),
                          sources=('input_feature', 'target_feature'))
    data_stream = DataStream.default_stream(dataset=dataset,
                                            iteration_scheme=ShuffledScheme(batch_size=batch_size,
                                                                            examples=dataset.num_examples))
    return data_stream

def set_update_function(recurrent_model,
                        output_model,
                        optimizer,
                        grad_clip=1.0):
    # set input data (time_length * num_samples * input_dims)
    input_data  = tensor.tensor3(name='input_data', dtype=floatX)
    # set input mask (time_length * num_samples)
    input_mask  = tensor.matrix(name='input_mask', dtype=floatX)
    # set init hidden/cell data (num_samples * hidden_dims)
    init_hidden = tensor.matrix(name='init_hidden', dtype=floatX)
    init_cell   = tensor.matrix(name='init_cell', dtype=floatX)

    # truncate grad
    truncate_grad_step = tensor.scalar(name='truncate_grad_step', dtype='int32')
    # set target data (time_length * num_samples * output_dims)
    target_data = tensor.tensor3(name='target_data', dtype=floatX)

    # get hidden data
    input_list  = [input_data, None, None, None, truncate_grad_step]
    hidden_data = get_tensor_output(input=input_list,
                                    layers=recurrent_model,
                                    is_training=True)[0]
    # get prediction data
    output_data = get_tensor_output(input=hidden_data,
                                    layers=output_model,
                                    is_training=True)

    # get cost (here mask_seq is like weight, sum over feature, and time)
    sample_cost = tensor.sqr(output_data-target_data)
    sample_cost = tensor.sum(sample_cost, axis=(0, 2))

    # get model updates
    model_cost         = sample_cost.mean()
    model_updates_dict = get_model_updates(layers=recurrent_model+output_model,
                                           cost=model_cost,
                                           optimizer=optimizer,
                                           use_grad_clip=grad_clip)

    update_function_inputs  = [input_data,
                               input_mask,
                               init_hidden,
                               init_cell,
                               target_data,
                               truncate_grad_step]
    update_function_outputs = [hidden_data,
                               output_data,
                               sample_cost]

    update_function = theano.function(inputs=update_function_inputs,
                                      outputs=update_function_outputs,
                                      updates=model_updates_dict,
                                      on_unused_input='ignore')

    return update_function

def set_generation_function(recurrent_model, output_model):
    # set input data (1*num_samples*features)
    input_data  = tensor.matrix(name='input_seq', dtype=floatX)
    # set init hidden/cell(num_samples*hidden_size)
    prev_hidden_data = tensor.matrix(name='prev_hidden_data', dtype=floatX)
    prev_cell_data   = tensor.matrix(name='prev_cell_data', dtype=floatX)

    # get hidden data
    recurrent_data = get_tensor_output(input=[input_data, prev_hidden_data, prev_cell_data], layers=recurrent_model, is_training=False)
    cur_hidden_data = recurrent_data[0]
    cur_cell_data   = recurrent_data[1]

    # get prediction data
    output_data = get_tensor_output(input=cur_hidden_data, layers=output_model, is_training=False)

    # input data
    generation_function_inputs  = [input_data,
                                   prev_hidden_data,
                                   prev_cell_data]
    generation_function_outputs = [cur_hidden_data,
                                   cur_cell_data,
                                   output_data]

    generation_function = theano.function(inputs=generation_function_inputs,
                                          outputs=generation_function_outputs,
                                          on_unused_input='ignore')
    return generation_function

def train_model(recurrent_model,
                output_model,
                num_hiddens,
                model_optimizer,
                data_stream,
                num_epochs,
                model_name):

    update_function = set_update_function(recurrent_model=recurrent_model,
                                          output_model=output_model,
                                          optimizer=model_optimizer,
                                          grad_clip=1.0)

    generation_function = set_generation_function(recurrent_model=recurrent_model,
                                                  output_model=output_model)

    # for each epoch
    cost_list = []
    cnt = 0
    for e in xrange(num_epochs):
        # get data iterator
        data_iterator = data_stream.get_epoch_iterator()
        # for each batch
        for batch_idx, batch_data in enumerate(data_iterator):
            input_data  = numpy.swapaxes(batch_data[0], axis1=0, axis2=1)
            input_mask  = numpy.ones(shape=input_data.shape[:2], dtype=floatX)
            target_data = numpy.swapaxes(batch_data[1], axis1=0, axis2=1)

            input_data  = (input_data/(2.**15)).astype(floatX)
            target_data = (target_data/(2.**15)).astype(floatX)

            time_length = input_data.shape[0]
            num_samples = input_data.shape[1]

            truncate_grad_step = int(numpy.clip(numpy.asarray(0.001*cnt), 1, time_length))
            truncate_grad_step = time_length
            cnt = cnt + 1

            # update model
            update_input  = [input_data,
                             input_mask,
                             None,
                             None,
                             target_data,
                             truncate_grad_step]
            update_output = update_function(*update_input)

            # update result
            sample_cost = update_output[2].mean()
            if (batch_idx+1)%1000==0:
                print 'epoch {}, batch_idx {} : cost {} truncate({})'.format(e, batch_idx, sample_cost, truncate_grad_step)
                cost_list.append(sample_cost)

            if (batch_idx+1)%1000==0:
                plot_learning_curve(cost_values=[cost_list,],
                                    cost_names=['Input cost (train)',],
                                    save_as=model_name+'.png',
                                    legend_pos='upper left')

            if (batch_idx+1)%10000==0:
                generation_sample = 10
                generation_length = 1000
                input_data  = numpy.random.uniform(low=-1.0, high=1.0, size=(generation_sample, input_feature_size)).astype(floatX)
                hidden_data = numpy.random.uniform(low=-1.0, high=1.0, size=(generation_sample, num_hiddens)).astype(floatX)
                output_data = numpy.zeros(shape=(generation_length, generation_sample, input_feature_size))
                for t in xrange(generation_length):
                    [hidden_data, cell_data, input_data] = generation_function(input_data, hidden_data)
                    output_data[t] = input_data

                output_data = numpy.swapaxes(output_data, axis1=0, axis2=1)
                output_data = output_data*(2.**15)
                output_data = output_data.astype(numpy.int16)
                save_wavfile(output_data, model_name+'_sample')




if __name__=="__main__":
    youtube_id    = 'XqaJ2Ol5cC4'
    window_size   = 1000
    hidden_size   = 100
    batch_size    = 64
    learning_rate = 1e-5

    data_path  = '/data/lisatmp4/taesup/data/YouTubeAudio/{}_{}.hdf5'.format(youtube_id, window_size)

    model_name = 'lstm_rnn_' \
                 + '_WINDOW{}'.format(int(window_size)) \
                 + '_HIDDEN{}'.format(int(hidden_size)) \
                 + '_LR{}'.format(int(-numpy.log10(learning_rate))) \

    # set model
    recurrent_model = set_recurrent_model(input_size=1, hidden_size=hidden_size)
    output_model    = set_output_model(input_size=hidden_size, output_size=1)

    # set optimizer
    optimizer = RmsProp(learning_rate=learning_rate).update_params

    # set data stream
    data_stream =set_datastream(data_path, batch_size)

    # train model
    train_model(recurrent_model=recurrent_model,
                output_model=output_model,
                num_hiddens=hidden_size,
                model_optimizer=optimizer,
                data_stream=data_stream,
                num_epochs=100,
                model_name=model_name)