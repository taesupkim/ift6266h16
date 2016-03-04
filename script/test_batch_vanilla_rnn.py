__author__ = 'KimTS'
import theano
import numpy
from theano import tensor
from fuel.datasets.hdf5 import H5PYDataset
from fuel.streams import DataStream
from fuel.schemes import ShuffledScheme
from layer.activations import Tanh
from layer.layers import LinearLayer, RecurrentLayer
from layer.layer_utils import get_tensor_output, get_model_updates
from utils.utils import merge_dicts
from optimizer.rmsprop import RmsProp
from numpy.random import RandomState
from theano.sandbox.rng_mrg import MRG_RandomStreams
from utils.display import plot_learning_curve
theano_rng = MRG_RandomStreams(42)
np_rng = RandomState(42)

floatX = theano.config.floatX



def set_recurrent_model(input_size, hidden_size):
    layers = []
    layers.append(RecurrentLayer(input_dim=input_size,
                                 hidden_dim=hidden_size,
                                 name='recurrent_layer'))

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
                        recurrent_optimizer,
                        output_optimizer):
    # set input data (time_step*num_samples*features)
    input_seq   = tensor.tensor3(name='input_seq', dtype=floatX)
    # set target data (time_step*num_samples*output_size)
    target_seq  = tensor.tensor3(name='target_seq', dtype=floatX)

    # truncate grad
    truncate_grad_step = tensor.scalar(name='truncate_grad_step', dtype='int32')

    # get hidden data
    hidden_seq = get_tensor_output(input=[input_seq, None, None, truncate_grad_step], layers=recurrent_model, is_training=True)
    # get prediction data
    output_seq = get_tensor_output(input=hidden_seq, layers=output_model, is_training=True)

    # get cost (here mask_seq is like weight, sum over feature)
    sequence_cost = tensor.sqr(output_seq-target_seq)
    sample_cost   = tensor.sum(sequence_cost, axis=(0, 2))

    # get model updates
    recurrent_cost         = sample_cost.mean()
    recurrent_updates_dict = get_model_updates(layers=recurrent_model,
                                               cost=recurrent_cost,
                                               optimizer=recurrent_optimizer,
                                               use_grad_clip=1.0)

    output_cost         = sample_cost.mean()
    output_updates_dict = get_model_updates(layers=output_model,
                                            cost=output_cost,
                                            optimizer=output_optimizer,
                                            use_grad_clip=1.0)

    update_function_inputs  = [input_seq,
                               target_seq,
                               truncate_grad_step]
    update_function_outputs = [hidden_seq,
                               output_seq,
                               sample_cost]

    update_function = theano.function(inputs=update_function_inputs,
                                      outputs=update_function_outputs,
                                      updates=merge_dicts([recurrent_updates_dict, output_updates_dict]),
                                      on_unused_input='ignore')

    return update_function

def set_generation_function(recurrent_model, output_model):
    # set input data (1*num_samples*features)
    input_data  = tensor.matrix(name='input_seq', dtype=floatX)
    # set init hidden(num_samples*hidden_size)
    prev_hidden_data = tensor.matrix(name='prev_hidden_data', dtype=floatX)

    # get hidden data
    cur_hidden_data = get_tensor_output(input=[input_data, prev_hidden_data], layers=recurrent_model, is_training=False)
    # get prediction data
    output_data = get_tensor_output(input=cur_hidden_data, layers=output_model, is_training=False)

    # input data
    generation_function_inputs  = [input_data,
                                   prev_hidden_data]
    generation_function_outputs = [prev_hidden_data,
                                   output_data]

    generation_function = theano.function(inputs=generation_function_inputs,
                                          outputs=generation_function_outputs,
                                          on_unused_input='ignore')
    return generation_function

def train_model(recurrent_model,
                output_model,
                recurrent_optimizer,
                output_optimizer,
                data_stream,
                num_epochs,
                model_name):

    update_function = set_update_function(recurrent_model=recurrent_model,
                                          output_model=output_model,
                                          recurrent_optimizer=recurrent_optimizer,
                                          output_optimizer=output_optimizer)

    generation_function = set_generation_function(recurrent_model=recurrent_model,
                                                  output_model=output_model)


    cost_list = []
    for e in xrange(num_epochs):
        data_iterator = data_stream.get_epoch_iterator()
        for batch_idx, batch_data in enumerate(data_iterator):
            input_seq  = numpy.swapaxes(batch_data[0], axis1=0, axis2=1)
            target_seq = numpy.swapaxes(batch_data[1], axis1=0, axis2=1)

            # normalize into [-1., 1.]
            input_seq  = (input_seq/(2.**15)).astype(floatX)
            target_seq = (target_seq/(2.**15)).astype(floatX)

            truncate_grad_step = 10

            # update model
            update_input  = [input_seq,
                             target_seq,
                             truncate_grad_step]
            update_output = update_function(*update_input)

            # update result
            sample_cost = update_output[2].mean()
            if (batch_idx+1)%100==0:
                print e, batch_idx, sample_cost
                cost_list.append(sample_cost)

            if (batch_idx+1)%1000==0:
                plot_learning_curve(cost_values=[cost_list,],
                                    cost_names=['Input cost (train)',],
                                    save_as=model_name+'.png',
                                    legend_pos='upper left')

if __name__=="__main__":
    youtube_id    = 'XqaJ2Ol5cC4'
    window_size   = 100
    hidden_size   = 100
    batch_size    = 64
    learning_rate = 1e-5

    data_path  = '/data/lisatmp4/taesup/data/YouTubeAudio/{}_{}.hdf5'.format(youtube_id, window_size)
    model_name = 'vanilla_rnn_' \
                 + '_WINDOW{}'.format(int(window_size)) \
                 + '_HIDDEN{}'.format(int(hidden_size)) \
                 + '_BATCH{}'.format(int(batch_size)) \
                 + '_LR{}'.format(int(-numpy.log10(learning_rate))) \

    # set model
    recurrent_model = set_recurrent_model(input_size=1, hidden_size=hidden_size)
    output_model    = set_output_model(input_size=hidden_size, output_size=1)

    # set optimizer
    recurrent_optimizer = RmsProp(learning_rate=learning_rate).update_params
    output_optimizer    = RmsProp(learning_rate=learning_rate).update_params

    # set data stream
    data_stream =set_datastream(data_path=data_path, batch_size=batch_size)

    # train model
    train_model(recurrent_model=recurrent_model,
                output_model=output_model,
                recurrent_optimizer=recurrent_optimizer,
                output_optimizer=output_optimizer,
                data_stream=data_stream,
                num_epochs=100,
                model_name=model_name)