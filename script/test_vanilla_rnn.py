__author__ = 'KimTS'
import theano
import numpy
from theano import tensor
from data.window import Window
from layer.layers import LinearLayer, RecurrentLayer
from layer.layer_utils import get_tensor_output, get_model_updates
from utils.utils import merge_dicts
from optimizer.rmsprop import RmsProp
from numpy.random import RandomState
from theano.sandbox.rng_mrg import MRG_RandomStreams
theano_rng = MRG_RandomStreams(42)
np_rng = RandomState(42)

floatX = theano.config.floatX
input_feature_size = 1
hidden_size = 100

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
    return layers

def set_datastream(window_size=100,
                   offset=1,
                   youtube_id='XqaJ2Ol5cC4'):
    from fuel.datasets.youtube_audio import YouTubeAudio
    data_stream = YouTubeAudio(youtube_id).get_example_stream()
    data_stream = Window(offset=offset,
                         source_window=window_size,
                         target_window=window_size,
                         overlapping=True,
                         data_stream=data_stream)
    return data_stream

def set_update_function(recurrent_model,
                        output_model,
                        recurrent_optimizer,
                        output_optimizer):
    # set input data (time_step*num_samples*features)
    input_seq   = tensor.tensor3(name='input_seq', dtype=floatX)
    # set target data (time_step*num_samples*output_size)
    target_seq  = tensor.tensor3(name='target_seq', dtype=floatX)

    # grad clip
    grad_clip = tensor.scalar(name='grad_clip', dtype=floatX)

    # get hidden data
    hidden_seq = get_tensor_output(input=[input_seq,], layers=recurrent_model, is_training=True)
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
                                               use_grad_clip=grad_clip)

    output_cost         = sample_cost.mean()
    output_updates_dict = get_model_updates(layers=output_model,
                                            cost=output_cost,
                                            optimizer=output_optimizer,
                                            use_grad_clip=grad_clip)

    update_function_inputs  = [input_seq,
                               target_seq,
                               grad_clip]
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


    for e in xrange(num_epochs):
        data_iterator = data_stream.get_epoch_iterator()
        for batch_idx, batch_data in enumerate(data_iterator):
            if numpy.ndim(batch_data[0])==2:
                input_seq  = numpy.expand_dims(batch_data[0], axis=0)
                input_seq  = numpy.swapaxes(input_seq, axis1=0, axis2=1)
                target_seq = numpy.expand_dims(batch_data[1], axis=0)
                target_seq = numpy.swapaxes(target_seq, axis1=0, axis2=1)
            else:
                input_seq  = numpy.swapaxes(batch_data[0], axis1=0, axis2=1)
                target_seq = numpy.swapaxes(batch_data[1], axis1=0, axis2=1)
            mask_seq    = numpy.ones(shape=(input_seq.shape[1], input_seq.shape[0]), dtype=floatX)
            grad_clip   = 0.0

            # update model
            print 'input_seq.shape : ', input_seq.shape
            print 'mask_seq.shape : ', mask_seq.shape
            print 'target_seq.shape : ', target_seq.shape
            raw_input()
            update_input  = [input_seq, target_seq, grad_clip]
            update_output = update_function(*update_input)

            # update result
            hidden_seq  = update_output[0].swapaxes(axis1=0, axis2=1)
            output_seq  = update_output[1].swapaxes(axis1=0, axis2=1)
            sample_cost = update_output[2]

            print sample_cost
            raw_input()



if __name__=="__main__":
    window_size   = 100
    hidden_size   = 10
    learning_rate = 1e-2

    model_name = 'vanilla_rnn_' \
                 + '_WINDOW{}'.format(int(window_size)) \
                 + '_HIDDEN{}'.format(int(hidden_size)) \
                 + '_LR{}'.format(int(-numpy.log10(learning_rate))) \

    # set model
    recurrent_model = set_recurrent_model(input_size=1, hidden_size=hidden_size)
    output_model    = set_output_model(input_size=hidden_size, output_size=1)

    # set optimizer
    recurrent_optimizer = RmsProp(learning_rate=learning_rate).update_params
    output_optimizer    = RmsProp(learning_rate=learning_rate).update_params

    # set data stream
    data_stream =set_datastream(window_size=window_size)

    # train model
    train_model(recurrent_model=recurrent_model,
                output_model=output_model,
                recurrent_optimizer=recurrent_optimizer,
                output_optimizer=output_optimizer,
                data_stream=data_stream,
                num_epochs=100,
                model_name=model_name)