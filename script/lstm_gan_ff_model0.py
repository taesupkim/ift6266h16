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
from optimizer.adagrad import AdaGrad
from numpy.random import RandomState
from theano.sandbox.rng_mrg import MRG_RandomStreams
from scipy.io import wavfile
from utils.utils import merge_dicts
theano_rng = MRG_RandomStreams(42)
np_rng = RandomState(42)

floatX = theano.config.floatX
sampling_rate = 16000

def set_generator_model(input_size,
                        hidden_size):
    layers = []
    layers.append(SingleLstmGanForceLayer(input_dim=input_size,
                                          hidden_dim=hidden_size,
                                          name='generator_model'))
    return layers

def set_discriminator_feature_model(hidden_size,
                                    feature_size):
    layers = []
    layers.append(LinearLayer(input_dim=hidden_size,
                              output_dim=hidden_size/2,
                              name='discriminator_feature_linear0'))
    layers.append(Relu(name='discriminator_feature_relu0'))
    layers.append(LinearLayer(input_dim=hidden_size/2,
                              output_dim=feature_size,
                              name='discriminator_feature_linear1'))
    layers.append(Relu(name='discriminator_feature_relu1'))
    return layers

def set_discriminator_output_model(feature_size):
    layers = []
    layers.append(LinearLayer(input_dim=feature_size*2,
                              output_dim=feature_size,
                              name='discriminator_output_linear0'))
    layers.append(Relu(name='discriminator_output_relu0'))
    layers.append(LinearLayer(input_dim=feature_size,
                              output_dim=1,
                              name='discriminator_output_linear1'))
    layers.append(Logistic(name='discriminator_output_linear1'))
    return layers

def set_gan_update_function(generator_model,
                            discriminator_feature_model,
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
    generator_input_data_list = [input_sequence, 1]

    # get generator output data
    generator_output = generator_model[0].forward(generator_input_data_list, is_training=True)
    output_sequence  = generator_output[0]
    data_hidden      = generator_output[1]
    data_cell        = generator_output[2]
    model_hidden     = generator_output[3]
    model_cell       = generator_output[4]

    # condition_cell    = data_cell[:-1]
    # condition_cell    = theano.gradient.disconnected_grad(condition_cell)
    # condition_feature = get_tensor_output(condition_cell,
    #                                       discriminator_feature_model,
    #                                       is_training=True)
    #
    # positive_cell    = data_cell[1:]
    # positive_feature = get_tensor_output(positive_cell,
    #                                      discriminator_feature_model,
    #                                      is_training=True)
    # negative_cell    = model_cell[1:]
    # negative_feature = get_tensor_output(negative_cell,
    #                                      discriminator_feature_model,
    #                                      is_training=True)

    condition_hid    = data_hidden[:-1]
    condition_hid    = theano.gradient.disconnected_grad(condition_hid)
    condition_feature = get_tensor_output(condition_hid,
                                          discriminator_feature_model,
                                          is_training=True)

    positive_hid     = data_hidden[1:]
    positive_feature = get_tensor_output(positive_hid,
                                         discriminator_feature_model,
                                         is_training=True)
    negative_hid     = model_hidden[1:]
    negative_feature = get_tensor_output(negative_hid,
                                         discriminator_feature_model,
                                         is_training=True)

    positive_pair = tensor.concatenate([condition_feature, positive_feature], axis=2)
    negative_pair = tensor.concatenate([condition_feature, negative_feature], axis=2)

    positive_score = get_tensor_output(positive_pair,
                                       discriminator_output_model,
                                       is_training=True)
    negative_score = get_tensor_output(negative_pair,
                                       discriminator_output_model,
                                       is_training=True)

    generator_gan_cost = tensor.nnet.binary_crossentropy(output=negative_score,
                                                         target=tensor.ones_like(negative_score))

    discriminator_gan_cost = (tensor.nnet.binary_crossentropy(output=positive_score,
                                                              target=tensor.ones_like(positive_score)) +
                              tensor.nnet.binary_crossentropy(output=negative_score,
                                                              target=tensor.zeros_like(negative_score)))

    # set generator update
    generator_updates_cost = generator_gan_cost.mean()
    generator_updates_dict = get_model_updates(layers=generator_model,
                                               cost=generator_updates_cost,
                                               optimizer=generator_optimizer,
                                               use_grad_clip=generator_grad_clipping)

    generator_gradient_dict  = get_model_gradients(generator_model, generator_updates_cost)
    generator_gradient_norm  = 0.
    for grad in generator_gradient_dict:
        generator_gradient_norm += tensor.sum(grad**2)
    generator_gradient_norm  = tensor.sqrt(generator_gradient_norm)

    # set discriminator update
    discriminator_updates_cost = discriminator_gan_cost.mean()
    discriminator_updates_dict = get_model_updates(layers=discriminator_feature_model+discriminator_output_model,
                                                   cost=discriminator_updates_cost,
                                                   optimizer=discriminator_optimizer,
                                                   use_grad_clip=discriminator_grad_clipping)

    discriminator_gradient_dict  = get_model_gradients(discriminator_feature_model+discriminator_output_model,
                                                       discriminator_updates_cost)
    discriminator_gradient_norm  = 0.
    for grad in discriminator_gradient_dict:
        discriminator_gradient_norm += tensor.sum(grad**2)
    discriminator_gradient_norm  = tensor.sqrt(discriminator_gradient_norm)

    square_error = tensor.sqr(target_sequence-output_sequence).sum(axis=2)

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

def set_tf_update_function(generator_model,
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
    output_data_set = generator_model[0].forward(generator_input_data_list, is_training=True)
    output_sequence = output_data_set[0]

    # get square error
    square_error = tensor.sqr(target_sequence-output_sequence).sum(axis=2)

    # set generator update
    tf_updates_cost = square_error.mean()
    tf_updates_dict = get_model_updates(layers=generator_model,
                                        cost=tf_updates_cost,
                                        optimizer=generator_optimizer)

    generator_gradient_dict  = get_model_gradients(generator_model, tf_updates_cost)
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

def set_evaluation_function(generator_model):

    # input sequence data (time_length * num_samples * input_dims)
    input_sequence  = tensor.tensor3(name='input_sequence',
                                     dtype=floatX)
    target_sequence  = tensor.tensor3(name='target_sequence',
                                    dtype=floatX)
    # set generator input data list
    generator_input_data_list = [input_sequence,]

    # get generator output data
    output_data_set = generator_model[0].forward(generator_input_data_list, is_training=True)
    output_sequence = output_data_set[0]

    # get square error
    square_error = tensor.sqr(target_sequence-output_sequence).sum(axis=2)

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


def set_sample_function(generator_model):

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
    output_data_set = generator_model[0].forward(generator_input_data_list, is_training=False)
    sample_data = output_data_set[0]
    hidden_data = output_data_set[1]
    cell_data   = output_data_set[2]

    # input data
    sample_function_inputs  = [init_input_data,
                               init_hidden_data,
                               init_cell_data]
    sample_function_outputs = [sample_data,
                               hidden_data,
                               cell_data]

    sample_function = theano.function(inputs=sample_function_inputs,
                                      outputs=sample_function_outputs,
                                      on_unused_input='ignore')
    return sample_function

def train_model(feature_size,
                hidden_size,
                init_window_size,
                generator_model,
                generator_gan_optimizer,
                generator_tf_optimizer,
                discriminator_feature_model,
                discriminator_output_model,
                discriminator_gan_optimizer,
                num_epochs,
                model_name):

    # generator updater
    print 'COMPILING GAN UPDATE FUNCTION '
    gan_updater = set_gan_update_function(generator_model=generator_model,
                                          discriminator_feature_model=discriminator_feature_model,
                                          discriminator_output_model=discriminator_output_model,
                                          generator_optimizer=generator_gan_optimizer,
                                          discriminator_optimizer=discriminator_gan_optimizer,
                                          generator_grad_clipping=.0,
                                          discriminator_grad_clipping=.0)

    print 'COMPILING TF UPDATE FUNCTION '
    tf_updater = set_tf_update_function(generator_model=generator_model,
                                        generator_optimizer=generator_tf_optimizer,
                                        generator_grad_clipping=.0)
    # sample generator
    print 'COMPILING SAMPLING FUNCTION '
    sample_generator = set_sample_function(generator_model=generator_model)

    print 'READ RAW WAV DATA'
    _, train_raw_data = wavfile.read('/data/lisatmp4/taesup/data/YouTubeAudio/XqaJ2Ol5cC4.wav')
    valid_raw_data  = train_raw_data[160000000:]
    train_raw_data  = train_raw_data[:160000000]
    train_raw_data  = train_raw_data[2000:]
    train_raw_data  = (train_raw_data/(1.15*2.**13)).astype(floatX)
    valid_raw_data  = (valid_raw_data/(1.15*2.**13)).astype(floatX)

    num_train_total_steps = train_raw_data.shape[0]
    num_valid_total_steps = valid_raw_data.shape[0]
    batch_size      = 64

    num_samples      = 10
    last_seq_idx     = num_valid_total_steps-feature_size
    valid_seq_orders = np_rng.permutation(last_seq_idx)
    valid_seq_orders = valid_seq_orders[:last_seq_idx-last_seq_idx%num_samples]
    valid_seq_orders = valid_seq_orders.reshape((-1, num_samples))
    valid_seq_orders = valid_seq_orders[0]
    valid_source_idx  = valid_seq_orders.reshape((num_samples, 1)) + numpy.repeat(numpy.arange(feature_size).reshape((1, feature_size)), num_samples, axis=0)
    valid_source_data = valid_raw_data[valid_source_idx]
    valid_source_data = valid_source_data.reshape((num_samples, feature_size))

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

    train_batch_count = 0
    for e in xrange(num_epochs):
        window_size      = init_window_size + 5*e
        sequence_size    = feature_size*window_size
        last_seq_idx     = num_train_total_steps-(sequence_size+feature_size)
        train_seq_orders = np_rng.permutation(last_seq_idx)
        train_seq_orders = train_seq_orders[:last_seq_idx-last_seq_idx%batch_size]
        train_seq_orders = train_seq_orders.reshape((-1, batch_size))

        # for each batch
        for batch_idx, batch_info in enumerate(train_seq_orders):
            # source data
            train_source_idx  = batch_info.reshape((batch_size, 1)) + numpy.repeat(numpy.arange(sequence_size).reshape((1, sequence_size)), batch_size, axis=0)
            train_source_data = train_raw_data[train_source_idx]
            train_source_data = train_source_data.reshape((batch_size, window_size, feature_size))
            train_source_data = numpy.swapaxes(train_source_data, axis1=0, axis2=1)

            # target data
            train_target_idx  = train_source_idx + feature_size
            train_target_data = train_raw_data[train_target_idx]
            train_target_data = train_target_data.reshape((batch_size, window_size, feature_size))
            train_target_data = numpy.swapaxes(train_target_data, axis1=0, axis2=1)

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
                print '============{}_LENGTH{}============'.format(model_name, window_size)
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


            if train_batch_count%100==0:
                num_sec = 10
                sampling_length = num_sec*sampling_rate/feature_size

                curr_input_data  = valid_source_data
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
    hidden_size   =  800
    lr=1e-4

    model_name = 'LSTM_GAN_HIDDEN_FF(TRUNC_ADAGRAD)' \
                + '_FEATURE{}'.format(int(feature_size)) \
                + '_HIDDEN{}'.format(int(hidden_size)) \

    # generator model
    generator_model = set_generator_model(input_size=feature_size,
                                          hidden_size=hidden_size)

    # discriminator model
    discriminator_feature_model = set_discriminator_feature_model(hidden_size=hidden_size,
                                                                  feature_size=256)
    discriminator_output_model = set_discriminator_output_model(feature_size=256)

    # set optimizer
    tf_generator_optimizer      = RmsProp(learning_rate=0.001).update_params
    gan_generator_optimizer     = RmsProp(learning_rate=0.001).update_params
    gan_discriminator_optimizer = AdaGrad(learning_rate=0.0001).update_params


    train_model(feature_size=feature_size,
                hidden_size=hidden_size,
                init_window_size=100,
                generator_model=generator_model,
                generator_gan_optimizer=gan_generator_optimizer,
                generator_tf_optimizer=tf_generator_optimizer,
                discriminator_feature_model=discriminator_feature_model,
                discriminator_output_model=discriminator_output_model,
                discriminator_gan_optimizer=gan_discriminator_optimizer,
                num_epochs=10,
                model_name=model_name)
