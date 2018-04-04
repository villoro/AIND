"""
    Neural networks models for speech
"""

from keras.models import Model
from keras.layers import (
    Input, BatchNormalization, Activation,
    Conv1D, LeakyReLU, Dropout,
    Bidirectional, SimpleRNN, GRU, TimeDistributed, Dense
)


def cnn_output_length(input_length, filter_size, border_mode, stride, dilation=1):
    """ Compute the length of the output sequence after 1D convolution along
        time. Note that this function is in line with the function used in
        Convolution1D class from Keras.
    Params:
        input_length (int): Length of the input sequence.
        filter_size (int): Width of the convolution kernel.
        border_mode (str): Only support `same` or `valid`.
        stride (int): Stride size used in 1D convolution.
        dilation (int)
    """
    if input_length is None:
        return None
    assert border_mode in {'same', 'valid'}
    dilated_filter_size = filter_size + (filter_size - 1) * (dilation - 1)
    if border_mode == 'same':
        output_length = input_length
    elif border_mode == 'valid':
        output_length = input_length - dilated_filter_size + 1
    return (output_length + stride - 1) // stride


def simple_rnn_model(input_dim, output_dim=29):
    """
        MODEL 0: Simple RNN model with only one recurrent layer

        Args:
            input_dim:  dimension of input data
            output_dim: number of possible outputs
    """

    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))

    # Add recurrent layer
    simp_rnn = GRU(output_dim, return_sequences=True,
                   implementation=2, name='rnn')(input_data)

    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(simp_rnn)

    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: x
    print(model.summary())
    return model


def rnn_model(input_dim, units, activation, output_dim=29):
    """
        MODEL 1: RNN with batch_norm and a time distributed layer

        Args:
            input_dim:  dimension of input data
            units:      rnn dimension
            activation: activation function for RNN
            output_dim: number of possible outputs
    """

    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))

    # Add recurrent layer
    simp_rnn = GRU(units, activation=activation,
                   return_sequences=True, implementation=2, name='rnn')(input_data)

    # Add batch normalization
    bn_rnn = BatchNormalization(name='bn_rnn')(simp_rnn)

    # Add a TimeDistributed(Dense(output_dim)) layer
    time_dense = TimeDistributed(Dense(output_dim))(bn_rnn)

    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(time_dense)

    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: x
    print(model.summary())
    return model


def cnn_rnn_model(input_dim, filters, kernel_size, stride,
                  border_mode, units, output_dim=29):
    """
        MODEL 2: NN with convolutional and recurrent layers

        Args:
            input_dim:      dimension of input data
            filters:        number of filters in the convolution
            kernel_size:    convolution kernel size
            stride:         convolutional stride
            border_mode:    type of border [valid/same]
            units:          rnn dimension
            output_dim:     number of possible outputs
    """

    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))

    # Add convolutional layer
    conv_1d = Conv1D(filters, kernel_size, strides=stride, padding=border_mode,
                     activation='relu', name='conv1d')(input_data)

    # Add batch normalization
    bn_cnn = BatchNormalization(name='bn_conv_1d')(conv_1d)

    # Add a recurrent layer
    simp_rnn = SimpleRNN(units, activation='relu',
                         return_sequences=True, name='rnn')(bn_cnn)

    # Add batch normalization
    bn_rnn = BatchNormalization(name='bn_rnn')(simp_rnn)

    # Add a TimeDistributed(Dense(output_dim)) layer
    time_dense = TimeDistributed(Dense(output_dim))(bn_rnn)

    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(time_dense)

    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: cnn_output_length(x, kernel_size, border_mode, stride)
    print(model.summary())
    return model


def deep_rnn_model(input_dim, units, recur_layers, output_dim=29):
    """
        MODEL 3: NN that has 'N' recurrent layers with batch_norm and a time distributed layer

        Args:
            input_dim:      dimension of input data
            units:          rnn dimension
            recur_layers:   number of recurrent layers
            output_dim:     number of possible outputs
    """

    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))

    # Add recurrent layers, each with batch normalization
    rnn = input_data
    for i in range(recur_layers):
        rnn = GRU(units, return_sequences=True,
                  implementation=2, name='rnn{}'.format(i))(rnn)
        rnn = BatchNormalization(name='bn_rnn{}'.format(i))(rnn)

    # Add a TimeDistributed(Dense(output_dim)) layer
    time_dense = TimeDistributed(Dense(output_dim))(rnn)

    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(time_dense)

    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: x
    print(model.summary())
    return model


def bidirectional_rnn_model(input_dim, units, output_dim=29):
    """
        MODEL 4: bidirectional RNN with batch_norm and a time distributed layer

        Args:
            input_dim:      dimension of input data
            units:          rnn dimension
            recur_layers:   number of recurrent layers
            output_dim:     number of possible outputs
    """

    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))

    # Add bidirectional recurrent layer
    bidir_rnn = Bidirectional(GRU(units, return_sequences=True, implementation=2,
                                  name='bidirectional'))(input_data)

    # Add a TimeDistributed(Dense(output_dim)) layer
    time_dense = TimeDistributed(Dense(output_dim))(bidir_rnn)

    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(time_dense)

    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: x
    print(model.summary())
    return model


def final_model(input_dim, filters, kernel_size, stride,
                border_mode, units, recur_layers, td_layers,
                dropout, alpha=0.1, output_dim=29):
    """
        MODEL 5+: NN that has 1 convolutional layer followed by some recurrent layers.
        Each recurrent layer has batch_norm.
        Then it has some time_distributed layers with batch_norm and leaky_relu as activation.

        Args:
            input_dim:      dimension of input data
            filters:        number of filters in the convolution
            kernel_size:    convolution kernel size
            stride:         convolutional stride
            border_mode:    type of border [valid/same]
            units:          rnn dimension
            recur_layers:   number of recurrent layers
            td_layers:      number of time distributed layers
            dropout:        dropout
            alpha:          left slope for the leaky relu
            output_dim:     number of possible outputs
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))

    # Convolution layer
    rnn = Conv1D(filters, kernel_size, strides=stride, padding=border_mode,
                 name='conv1d')(input_data)
    rnn = LeakyReLU(alpha=alpha, name="leaky_conv1d")(rnn)
    rnn = BatchNormalization(name='bn_conv1d')(rnn)

    # Recurrent layers
    rnn = input_data
    for i in range(recur_layers):
        rnn = Bidirectional(GRU(units, return_sequences=True,
                                implementation=2, name='rnn_{}'.format(i)))(rnn)
        rnn = BatchNormalization(name='bn_rnn_{}'.format(i))(rnn)

    # Time distributed layers
    for i in range(td_layers):
        rnn = TimeDistributed(Dense(output_dim, name="td_{}".format(i)))(rnn)
        rnn = LeakyReLU(alpha=alpha, name="leaky_td_{}".format(i))(rnn)
        rnn = BatchNormalization(name='bn_td_{}'.format(i))(rnn)

    rnn = Dropout(dropout)(rnn)

    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(rnn)

    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: x
    print(model.summary())
    return model
