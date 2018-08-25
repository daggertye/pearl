import tensorflow as tf

def mlp(input_placeholder, output_size, scope, hidden_layers=1, size=64, activation=tf.tanh, output_activation=None):
    """
    Builds a mlp (multi-layer-perceptron) with certain hyperparameters

    Params
    ------
        input_placeholder (tf.Tensor) : 
            the input tensor, should be of shape [None, input_size] for mlp, but other shapes (of size >= 2) work
        ouput_size (int) :
            the output_size.
        scope (string) :
            name of everything
        hidden_layers (int -- 1) : 
            number of hidden layers
        size (int -- 64) :
            number of neurons per hidden layer
        activation (tf activation -- tf.tanh) :
            activation function for hidden layers
        output_activation (tf activation -- None) :
            activation function for final layer
    
    Returns
    -------
        tf.Tensor : the output tensor of shape input_placeholder.shape[0 : -2] + [output_size]
    """
    with tf.variable_scope(scope):
        temp = input_placeholder
        for _ in range(hidden_layers): temp = tf.layers.dense(temp, size, activation=activation)
        temp = tf.layers.dense(temp, output_size, activation=output_activation)
        return temp