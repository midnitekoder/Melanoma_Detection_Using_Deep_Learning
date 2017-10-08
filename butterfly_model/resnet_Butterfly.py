from keras.models import Model
from keras.layers import (
    Input,
    Activation,
    merge,
    Dense,
    Flatten,
    Lambda
)
from keras.layers.convolutional import (
    Convolution2D,
    MaxPooling2D,
    AveragePooling2D
)
from keras.layers.normalization import BatchNormalization
from keras import backend as K
from keras.regularizers import l2
import numpy as np
import time
import tensorflow as tf

count_of_blocks = 0
nb_total_blocks = 24 
#delete note to self if it works.


###
def get_p_survival(block=0, nb_total_blocks=9, p_survival_end=0.5, mode='linear_decay'):
    """
    See eq. (4) in stochastic depth paper: http://arxiv.org/pdf/1603.09382v1.pdf
    """
    if mode == 'uniform':
        return p_survival_end
    elif mode == 'linear_decay':
        return 1 - ((block + 1) / nb_total_blocks) * (1 - p_survival_end)
    else:
        raise

###
def stochastic_survival(y, p_survival=1.0):
    # binomial random variable
    shape = (1,)
    dtype = K.floatx()
    seed = np.random.randint(10e6)
    p = p_survival
    survival=tf.where(tf.random_uniform(shape, dtype=dtype, seed=seed) <= p,
                    tf.ones(shape, dtype=dtype),
                    tf.zeros(shape, dtype=dtype))

    #survival = K.random_binomial((1,), p=p_survival)
    #survival = tf.where(tf.random_uniform(shape,dtype = dtype,seed = seed) <= p, tf.ones(shape,dtype-dtype), tf.zeros(shape,dtype=dtype))
    # during testing phase:
    # - scale y (see eq. (6))
    # - p_survival effectively becomes 1 for all layers (no layer dropout)
    return K.in_test_phase(tf.constant(p_survival, dtype='float32') * y, survival * y) #note to self: was weirdly spaced before. 

# Helper to build a conv -> BN -> relu block
def _conv_bn_relu(nb_filter, nb_row, nb_col, subsample=(1, 1)):
    def f(input):
        conv = Convolution2D(nb_filter=nb_filter, nb_row=nb_row, nb_col=nb_col, subsample=subsample,
                             init="he_normal", border_mode="same")(input)
        norm = BatchNormalization(mode=0, axis=CHANNEL_AXIS)(conv)
        return Activation("relu")(norm)

    return f


# Helper to build a BN -> relu -> conv block
# This is an improved scheme proposed in http://arxiv.org/pdf/1603.05027v2.pdf
def _bn_relu_conv(nb_filter, nb_row, nb_col, subsample=(1, 1)):
    def f(input):
        norm = BatchNormalization(mode=0, axis=CHANNEL_AXIS)(input)
        activation = Activation("relu")(norm)
        return Convolution2D(nb_filter=nb_filter, nb_row=nb_row, nb_col=nb_col, subsample=subsample,
                             init="he_normal", border_mode="same")(activation)

    return f

#this function is for shortcut connection. It decides the stride based on the size of
#output(residual block) expected and the size of input provided.
#you can see stride_width it is calculated as the ratio of size of input row length by residual(output) row length
#if the depth of input and expected output is same then it means that number of filters do not change from the previous basic_block to this one
# if the depth is different then we use convolution  to create appropriate depth of the shortcut connection
#after that we merge (add) the output of shortcut connection and residual. 
# Adds a shortcut between input and residual block and merges them with "sum"
def _shortcut(input, residual):
    # Expand channels of shortcut to match residual.
    # Stride appropriately to match residual (width, height)
    # Should be int if network architecture is correctly configured.
    stride_width = input._keras_shape[ROW_AXIS] // residual._keras_shape[ROW_AXIS]
    stride_height = input._keras_shape[COL_AXIS] // residual._keras_shape[COL_AXIS]
    equal_channels = residual._keras_shape[CHANNEL_AXIS] == input._keras_shape[CHANNEL_AXIS]

    shortcut = input
    # 1 X 1 conv if shape is different. Else identity.
    if stride_width > 1 or stride_height > 1 or not equal_channels:
        shortcut = Convolution2D(nb_filter=residual._keras_shape[CHANNEL_AXIS],
                                 nb_row=1, nb_col=1,
                                 subsample=(stride_width, stride_height),
                                 init="he_normal", border_mode="valid")(input)
    
    
    return merge([shortcut, residual], mode="sum")
#residual input is (residual) and shortcut is OUTPUT from 2nd convolution layer

# Builds a residual block with repeating bottleneck blocks.
# it creates residual block based on the repetition
# i is the index for the block of a particular size of filters (ie, 64 or 128 or 256 or 512) there are going to many basic blocks of each of the size mentioned so i is the index of the block
# RES - drop residual output 
# note to self what is equivalent to blocks_per_group in SD code?
def _residual_block(block_function, nb_filters, repetitions, is_first_layer=False,block_num=0):
    def f(input):
        for i in range(repetitions):
            init_subsample = (1, 1)
# if i ==0 , ie , first basic block of that number of filters then we need to reduce the dimensions of the output by 2 so init_subsample makes the stride 2 along x as well as y 
            if i == 0 and not is_first_layer:
                init_subsample = (2, 2)
            input = block_function(nb_filters=nb_filters, init_subsample=init_subsample,is_first_block_of_first_layer=(is_first_layer and i == 0),block_num=block_num+i)(input)
            #p_survival = get_p_survival(block=i, nb_total_blocks=nb_total_blocks, p_survival_end=0.5, mode='linear_decay')
            #input = Lambda(stochastic_survival, arguments={'p_survival': p_survival})(input) ## CHANGED Y.O to shortcut
### I'm pretty sure p_survival is 1 if block is 0 but i need to check if there's an error      
# i feel here you have to add code to drop that residual block or not based on bernouli theorem
        return input

    return f

def fused_block(nb_filters, init_subsample=(1, 1),is_first_block_of_first_layer=False,block_num=0):
    def f(input):
        global count_of_blocks
        count_of_blocks=count_of_blocks+1
        block1=basic_block(nb_filters=nb_filters, init_subsample=init_subsample,is_first_block_of_first_layer=is_first_block_of_first_layer,block_num=block_num)(input)
        block2=basic_block(nb_filters=nb_filters, init_subsample=init_subsample,is_first_block_of_first_layer=is_first_block_of_first_layer,block_num=block_num)(input)

        #block1=basic_block(nb_filters=nb_filters,tlbr=count_of_blocks%4, init_subsample=init_subsample)(input)
        #block2=basic_block(nb_filters=nb_filters,tlbr=(count_of_blocks+2)%4, init_subsample=init_subsample)(input)
        return merge([block1,block2], mode="sum")
    return f
# Basic 3 X 3 convolution blocks.
# Use for resnet with layers <= 34
# Follows improved proposed scheme in http://arxiv.org/pdf/1603.05027v2.pdf
def basic_block(nb_filters, init_subsample=(1, 1),is_first_block_of_first_layer=False,block_num=0):
    def f(input):

        if is_first_block_of_first_layer:
            # don't repeat bn->relu since we just did bn->relu->maxpool
            conv1 = Convolution2D(nb_filter=nb_filters,
                                 nb_row=3, nb_col=3,
                                 subsample=init_subsample,
                                 init="he_normal", border_mode="same",
                                 W_regularizer=l2(0.0001))(input)
        else:
            conv1 = _bn_relu_conv(nb_filter=nb_filters, nb_row=3, nb_col=3, subsample=init_subsample)(input)

        residual = _bn_relu_conv(nb_filter=nb_filters, nb_row=3, nb_col=3)(conv1)
        p_survival = get_p_survival(block=block_num, nb_total_blocks=nb_total_blocks, p_survival_end=0.5, mode='linear_decay')
        residual = Lambda(stochastic_survival, arguments={'p_survival': p_survival})(residual) ## CHANGED Y.O to shortcut

        return _shortcut(input, residual)

    return f
        

#nothing to change here
def handle_dim_ordering():
    global ROW_AXIS
    global COL_AXIS
    global CHANNEL_AXIS
    if K.image_dim_ordering() == 'tf':
        ROW_AXIS = 1
        COL_AXIS = 2
        CHANNEL_AXIS = 3
    else:
        CHANNEL_AXIS = 1
        ROW_AXIS = 2
        COL_AXIS = 3


class ResnetBuilder(object):
    @staticmethod
    def build(input_shape, num_outputs, block_fn, repetitions):
        """
        Builds a custom ResNet like architecture.
        :param input_shape: The input shape in the form (nb_channels, nb_rows, nb_cols)

        :param num_outputs: The number of outputs at final softmax layer

        :param block_fn: The block function to use. This is either :func:`basic_block` or :func:`bottleneck`.
        The original paper used basic_block for layers < 50

        :param repetitions: Number of repetitions of various block units.
        At each block unit, the number of filters are doubled and the input size is halved

        :return: The keras model.
        """
        handle_dim_ordering()
        if len(input_shape) != 3:
            raise Exception("Input shape should be a tuple (nb_channels, nb_rows, nb_cols)")

        '''
        # Permute dimension order if necessary
        if K.image_dim_ordering() == 'tf':
            input_shape = (input_shape[1], input_shape[2], input_shape[0])
'''
        input = Input(shape=input_shape)
        conv1 = _conv_bn_relu(nb_filter=64, nb_row=7, nb_col=7, subsample=(2, 2))(input)
        pool1 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), border_mode="same")(conv1)

        block = pool1
        block_num = 0
        nb_filters = 64
        for i, r in enumerate(repetitions):
            block = _residual_block(block_fn, nb_filters=nb_filters, repetitions=r, is_first_layer=i == 0,block_num=block_num)(block)
            block_num+=r
            nb_filters *= 2

        # Classifier block
        pool2 = AveragePooling2D(pool_size=(block._keras_shape[ROW_AXIS],
                                            block._keras_shape[COL_AXIS]),
                                 strides=(1, 1))(block)
        flatten1 = Flatten()(pool2)
        dense = Dense(output_dim=num_outputs, init="he_normal", activation="softmax")(flatten1)

        model = Model(input=input, output=dense)
        return model
    @staticmethod
    def build_ror(input_shape, num_outputs, block_fn, repetitions,level_m):
        """
        Builds a custom ResNet like architecture.
        :param input_shape: The input shape in the form (nb_channels, nb_rows, nb_cols)

        :param num_outputs: The number of outputs at final softmax layer

        :param block_fn: The block function to use. This is either :func:`basic_block` or :func:`bottleneck`.
        The original paper used basic_block for layers < 50

        :param repetitions: Number of repetitions of various block units.
        At each block unit, the number of filters are doubled and the input size is halved
        
        :param level_m: m leveled ROR. Assuming m=3.


        :return: The keras model.
        """
        
        level_m=3
        
        handle_dim_ordering()
        if len(input_shape) != 3:
            raise Exception("Input shape should be a tuple (nb_channels, nb_rows, nb_cols)")

        '''
        # Permute dimension order if necessary
        if K.image_dim_ordering() == 'tf':
            input_shape = (input_shape[1], input_shape[2], input_shape[0])
'''
        input = Input(shape=input_shape)
        conv1 = _conv_bn_relu(nb_filter=64, nb_row=7, nb_col=7, subsample=(2, 2))(input)
        pool1 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), border_mode="same")(conv1)

        block = pool1
        nb_filters = 64
        # What is L? Number of repeats through nb_filters? ####
        
        L = sum(repetitions) 
        l=0
        
        saved = input
        
        for i, r in enumerate(repetitions):
        #for some count =l/3, call shortcut function
        #this code divides the basic blocks into 3 parts irrespective of the number of filters, it takes care of everything and here nothing should be changed.
            if l<L/3 and l+r >L/3 :
                block = _residual_block(block_fn, nb_filters=nb_filters, repetitions=L/3-l, is_first_layer=i == 0,block_num=l)(block)
                block = _shortcut(saved,block) #resblock(block) + prev = ror shortcut
                saved = block
                #if 2L/3 occurs in r-L/3 part... gotta if it and exeecute similar to elif part.
                block = _residual_block(block_fn, nb_filters=nb_filters, repetitions=r-(L/3-l), is_first_layer=i == 0,block_num=l)(block)
            elif l<2*L/3 and l+r> 2*L/3 :
                block = _residual_block(block_fn, nb_filters=nb_filters, repetitions=2*L/3-l, is_first_layer=i == 0,block_num=l)(block)
                block = _shortcut(saved,block)
                saved = block
                block = _residual_block(block_fn, nb_filters=nb_filters, repetitions=r-(2*L/3-l), is_first_layer=i == 0,block_num=l)(block)
            else:
                block = _residual_block(block_fn, nb_filters=nb_filters, repetitions=r, is_first_layer=i == 0,block_num=l)(block)
                if l == L/3 or 2*L/3 :
                    block = _shortcut(saved,block)
                    saved = block
            l+=r
            nb_filters *= 2
        
        block = _shortcut(input, block)  # this is the top most level of shortcut connection that connects input to the first basic block to the output of last basic block..see in my diagram

        # Classifier block
        pool2 = AveragePooling2D(pool_size=(block._keras_shape[ROW_AXIS],
                                            block._keras_shape[COL_AXIS]),
                                 strides=(1, 1))(block)
        flatten1 = Flatten()(pool2)
        dense = Dense(output_dim=num_outputs, init="he_normal", activation="softmax")(flatten1)

        model = Model(input=input, output=dense)
        return model

    @staticmethod
    def build_resnet_18(input_shape, num_outputs):
        return ResnetBuilder.build(input_shape, num_outputs, basic_block, [2, 2, 2, 2])

    @staticmethod
    def build_resnet_34(input_shape, num_outputs):
        return ResnetBuilder.build(input_shape, num_outputs, basic_block, [3, 4, 6, 3])

    @staticmethod
    def build_resnet_50(input_shape, num_outputs):
        return ResnetBuilder.build(input_shape, num_outputs, bottleneck, [3, 4, 6, 3])

    @staticmethod
    def build_resnet_101(input_shape, num_outputs):
        return ResnetBuilder.build(input_shape, num_outputs, bottleneck, [3, 4, 23, 3])

    @staticmethod
    def build_resnet_152(input_shape, num_outputs):
        return ResnetBuilder.build(input_shape, num_outputs, bottleneck, [3, 8, 36, 3])
    @staticmethod    
    def build_ror_20(input_shape,num_outputs,level_m):
        return ResnetBuilder.build_ror(input_shape,num_outputs,basic_block,[3, 3,3],level_m)
    @staticmethod
    def build_ror_50(input_shape,num_outputs,level_m):
        return ResnetBuilder.build_ror(input_shape,num_outputs,basic_block,[8,8,8,8,8,8],level_m)
    @staticmethod
    def build_butternet_101(input_shape, num_outputs):
        return ResnetBuilder.build(input_shape, num_outputs, fused_block, [3, 4, 23, 3])
    @staticmethod
    def build_butternet_10(input_shape, num_outputs):
        return ResnetBuilder.build(input_shape, num_outputs, fused_block, [1,1,1,1])
    @staticmethod
    def build_butternet_92(input_shape, num_outputs):
        return ResnetBuilder.build(input_shape, num_outputs, fused_block, [20,1,1,1])
    @staticmethod
    def build_butternet_26(input_shape, num_outputs):
        return ResnetBuilder.build(input_shape, num_outputs, fused_block, [2,2,1,1])
    @staticmethod
    def build_butternet_38(input_shape, num_outputs):
        return ResnetBuilder.build(input_shape, num_outputs, fused_block, [5,2,1,1])
    @staticmethod
    def build_butternet_58(input_shape, num_outputs):
        return ResnetBuilder.build(input_shape, num_outputs, fused_block, [10,2,1,1])

def main():
    model = ResnetBuilder.build_resnet_18((3, 224, 224), 1000)
    model.compile(loss="categorical_crossentropy", optimizer="sgd")
    model.summary()


if __name__ == '__main__':
    main()

 
