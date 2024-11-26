import os
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import add,Input,Conv1D,Activation,Flatten,Dense,MaxPool1D,SeparableConv1D
from tensorflow.keras.layers import BatchNormalization,ReLU,Dropout


"""
    Basice Blocks are defined as follow
"""
def TempoConvBlock(x,filters,kernel_size,dilation_rate):
    r=Conv1D(filters,kernel_size,padding='same',dilation_rate=dilation_rate,activation='relu')(x) #第一卷积
    r=Conv1D(filters,kernel_size,padding='same',dilation_rate=dilation_rate)(r) #第二卷积
    if x.shape[-1]==filters:
        shortcut=x
    else:
        shortcut=Conv1D(filters,kernel_size,padding='same')(x)  #shortcut（捷径）
    o=add([r,shortcut])
    o=Activation('relu')(o)  #激活函数
    return o

def AdaptChannelAgg(x):
    x = Conv1D(1,1)(x)
    x = Flatten()(x)
    return x



"""
    Main Networks
"""
def FBNet_min( weights=None,
              input_shape=[1024,2],
              classes=24,
              **kwargs       ):
    
    # === Check Inputs ===
    if weights is not None and not (os.path.exists(weights)):
        raise ValueError('The `weights` argument should be either `None` (random initialization), or the path to the weights file to be loaded.')
    # === Hyper Parameters ===
    f = [16,16,16,16,16]
    k = [5,5,5,5,5]
    d = [1,4,16,64,128]
    # === Main Sturcture ===
    input = Input(input_shape,name='input')
    x = MaxPool1D(2)(input)
    # TCB
    for i in range(len(f)):
        x = TempoConvBlock(x,filters=f[i],kernel_size=k[i],dilation_rate=d[i])
    # ACA
    x = AdaptChannelAgg(x)
    x = Dense(classes,activation='softmax',name='out')(x)
    # === Build Model ===
    model = Model(inputs = input,outputs = x)
    # === Load Weights ===
    if weights is not None:
        model.load_weights(weights)
    return model


def FBNet_6k( weights=None,
              input_shape=[1024,2],
              classes=24,
              **kwargs       ):
    
    # === Check Inputs ===
    if weights is not None and not (os.path.exists(weights)):
        raise ValueError('The `weights` argument should be either `None` (random initialization), or the path to the weights file to be loaded.')
    # === Hyper Parameters ===
    f = [32,32,32,32,32]
    k = [5,5,5,5,5]
    d = [1,2,4,16,32]
    # === Main Sturcture ===
    input = Input(input_shape,name='input')
    x = MaxPool1D(2)(input)
    # TCB
    for i in range(len(f)):
        x = TempoConvBlock(x,filters=f[i],kernel_size=k[i],dilation_rate=d[i])
    # ACA
    x = AdaptChannelAgg(x)
    x = Dense(classes,activation='softmax',name='out')(x)
    # === Build Model ===
    model = Model(inputs = input,outputs = x)
    # === Load Weights ===
    if weights is not None:
        model.load_weights(weights)
    return model


def FBNet_7k(weights=None,
             input_shape=[1024,2],
             classes=24,
             **kwargs       ):
    
    # === Check Inputs ===
    if weights is not None and not (os.path.exists(weights)):
        raise ValueError('The `weights` argument should be either `None` (random initialization), or the path to the weights file to be loaded.')
    # === Hyper Parameters ===
    f = [32,32,32,32,32]
    k = [5,5,5,5,5]
    d = [1,4,16,64,128]
    # === Main Sturcture ===
    input = Input(input_shape,name='input')
    # TCB
    x = MaxPool1D(1)(input)
    for i in range(len(f)):
        x = TempoConvBlock(x,filters=f[i],kernel_size=k[i],dilation_rate=d[i])
    # ACA
    x = AdaptChannelAgg(x)
    x = Dense(classes,activation='softmax',name='out')(x)
    # === Build Model ===
    model = Model(inputs = input,outputs = x)
    # === Load Weights ===
    if weights is not None:
        model.load_weights(weights)
    return model


def FBNet_10k(weights=None,
              input_shape=[1024,2],
              classes=24,
              **kwargs      ):
    # === Check Inputs ===
    if weights is not None and not (os.path.exists(weights)):
        raise ValueError('The `weights` argument should be either `None` (random initialization), or the path to the weights file to be loaded.')
    # === Hyper Parameters ===
    f = [32,32,32,32,32,32,32,32]
    k = [5,5,5,5,5,5,5,5]
    d = [1,2,4,8,16,32,64,128]
    # === Main Sturcture ===
    input = Input(input_shape,name='input')
    # TCB
    x = MaxPool1D(1)(input)
    for i in range(len(f)):
        x = TempoConvBlock(x,filters=f[i],kernel_size=k[i],dilation_rate=d[i])
    # ACA
    x = AdaptChannelAgg(x)
    x = Dense(classes,activation='softmax',name='out')(x)
    # === Build Model ===
    model = Model(inputs = input,outputs = x)
    # === Load Weights ===
    if weights is not None:
        model.load_weights(weights)
    return model
