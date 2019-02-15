from tensorflow import keras


def conv(x, filters, kernel_size, padding='same', activation='relu'):
    layer = keras.layers.Conv2D(filters=filters, kernel_size=(kernel_size,kernel_size), padding=padding, activation=activation)(x)
    layer = keras.layers.BatchNormalization()(layer)
    return layer

def maxpool(x):
    return keras.layers.MaxPool2D()(x)

def concatenate(layers):
    return keras.layers.Concatenate(axis=-1)(layers)

def msb(x, filters, kernel_sizes, padding='same', activation='relu'):
    last = []

    for kernel_size in kernel_sizes:
        last.append(conv(x, filters, kernel_size))
    return concatenate(last)

def model() -> keras.Model:
    inputs = keras.layers.Input((None, None, 3))

    m = conv(inputs, 64, 9)
    
    m = msb(m, 16, (9, 7, 5, 3))
    m = maxpool(m)
    
    m = msb(m, 32, (9, 7, 5, 3))
    m = msb(m, 32, (9, 7, 5, 3))
    m = maxpool(m)

    m = msb(m, 64, (7, 5, 3))
    m = msb(m, 64, (7, 5, 3))

    # m = mlp(m, 1000, 1)
    last = conv(m, 1, 1)
    m = keras.Model(inputs=inputs, outputs=last)
    return m
