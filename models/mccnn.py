from tensorflow import keras


def conv(x, filters, kernel_size, padding='same', activation='relu'):
    layer = keras.layers.Conv2D(filters=filters, kernel_size=(kernel_size,kernel_size), padding=padding, activation=activation)(x)
    layer = keras.layers.BatchNormalization()(layer)
    return layer

def maxpool(x):
    return keras.layers.MaxPool2D()(x)

def concatenate(layers):
    return keras.layers.Concatenate(axis=-1)(layers)

def model() -> keras.Model:
    inputs = keras.layers.Input((None, None, 3))
    
    x = conv(inputs, 16, 9)
    x = maxpool(x)
    x = conv(x, 32, 7)
    x = maxpool(x)
    x = conv(x, 16, 7)
    x = conv(x, 8, 7)

    y = conv(inputs, 20, 7)
    y = maxpool(y)
    y = conv(y, 40, 5)
    y = maxpool(y)
    y = conv(y, 20, 5)
    y = conv(y, 10, 5)

    z = conv(inputs, 24, 5)
    z = maxpool(z)
    z = conv(z, 48, 3)
    z = maxpool(z)
    z = conv(z, 24, 3)
    z = conv(z, 12, 3)

    conc = concatenate([x, y, z])

    last = conv(conc, 1, 1)
    m = keras.Model(inputs=inputs, outputs=last)
    return m
