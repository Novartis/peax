"""
Copyright 2018 Novartis Institutes for BioMedical Research Inc.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

"""Neural net Models"""

from keras.layers import (
    AveragePooling1D,
    Input,
    Dense,
    Dropout,
    Conv1D,
    Conv2D,
    Conv2DTranspose,
    MaxPooling1D,
    UpSampling1D,
    LSTM,
    RepeatVector,
    Flatten,
    Reshape,
)
from keras.models import Model
from keras.regularizers import l1
from keras.utils import plot_model


def cnn3(
    input_dim,
    channels=1,
    optimizer="adam",
    loss="mse",
    cfilters=[120],
    ckernel_sizes=[9],
    dunits=[256, 64, 16],
    embedding=10,
    dropouts=[0.0, 0.0, 0.0],
    metrics=[],
    reg_lambda=0.0,
    summary=False,
    plot=False,
):
    inputs = Input(shape=(input_dim, 1), name="decoded_input")

    num_cfilter = len(cfilters)
    num_dunits = len(dunits)

    encoded = inputs
    for i, f in enumerate(cfilters):
        encoded = Conv1D(
            f,
            ckernel_sizes[i],
            strides=2,
            activation="relu",
            padding="same",
            name="conv{}".format(i),
        )(encoded)
        encoded = Dropout(dropouts[i], name="drop{}".format(i))(encoded)

    encoded = Flatten(name="flatten")(encoded)

    for i, u in enumerate(dunits):
        k = num_cfilter + i
        encoded = Dense(u, activation="relu", name="fc{}".format(k))(encoded)
        encoded = Dropout(dropouts[i], name="drop{}".format(k))(encoded)

    encoded = Dense(
        embedding, activation="relu", name="embed", kernel_regularizer=l1(reg_lambda)
    )(encoded)

    decoded = encoded
    for i, u in enumerate(reversed(dunits)):
        k = num_cfilter + num_dunits + i
        decoded = Dense(u, activation="relu", name="fc{}".format(k))(decoded)
        decoded = Dropout(dropouts[i], name="dropout{}".format(k))(decoded)

    decoded = Dense(
        int(input_dim / (2 ** len(cfilters))) * cfilters[-1],
        activation="relu",
        name="blowup",
    )(decoded)
    decoded = Reshape(
        (int(input_dim / (2 ** len(cfilters))), cfilters[-1]), name="unflatten"
    )(decoded)

    for i, f in enumerate(reversed(cfilters[:-1])):
        k = num_cfilter + (num_dunits * 2) + i
        j = num_cfilter - i - 2
        decoded = UpSampling1D(2, name="upsample{}".format(i))(decoded)
        decoded = Conv1D(
            f,
            ckernel_sizes[:-1][j],
            activation="relu",
            padding="same",
            name="deconv{}".format(i),
        )(decoded)
        decoded = Dropout(dropouts[i], name="drop{}".format(k))(decoded)

    decoded = UpSampling1D(2, name="upsample{}".format(len(cfilters) - 1))(decoded)
    decoded = Conv1D(
        channels, ckernel_sizes[0], activation="sigmoid", padding="same", name="out"
    )(decoded)

    autoencoder = Model(inputs, decoded)
    autoencoder.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    encoder = Model(inputs, encoded)

    encoded_input = Input(shape=(embedding,), name="encoded_input")
    decoded_input = encoded_input
    k = num_dunits * 2 + num_cfilter * 2 + 3
    for i in range(k, len(autoencoder.layers)):
        decoded_input = autoencoder.layers[i](decoded_input)
    decoder = Model(encoded_input, decoded_input)

    if summary:
        print(autoencoder.summary())
        print(encoder.summary())
        print(decoder.summary())

    if plot:
        plot_model(
            autoencoder, to_file="cnn3_ae.png", show_shapes=True, show_layer_names=True
        )
        plot_model(
            encoder, to_file="cnn3_de.png", show_shapes=True, show_layer_names=True
        )
        plot_model(
            encoder, to_file="cnn3_en.png", show_shapes=True, show_layer_names=True
        )

    return (encoder, decoder, autoencoder)


def cnn(
    input_shape=(120, 1),
    optimizer="adadelta",
    loss="binary_crossentropy",
    avg_pooling=False,
    filters=[64, 32, 16, 32],
    kernel_sizes=[5, 5, 3],
    metrics=[],
    summary=False,
    sample_weight_mode=None,
):
    # `% 8` because we have 3 pooling steps of 2, hence, 2^3 = 8
    if input_shape[0] % 8 == 0:
        pad3 = "same"
    else:
        pad3 = "valid"

    inputs = Input(shape=input_shape, name="decoded_input")

    pooling = MaxPooling1D if not avg_pooling else AveragePooling1D

    x = Conv1D(
        filters[0], kernel_sizes[0], activation="relu", padding="same", name="conv1"
    )(inputs)
    x = pooling(2, padding="same", name="pool1")(x)
    x = Conv1D(
        filters[1], kernel_sizes[1], activation="relu", padding="same", name="conv2"
    )(x)
    x = pooling(2, padding="same", name="pool2")(x)
    x = Conv1D(
        filters[2], kernel_sizes[2], activation="relu", padding=pad3, name="conv3"
    )(x)
    x = pooling(2, padding=pad3, name="pool3")(x)
    x = Flatten(name="flatten")(x)
    encoded = Dense(filters[3], activation="relu", name="embed")(x)

    x = Dense(filters[2] * int(input_shape[0] / 8), activation="relu", name="deembed")(
        encoded
    )
    x = Reshape((int(input_shape[0] / 8), filters[2]), name="unflatten")(x)
    # x = Conv1D(
    #     filters[2],
    #     kernel_sizes[2],
    #     activation='relu',
    #     padding='same',
    #     name='deconv0'
    # )(x)
    x = UpSampling1D(2, name="up1")(x)
    x = Conv1D(
        filters[1], kernel_sizes[2], activation="relu", padding="same", name="deconv1"
    )(x)
    x = UpSampling1D(2, name="up2")(x)
    x = Conv1D(
        filters[0], kernel_sizes[1], activation="relu", padding="same", name="deconv2"
    )(x)
    x = UpSampling1D(2, name="up3")(x)
    decoded = Conv1D(
        input_shape[1],
        kernel_sizes[0],
        activation="sigmoid",
        padding="same",
        name="deconv3",
    )(x)

    autoencoder = Model(inputs, decoded)
    autoencoder.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=metrics,
        sample_weight_mode=sample_weight_mode,
    )

    encoder = Model(inputs, encoded)

    encoded_input = Input(shape=(filters[3],), name="encoded_input")
    decoded_input = encoded_input
    for i in range(9, len(autoencoder.layers)):
        decoded_input = autoencoder.layers[i](decoded_input)
    decoder = Model(encoded_input, decoded_input)

    if summary:
        print(autoencoder.summary(), encoder.summary(), decoder.summary())

    return (encoder, decoder, autoencoder)


def cnn2(
    input_shape=(120, 1),
    optimizer="adadelta",
    loss="binary_crossentropy",
    filters=[64, 32, 16, 32],
    kernel_sizes=[5, 5, 3],
    metrics=[],
    summary=False,
    dr=False,
):
    # `% 8` because we have 3 pooling steps of 2, hence, 2^3 = 8
    if input_shape[0] % 8 == 0:
        pad3 = "same"
    else:
        pad3 = "valid"

    inputs = Input(shape=input_shape, name="decoded_input")

    x = Conv1D(
        filters[0],
        kernel_sizes[0],
        strides=2,
        activation="relu",
        padding="same",
        name="conv1",
    )(inputs)
    x = Conv1D(
        filters[1],
        kernel_sizes[1],
        strides=2,
        activation="relu",
        padding="same",
        name="conv2",
    )(x)
    x = Conv1D(
        filters[2],
        kernel_sizes[2],
        strides=2,
        activation="relu",
        padding=pad3,
        name="conv3",
    )(x)
    if dr:
        x = Flatten(name="flatten")(x)
        encoded = Dense(filters[3], activation="relu", name="embed")(x)
    else:
        encoded = Flatten(name="flatten")(x)

    if dr:
        x = Dense(
            filters[2] * int(input_shape[0] / 8), activation="relu", name="deembed"
        )(encoded)

        x = Reshape((int(input_shape[0] / 8), filters[2]), name="unflatten")(x)
    else:
        x = Reshape((int(input_shape[0] / 8), filters[2]), name="unflatten")(encoded)

    x = UpSampling1D(2, name="up1")(x)
    x = Conv1D(
        filters[1], kernel_sizes[2], activation="relu", padding="same", name="deconv1"
    )(x)
    x = UpSampling1D(2, name="up2")(x)
    x = Conv1D(
        filters[0], kernel_sizes[1], activation="relu", padding="same", name="deconv2"
    )(x)
    x = UpSampling1D(2, name="up3")(x)
    decoded = Conv1D(
        input_shape[1],
        kernel_sizes[0],
        activation="sigmoid",
        padding="same",
        name="deconv3",
    )(x)

    autoencoder = Model(inputs, decoded)
    autoencoder.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    encoder = Model(inputs, encoded)

    if dr:
        encoded_input = Input(shape=(filters[3],), name="encoded_input")
    else:
        encoded_input = Input(
            shape=(filters[2] * int(input_shape[0] / 8),), name="encoded_input"
        )
    decoded_input = encoded_input
    mid = 6 if dr else 5
    for i in range(mid, len(autoencoder.layers)):
        decoded_input = autoencoder.layers[i](decoded_input)
    decoder = Model(encoded_input, decoded_input)

    if summary:
        print(autoencoder.summary(), encoder.summary(), decoder.summary())

    return (encoder, decoder, autoencoder)


def cae2d(
    input_shape=(120, 50, 1),
    optimizer="adam",
    loss="mse",
    filters=[32, 64, 128],
    kernel_sizes=[5, 5, 3],
    dunits=[512, 256, 128],
    embedding=10,
    metrics=[],
    summary=False,
    dr=False,
):
    # `% 8` because we have 3 pooling steps of 2, hence, 2^3 = 8
    if input_shape[0] % 8 == 0:
        pad3 = "same"
    else:
        pad3 = "valid"

    inputs = Input(shape=input_shape, name="decoded_input")

    x = Conv2D(
        filters[0],
        kernel_sizes[0],
        strides=2,
        activation="relu",
        padding="same",
        name="conv1",
    )(inputs)
    x = Conv2D(
        filters[1],
        kernel_sizes[1],
        strides=2,
        activation="relu",
        padding="same",
        name="conv2",
    )(x)
    x = Conv2D(
        filters[2],
        kernel_sizes[2],
        strides=2,
        activation="relu",
        padding=pad3,
        name="conv3",
    )(x)
    x = Flatten(name="flatten")(x)
    x = Dense(dunits[0], activation="relu", name="fc1")(x)
    x = Dense(dunits[1], activation="relu", name="fc2")(x)
    x = Dense(dunits[2], activation="relu", name="fc3")(x)

    encoded = Dense(embedding, activation="relu", name="embed")(x)

    x = Dense(dunits[2], activation="relu", name="dfc1")(encoded)
    x = Dense(dunits[1], activation="relu", name="dfc2")(x)
    x = Dense(dunits[0], activation="relu", name="dfc3")(x)
    x = Dense(
        int(input_shape[0] / 8) * int(input_shape[1] / 8) * filters[2],
        activation="relu",
        name="blowup",
    )(x)
    x = Reshape(
        (int(input_shape[0] / 8), int(input_shape[1] / 8), filters[2]), name="unflatten"
    )(x)
    x = Conv2DTranspose(
        filters[1],
        kernel_sizes[2],
        strides=2,
        activation="relu",
        padding=pad3,
        name="deconv1",
    )(x)
    x = Conv2DTranspose(
        filters[0],
        kernel_sizes[1],
        strides=2,
        activation="relu",
        padding="same",
        name="deconv2",
    )(x)
    decoded = Conv2DTranspose(
        input_shape[2],
        kernel_sizes[0],
        strides=2,
        activation="sigmoid",
        padding="same",
        name="deconv3",
    )(x)

    autoencoder = Model(inputs, decoded)
    autoencoder.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    encoder = Model(inputs, encoded)

    encoded_input = Input(shape=(embedding,), name="encoded_input")
    decoded_input = encoded_input

    for i in range(9, len(autoencoder.layers)):
        decoded_input = autoencoder.layers[i](decoded_input)
    decoder = Model(encoded_input, decoded_input)

    if summary:
        print(autoencoder.summary(), encoder.summary(), decoder.summary())

    return (encoder, decoder, autoencoder)


def lstm(latent_dim):
    inputs = Input(shape=train.shape)
    encoded = LSTM(128)(inputs)

    decoded = RepeatVector(train.shape[0])(encoded)
    decoded = LSTM(train.shape[1], return_sequences=True)(decoded)

    autoencoder = Model(inputs, decoded)
    encoder = Model(inputs, encoded)

    autoencoder.compile(optimizer="rmsprop", loss="binary_crossentropy")

    return (encoder, decoder, autoencoder)
