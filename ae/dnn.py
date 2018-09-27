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

"""Dense Neural Network Models"""

from keras.layers import Input, Dense, Dropout
from keras.models import Model
from keras.regularizers import l1


def create(
    input_dim,
    optimizer="adam",
    loss="mse",
    units=[64, 32, 16],
    embedding=10,
    dropouts=[0.0, 0.0, 0.0],
    metrics=[],
    reg_lambda=0.0,
    summary=False,
):
    inputs = Input(shape=(input_dim,), name="decoded_input")

    encoded = inputs
    for i, u in enumerate(units):
        encoded = Dense(u, activation="relu", name="dense{}".format(i))(
            encoded
        )
        encoded = Dropout(dropouts[i], name="dropout{}".format(i))(encoded)

    encoded = Dense(
        embedding,
        activation="relu",
        name="embed",
        kernel_regularizer=l1(reg_lambda),
    )(encoded)

    decoded = encoded
    for i, u in enumerate(reversed(units)):
        k = i + len(units)
        decoded = Dense(u, activation="relu", name="dense{}".format(k))(
            decoded
        )
        decoded = Dropout(dropouts[i], name="dropout{}".format(k))(decoded)

    decoded = Dense(input_dim, activation="sigmoid", name="out")(decoded)

    autoencoder = Model(inputs, decoded)
    autoencoder.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    encoder = Model(inputs, encoded)

    encoded_input = Input(shape=(embedding,), name="encoded_input")
    decoded_input = encoded_input
    k = len(units) * 2 + 2
    for i in range(k, len(autoencoder.layers)):
        decoded_input = autoencoder.layers[i](decoded_input)
    decoder = Model(encoded_input, decoded_input)

    if summary:
        print(autoencoder.summary())
        print(encoder.summary())
        print(decoder.summary())

    return (encoder, decoder, autoencoder)
