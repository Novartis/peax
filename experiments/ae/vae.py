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

"""Example of VAE on MNIST dataset using MLP

The VAE has a modular design. The encoder, decoder and VAE
are 3 models that share weights. After training the VAE model,
the encoder can be used to  generate latent vectors.
The decoder can be used to generate MNIST digits by sampling the
latent vector from a Gaussian distribution with mean=0 and std=1.

# Reference

[1] Kingma, Diederik P., and Max Welling.
"Auto-encoding variational bayes."
https://arxiv.org/abs/1312.6114
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from keras.layers import Lambda, Input, Dense
from keras.models import Model
from keras.losses import mse, binary_crossentropy
from keras.utils import plot_model
from keras import backend as K


# reparameterization trick
# instead of sampling from Q(z|X), sample eps = N(0,I)
# z = z_mean + sqrt(var)*eps
def sampling(args):
    """Reparameterization trick by sampling from an isotropic unit Gaussian.

    # Arguments:
        args (tensor): mean and log of variance of Q(z|X)

    # Returns:
        z (tensor): sampled latent vector
    """

    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    # by default, random_normal has mean=0 and std=1.0
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon


def create(
    original_dim,
    intermediate_dim=128,
    latent_dim=2,
    use_mse=False,
    metrics=[],
    weights=None,
    save_weights=None,
    summary=False,
    plot=False,
):
    # VAE model = encoder + decoder
    # build encoder model
    inputs = Input(shape=(original_dim,), name="encoder_input")
    x = Dense(intermediate_dim, activation="relu")(inputs)
    z_mean = Dense(latent_dim, name="z_mean")(x)
    z_log_var = Dense(latent_dim, name="z_log_var")(x)

    # use reparameterization trick to push the sampling out as input
    # note that "output_shape" isn't necessary with the TensorFlow backend
    z = Lambda(sampling, output_shape=(latent_dim,), name="z")(
        [z_mean, z_log_var]
    )

    # instantiate encoder model
    encoder = Model(inputs, [z_mean, z_log_var, z], name="encoder")
    if summary:
        encoder.summary()

    if plot:
        plot_model(encoder, to_file="vae_mlp_encoder.png", show_shapes=True)

    # build decoder model
    latent_inputs = Input(shape=(latent_dim,), name="z_sampling")
    x = Dense(intermediate_dim, activation="relu")(latent_inputs)
    outputs = Dense(original_dim, activation="sigmoid")(x)

    # instantiate decoder model
    decoder = Model(latent_inputs, outputs, name="decoder")
    if summary:
        decoder.summary()

    if plot:
        plot_model(decoder, to_file="vae_mlp_decoder.png", show_shapes=True)

    # instantiate VAE model
    outputs = decoder(encoder(inputs)[2])
    vae = Model(inputs, outputs, name="vae_mlp")

    def vae_loss(metric=binary_crossentropy):

        def loss(inputs, outputs):
            reconstruction_loss = original_dim * metric(inputs, outputs)
            kl_loss = -0.5 * K.sum(
                1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1
            )
            return K.mean(reconstruction_loss + kl_loss)

        return loss

    # VAE loss = mse_loss or xent_loss + kl_loss
    loss_metric = mse if use_mse else binary_crossentropy

    vae.compile(optimizer="adam", loss=vae_loss(loss_metric), metrics=metrics)

    if summary:
        vae.summary()

    if plot:
        plot_model(encoder, to_file="vae_mlp.png", show_shapes=True)

    return encoder, decoder, vae
