#!/usr/bin/python3

# closely based on my preferred VAE implementation for my own research, which is in turn
# loosely based on https://www.tensorflow.org/tutorials/generative/cvae

# loss functions and encoder/decoder architecture borrowed from Ha's repository at
# https://github.com/hardmaru/WorldModelsExperiments/blob/master/carracing/vae/vae.py
# in order to reproduce the Schmidhuber and Ha results.

import tensorflow as tf
import numpy as np

# can use keras.layers.Conv2D just fine

def log_normal_pdf(x, mean, std_dev):
    var = std_dev*std_dev
    return -.5 * ((x - mean) ** 2. / var + tf.math.log(var*2*np.pi))

class VAE:
    def __init__(self, inference_conv_layers, generator_deconv_layers, image_shape, latent_dim):
        self.image_shape = image_shape
        self.latent_dim = latent_dim

        self.inference_layers = [tf.keras.layers.InputLayer(input_shape=image_shape)] + \
            inference_conv_layers + [tf.keras.layers.Flatten(), tf.keras.layers.Dense(2*latent_dim)]
        self.inference_net = tf.keras.Sequential(self.inference_layers)
        self.generator_layers = [tf.keras.layers.InputLayer(input_shape=(latent_dim,))] + generator_deconv_layers
        self.generator_net = tf.keras.Sequential(self.generator_layers)

        self.optimizer = tf.keras.optimizers.Adam(0.0005)

    @tf.function
    def get_mean(self, sample):
        if sample.dtype == tf.uint8:
            sample = tf.cast(sample, 'float32')/255
        latent_info = self.inference_net(sample)
        mean, log_var = tf.split(latent_info, [self.latent_dim, self.latent_dim], -1)
        return mean

    @tf.function
    def sample_latent(self, sample):
        if sample.dtype == tf.uint8:
            sample = tf.cast(sample, 'float32')/255
        if len(sample.shape) == 3:
            unbatched = True
            sample = tf.expand_dims(sample, axis=0)
        else:
            unbatched = False

        latent_info = self.inference_net(sample)
        mean, log_var = tf.split(latent_info, [self.latent_dim, self.latent_dim], -1)

        std_dev = tf.exp(log_var / 2.0)

        sampled_t = tf.random.normal(shape=std_dev.shape)
        sampled_z = mean + std_dev*sampled_t

        if unbatched:
            return sampled_z[0]
        else:
            return sampled_z

    @tf.function
    def train_vae(self, sample):
        ''' Sampling-based approach that gets means and standard deviations
            from inference(sample), then per batch item uses a single std normal
            sample to pick a vector for the generator to use.
            (This may result in unexpected gradients for the std deviation term;
            I need to think more about that.
            Should we sample more than one random vector per sample?) '''
        if sample.dtype == tf.uint8:
            sample = tf.cast(sample, 'float32')/255
        latent_info = self.inference_net(sample)
        mean, log_var = tf.split(latent_info, [self.latent_dim, self.latent_dim], -1)

        std_dev = tf.exp(log_var / 2.0)

        sampled_t = tf.random.normal(shape=std_dev.shape)
        sampled_z = mean + std_dev*sampled_t

        new = self.generator_net(sampled_z)

        # lifted from David Ha's github
        # Ha replaces the traditional probability-distribution loss with two things:
        # the error in reconstructing the sample
        reconstruction_loss = tf.reduce_sum((sample-new)**2) / sample.shape[0]
        # and the KL-divergence of the z-distribution from the standard normal distribution
        kl_loss = - 0.5 * tf.reduce_sum(1 + log_var - mean**2 - tf.exp(log_var)) / sample.shape[0]

        loss = reconstruction_loss + kl_loss

        cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(logits=new, labels=sample) # is this what we want?
        logpsample_z = -tf.reduce_sum(cross_ent, axis=[1, 2, 3])
        # the unconditioned latent distribution is assumed to be standard normal:
        logpz = tf.reduce_sum(log_normal_pdf(sampled_z, 0., 1.), axis=[1])
        logqz_sample = tf.reduce_sum(log_normal_pdf(sampled_z, mean, std_dev), axis=[1])
        loss = -tf.reduce_mean(logpsample_z + logpz - logqz_sample)

        vars = self.inference_net.trainable_variables + self.generator_net.trainable_variables
        grads = tf.gradients(loss, vars)
        self.optimizer.apply_gradients(zip(grads, vars))

        return loss

def random_sample(vae):
    import matplotlib.pyplot as plt

    dims = [0,1,2,3,4,5]

    for trial in range(4):
        latent = tf.random.normal((vae.latent_dim,), stddev=trial)
        for i, dim in enumerate(dims):
            for d in range(-2, 3):
                delta = tf.SparseTensor([[dim]], [d * 1.5], (vae.latent_dim,))
                generated = vae.generator_net(tf.expand_dims(tf.sparse.add(latent, delta), 0))
                generated = tf.nn.sigmoid(generated)
                plt.subplot(5,len(dims),(d+2)*len(dims)+i+1)
                plt.imshow(generated[0,:,:])
        plt.show()


def sample(vae, training_dataset):
    import matplotlib.pyplot as plt

    #random_sample(vae)
    n_samples = 4
    n_boards = 8
    for i, sample in enumerate(training_dataset):
        if i == n_samples*n_boards: break
        n = i % n_samples
        sample = sample[:2]
        latent = vae.sample_latent(sample)
        tf.print(latent)
        generated = vae.generator_net(tf.concat([latent, tf.reduce_mean(latent, axis=0, keepdims=True)], axis=0))
        generated = tf.nn.sigmoid(generated)
        plt.subplot(n_samples,5,1+5*n)
        plt.imshow(sample[0,:,:])
        plt.subplot(n_samples,5,2+5*n)
        plt.imshow(generated[0,:,:])
        plt.subplot(n_samples,5,3+5*n)
        plt.imshow(generated[2,:,:])
        plt.subplot(n_samples,5,4+5*n)
        plt.imshow(generated[1,:,:])
        plt.subplot(n_samples,5,5+5*n)
        plt.imshow(sample[1,:,:])
        if (i + 1) % n_samples == 0: plt.show()

def train(vae, training_dataset, distance_sampler):
    import matplotlib.pyplot as plt
    losses = []
    distances = []
    first_mean = vae.get_mean(distance_sampler)
    for epoch in range(8):
        total = 0
        count = 0
        for sample in training_dataset:
            loss = vae.train_vae(sample)
            losses.append(loss.numpy())
            current_mean = vae.get_mean(distance_sampler)
            cos = -tf.reduce_mean(tf.keras.losses.cosine_similarity(first_mean, current_mean))
            distances.append(cos.numpy())
            total += loss
            count += 1
            if count % 50 == 0:
                print(count, loss)
        print('Epoch', epoch, total/count)
    plt.plot(losses, label='VAE Loss')
    plt.xlabel('Training Steps')
    plt.ylabel('VAE Loss')
    plt.show()
    plt.plot(distances, label='Sample Embedding Similarity')
    plt.xlabel('Training Steps')
    plt.ylabel('Sample Embedding Similarity')
    plt.show()

# pulled from Schmidhuber & Ha "World Models"
def make_default_vae(image_shape=(96, 96, 3), latent_dim=32):
    sample_inference_conv_layers = [
        tf.keras.layers.Conv2D(filters=32, kernel_size=2, strides=(2, 2), activation='relu'),
        tf.keras.layers.Conv2D(filters=64, kernel_size=2, strides=(2, 2), activation='relu'),
        tf.keras.layers.Conv2D(filters=128, kernel_size=2, strides=(2, 2), activation='relu'),
        tf.keras.layers.Conv2D(filters=256, kernel_size=2, strides=(2, 2), activation='relu')]

    sample_generator_deconv_layers = [tf.keras.layers.Dense(units=1*1*1024, activation=tf.nn.relu),
        tf.keras.layers.Reshape(target_shape=(1, 1, 1024)),
        tf.keras.layers.Conv2DTranspose(filters=128, kernel_size=5, strides=(2, 2), padding="VALID", activation='relu'),
        tf.keras.layers.Conv2DTranspose(filters=64, kernel_size=5, strides=(2, 2), padding="VALID", activation='relu'),
        tf.keras.layers.Conv2DTranspose(filters=32, kernel_size=7, strides=(2, 2), padding="VALID", activation='relu'),
        tf.keras.layers.Conv2DTranspose(filters=3, kernel_size=6, strides=(3, 3), padding="VALID")]

    return VAE(sample_inference_conv_layers, sample_generator_deconv_layers, image_shape, latent_dim)

if __name__ == '__main__':
    states = np.load('out/half_cheetah_images.npy')
    np.random.seed(15)
    np.random.shuffle(states)
    states = np.uint8(states)
    # print(states.shape)
    # print(states[0])
    # states = state[:50000]
    # import matplotlib.pyplot as plt
    # plt.imshow(states[0])
    # plt.show()
    # plt.imshow(states[15])
    # plt.show()

    minibatch = 256

    training_dataset = tf.data.Dataset.from_tensor_slices(states).batch(minibatch)

    vae = make_default_vae(latent_dim=8)
    train(vae, training_dataset, states[:10])
    # sample(vae, training_dataset)

    vae.inference_net.save_weights('out/cheetah_inf.tfdat')
    vae.generator_net.save_weights('out/cheetah_gen.tfdat')