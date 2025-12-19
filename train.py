import tensorflow as tf
import numpy as np
import time
import argparse
import os
from datasetloader import IMDBLoader
from generator import Generator, GeneratorSimplified
from discriminator import LinearDiscriminator, EmbeddingDiscriminator

# Setup default arguments
parser = argparse.ArgumentParser(description="Train and run LEGAN")

# Data
parser.add_argument("--dataset", default="imdb", type=str, help="Name of the dataset")
parser.add_argument("--dataset_path", default="./data", type=str, help="Path to dataset")
parser.add_argument("--buffer_size", default=10000, type=int, help="Size of buffer for dataset")
parser.add_argument("--batch_size", default=64, type=int, help="Size of dataset batches")
parser.add_argument("--vocab_size", default=1000, type=int, help="Vocabulary size of dataset")
parser.add_argument("--print_loading_time", default=True, type=bool, help="Flag to print time taken to load dataset")

# Generator
parser.add_argument("--gen_embedding_dim", default=256, type=int, help="Dimension of the Embedding Layer")
parser.add_argument("--gen_rnn_units", default=1024, type=int, help="Number of RNN units in generator")

# Pretraining
parser.add_argument("--pretrain_gen_epochs", default=100, type=int, help="Number of pretrain epochs for Generator")
parser.add_argument("--pretrain_ckpt_max_count", default=1, type=int, help="Maximum number of checkpoints to keep")

# Post-Training
parser.add_argument("--posttrain_gen_epochs", default=100, type=int, help="Number of posttrain epochs for Generator")

# Checkpoint
parser.add_argument("--ckpt_path", default="./ckpt", type=str, help="Checkpoint Path for saved models")
parser.add_argument("--ckpt_frq", default = 10, type=int, help="Frequency in epochs for checkpoints to be made")
parser.add_argument("--ckpt_on", default = True, type=bool, help="Flag for if checkpoints should be made")

def split_input_for_sequence(sequence):
    input_seq = sequence[:-1]
    return input_seq

def split_input_for_target(sequence):
    target_text = sequence[1:]
    return target_text

def finished_training_file(path):
    with open(path, "w") as f:
        f.write("This file denotes that the pre-training is complete. Training will not be attempted.")
        f.write("Delete this file to load from current checkpoint and continue training")
        f.close()

def generate_fake_samples(generator, input, encoder, training=False):
    encoded_input = encoder(input)
    input_seq = [split_input_for_sequence(x) for x in encoded_input]
    random_noise = tf.random.normal([encoded_input.shape[0], encoded_input.shape[1] - 1, 16], 0.5, 0.5)
    pred = generator({"dataset_inputs": np.array(input_seq), "random_noise":random_noise}, training=training)
    return pred

def generator_loss(generated_output):
    cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    return cross_entropy(tf.ones_like(generated_output), generated_output)

def discriminator_loss(real_output, generated_output):
    cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    generated_loss = cross_entropy(tf.zeros_like(generated_output), generated_output)
    total_loss = real_loss + generated_loss
    return total_loss

def pretrain_generator(config, data_loader, generator, name):
    ckpt = tf.train.Checkpoint(generator)
    ckpt_path = os.path.join(config.ckpt_path , name + "_gen_pretrain")
    full_train_file = os.path.join(ckpt_path, "training_complete.txt")
    manager = tf.train.CheckpointManager(ckpt, ckpt_path, max_to_keep=config.pretrain_ckpt_max_count)

    ckpt.restore(manager.latest_checkpoint)
    if manager.latest_checkpoint:
        print("Restored from {}".format(manager.latest_checkpoint))

        if os.path.exists(full_train_file):
            return
    else:
        print("Initializing from scratch.")

    loss_func = tf.losses.SparseCategoricalCrossentropy(from_logits=True)
    optimiser = tf.optimizers.Adam()

    # Generator Pre-Train
    for epoch in range(config.pretrain_gen_epochs):
        starttime = time.perf_counter()
        print("Starting Epoch {}...".format(epoch + 1))
        loss_avg = []
        for batch in data_loader.train_dataset:
            input = data_loader.masked_encoder(batch[0])
            input_seq = [split_input_for_sequence(x) for x in input]
            true_seq = [split_input_for_target(x) for x in input]
            random_noise = tf.random.normal([input.shape[0], input.shape[1] - 1, 16], 0.5, 0.5)
            with tf.GradientTape() as tape:
                pred = generator({"dataset_inputs": np.array(input_seq), "random_noise":random_noise}, training=True)
                loss = loss_func(np.array(true_seq), pred)
                loss_avg.append(loss.numpy())
            gradients = tape.gradient(loss, generator.trainable_variables)
            optimiser.apply_gradients(zip(gradients, generator.trainable_variables))    
        endtime = time.perf_counter()
        print("Loss for epoch {}: {:.3f}".format(epoch+1, sum(loss_avg) / len(loss_avg)))
        print('Epoch {} took {:.2f}s'.format(epoch+1, float(endtime - starttime)))
        if (epoch+1) % config.ckpt_frq == 0:
            manager.save()

    # Create file that denotes trianing is complete. If this file is found in a checkpoint folder, training will not happen (latest checkpoint is still loaded)
    finished_training_file(full_train_file)

def pretrain_discriminator(config, data_loader, discriminator, generator, name):
    ckpt = tf.train.Checkpoint(discriminator)
    ckpt_path = os.path.join(config.ckpt_path , name + "_disc_pretrain")
    full_train_file = os.path.join(ckpt_path, "training_complete.txt")
    manager = tf.train.CheckpointManager(ckpt, ckpt_path, max_to_keep=config.pretrain_ckpt_max_count)

    ckpt.restore(manager.latest_checkpoint)
    if manager.latest_checkpoint:
        print("Restored from {}".format(manager.latest_checkpoint))

        if os.path.exists(full_train_file):
            return
    else:
        print("Initializing from scratch.")

    optimiser = tf.optimizers.Adam()

    # Discriminator Pre-Train
    for epoch in range(config.pretrain_gen_epochs):
        starttime = time.perf_counter()
        print("Starting Epoch {}...".format(epoch + 1))
        loss_avg = []
        for batch in data_loader.train_dataset:
            input = batch[0]
            # generate false samples with pre-trained generator
            gen_samples = generate_fake_samples(generator, input, data_loader.masked_encoder)
            
            with tf.GradientTape() as tape:
                pred_true = discriminator(input, training=True)
                pred_false = discriminator(gen_samples, use_max_layer=True, training=True)
                loss = discriminator_loss(pred_true, pred_false)
                loss_avg.append(loss.numpy())
            gradients = tape.gradient(loss, discriminator.trainable_variables)
            optimiser.apply_gradients(zip(gradients, discriminator.trainable_variables))    
        endtime = time.perf_counter()
        print("Loss for epoch {}: {:.3f}".format(epoch+1, sum(loss_avg) / len(loss_avg)))
        print('Epoch {} took {:.2f}s'.format(epoch+1, float(endtime - starttime)))
        if (epoch+1) % config.ckpt_frq == 0:
            manager.save()

    # Create file that denotes trianing is complete. If this file is found in a checkpoint folder, training will not happen (latest checkpoint is still loaded)
    finished_training_file(full_train_file)

def adversarial_training(config, data_loader, discriminator, generator, name):
    ckpt = tf.train.Checkpoint(discriminator=discriminator, generator=generator)
    ckpt_path = os.path.join(config.ckpt_path , name + "adversarial_training")
    full_train_file = os.path.join(ckpt_path, "training_complete.txt")
    manager = tf.train.CheckpointManager(ckpt, ckpt_path, max_to_keep=config.pretrain_ckpt_max_count)

    ckpt.restore(manager.latest_checkpoint)
    if manager.latest_checkpoint:
        print("Restored from {}".format(manager.latest_checkpoint))
    else:
        print("Initializing from scratch.")

    gen_optimiser = tf.optimizers.Adam()
    disc_optimiser = tf.optimizers.Adam()

    # Adversarial Training
    for epoch in range(config.posttrain_gen_epochs):
        starttime = time.perf_counter()
        print("Starting Epoch {}...".format(epoch + 1))
        gen_loss_avg = []
        disc_loss_avg = []
        for batch in data_loader.train_dataset:
            input = batch[0]
            with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
                # generate false samples with pre-trained generator
                gen_samples = generate_fake_samples(generator, input, data_loader.masked_encoder,training=True)
                pred_true = discriminator(input, training=True)
                pred_false = discriminator(gen_samples, use_max_layer=True, training=True)
                gen_loss = generator_loss(pred_false)
                disc_loss = discriminator_loss(pred_true, pred_false)
                gen_loss_avg.append(gen_loss.numpy())
                disc_loss_avg.append(disc_loss.numpy())
            gen_gradients = gen_tape.gradient(gen_loss, generator.trainable_variables)
            disc_gradients = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
            gen_optimiser.apply_gradients(zip(gen_gradients, generator.trainable_variables))   
            disc_optimiser.apply_gradients(zip(disc_gradients, discriminator.trainable_variables))    
        endtime = time.perf_counter()
        print("Loss for Generator epoch {}: {:.3f}".format(epoch+1, sum(gen_loss_avg) / len(gen_loss_avg)))
        print("Loss for Discriminator epoch {}: {:.3f}".format(epoch+1, sum(disc_loss_avg) / len(disc_loss_avg)))
        print('Epoch {} took {:.2f}s'.format(epoch+1, float(endtime - starttime)))
        if (epoch+1) % config.ckpt_frq == 0:
            manager.save()

    # Create file that denotes trianing is complete. If this file is found in a checkpoint folder, training will not happen (latest checkpoint is still loaded)
    finished_training_file(full_train_file)


def main():
    # Load args
    config = parser.parse_args()
    
    # Load data
    loader = IMDBLoader(config).load()

    # Create list of Discriminator vs Generator
    gan_tuples = [
                    ("Linear", Generator(config.vocab_size, config.gen_embedding_dim, config.gen_rnn_units), LinearDiscriminator(loader.encoder)),
                    ("EmbeddingLinear", Generator(config.vocab_size, config.gen_embedding_dim, config.gen_rnn_units), EmbeddingDiscriminator(loader.encoder, config.vocab_size)),
                    ("LinearSimple", GeneratorSimplified(config.vocab_size, config.gen_embedding_dim, config.gen_rnn_units), LinearDiscriminator(loader.encoder)),
                    ("EmbeddingLinearSimple", GeneratorSimplified(config.vocab_size, config.gen_embedding_dim, config.gen_rnn_units), EmbeddingDiscriminator(loader.encoder, config.vocab_size))
                ]

    for name, gen, disc in gan_tuples:
        # Pretrain Generator
        print("Pretraining Generator for {}".format(name))
        try:
            pretrain_generator(config, loader, gen, name)
        except Exception as e:
            print("Error when trying to run generator pre_train for model: {}. Error message: {}".format(name, e))

        # pretrain discriminator
        print("Pretraining Discriminator for {}".format(name))
        try:
            pretrain_discriminator(config, loader, disc, gen, name)
        except Exception as e:
            print("Error when trying to run discriminator pre_train for model: {}. Error message: {}".format(name, e))

        # adversarial training
        # pretrain discriminator
        print("Pretraining Discriminator for {}".format(name))
        try:
            adversarial_training(config, loader, disc, gen, name)
        except Exception as e:
            print("Error when trying to run discriminator pre_train for model: {}. Error message: {}".format(name, e))

if __name__ == "__main__":
    main()