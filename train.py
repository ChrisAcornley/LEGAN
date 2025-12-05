import tensorflow as tf
import argparse
from datasetloader import IMDBLoader
from generator import Generator
from discriminator import LinearDiscriminator, EmbeddingDiscriminator

parser = argparse.ArgumentParser(description='Train and run LEGAN')

# Data
parser.add_argument('--dataset', default="imdb", type=str, help='Name of the dataset')
parser.add_argument('--dataset_path', default="./data", type=str, help='Path to dataset')
parser.add_argument('--ckpt_path', default='./ckpt', type=str, help='Checkpoint Path for saved models')


def main():
    # Load args
    args = parser.parse_args()
    print(args)
    
    # Load data
    data_loader = IMDBLoader(args).load()

    # Create list of Discriminator vs Generator
    gan_tuples = [
                    ("Linear", Generator(), LinearDiscriminator()),
                    ("EmbeddingLinear", Generator(), EmbeddingDiscriminator())
                ]

    # foreach item in list
        # Pretrain generator
        # pretrain discriminator
        # adversarial training
    for name, gen, disc in gan_tuples:
        print(name + " " + gen + " " + disc)



















if __name__ == "__main__":
    main()