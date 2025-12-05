import tensorflow as tf
import argparse
from datasetloader import IMDBLoader
from generator import Generator
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

# Checkpoint
parser.add_argument("--ckpt_path", default="./ckpt", type=str, help="Checkpoint Path for saved models")
parser.add_argument("--ckpt_frq", default = 10, type=int, help="Frequency in epochs for checkpoints to be made")
parser.add_argument("--ckpt_on", default = True, type=bool, help="Flag for if checkpoints should be made")



def main():
    # Load args
    config = parser.parse_args()
    print(config)
    
    # Load data
    dataset = IMDBLoader(config).load()

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
        # Pretrain Generator
        try:



        except Exception as e:
            print("Error when trying to run {}. Error message: {}".format(name, e))




        print(name)



















if __name__ == "__main__":
    main()