import argparse
import pandas as pd

from generate.gen_from_cfg import generate
from generate.substitute_nums import replace_tokens_all


def parse_args():
    parser = argparse.ArgumentParser(
            description="Training set generator")

    parser.add_argument("--cfg_dir_root", type=str,
                        default="./cfgs/",
                        help="path to config folder")
    parser.add_argument("--grammar_name", type=str,
                        default="cfg-french.txt",
                        help="name of grammar")
    parser.add_argument("--data_dir_root", type=str,
                        default="./data/train/",
                        help="path to data")
    parser.add_argument("--data_name", type=str,
                        default="fr.csv",
                        help="name of dataset")
    parser.add_argument("--arena_dim", type=int,
                        default=100,
                        help="dimension of game board")
    parser.add_argument("--lang", type=str,
                        default="fr",
                        help="choices are (en, fr)")
    parser.add_argument("--n_samples_per_string", type=int,
                        default=20000,
                        help="number of samples to generate per cfg string")

    return parser.parse_args()


def print_args(args):
    """
        Print arguments (only show the relevant arguments)
    """
    print("\nParameters:")
    for attr, value in sorted(args.__dict__.items()):
        print("\t{}={}".format(attr.upper(), value))


def generate_data(args):
    """
    Generates training set.
    :param args
    :return:
    """
    cfg_strings = generate(args.cfg_dir_root + args.grammar_name)
    train_set = replace_tokens_all(cfg_strings, args.arena_dim, args.n_samples_per_string, args.lang)
    return train_set


def main():
    args = parse_args()
    print_args(args)

    # Generate data
    data = pd.DataFrame(generate_data(args), columns=["string", "x", "y"]).sample(frac=1)
    print(data)

    # Save data
    data.to_csv(args.data_dir_root + args.data_name)


if __name__ == "__main__":
    main()