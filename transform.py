import argparse
from data.data_collection import extract_concode_like_features

parser = argparse.ArgumentParser(description='transform.py')

parser.add_argument('-root_dir', type=str,
                    help="""Root directory containing all the .protobuf files. 
                    The directories are all recursively searched""")

parser.add_argument('-seed', type=int,
                    help="""Seed for splitting""", default=42)


def main():
    opt = parser.parse_args()

    extract_concode_like_features(opt.root_dir, opt.seed)


if __name__ == "__main__":
    main()
