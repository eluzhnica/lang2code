import random
import argparse
import torch
from torch import cuda
import torch.nn as nn
from preprocess import Vocab, CDDataset
from S2SModel import S2SModel
from Trainer import Trainer
import os
import numpy


def main():
    parser = argparse.ArgumentParser(description='train.py')

    parser.add_argument("-num_gpus", type=int)
    parser.add_argument("-max_camel", type=int, default=20)
    parser.add_argument('-src_word_vec_size', type=int, default=500,
                        help='Src word embedding sizes')
    parser.add_argument('-tgt_word_vec_size', type=int, default=500,
                        help='Tgt word embedding sizes')
    parser.add_argument('-enc_layers', type=int, default=1,
                        help='Number of layers in the encoder')
    parser.add_argument('-dec_layers', type=int, default=1,
                        help='Number of layers in the decoder')
    parser.add_argument('-rnn_size', type=int, default=500,
                        help='Size of LSTM hidden states')
    parser.add_argument('-decoder_rnn_size', type=int, default=1024,
                        help='Size of LSTM hidden states')
    parser.add_argument('-brnn', action="store_true",
                        help="Use a bidirectional RNN in the encoder")
    parser.add_argument('-data', required=True,
                        help="""Path prefix to the ".train.pt" and
                      ".valid.pt" file path from preprocess.py""")
    parser.add_argument('-save_model', default='model',
                        help="""Model filename (the model will be saved as
                      <save_model>_epochN_PPL.pt where PPL is the
                      validation perplexity""")
    # GPU
    parser.add_argument('-seed', type=int, default=-1,
                        help="""Random seed used for the experiments
                      reproducibility.""")

    # Optimization options
    parser.add_argument('-batch_size', type=int, default=1,
                        help='Maximum batch size')
    parser.add_argument('-epochs', type=int, default=30,
                        help='Number of training epochs')
    parser.add_argument('-dropout', type=float, default=0.3,
                        help="Dropout probability; applied in LSTM stacks.")
    # learning rate
    parser.add_argument('-learning_rate', type=float, default=1.0,
                        help="""Starting learning rate. If adagrad/adadelta/adam
                      is used, then this is the global learning rate.
                      Recommended settings: sgd = 1, adagrad = 0.1,
                      adadelta = 1, adam = 0.001""")
    parser.add_argument('-learning_rate_decay', type=float, default=0.8,
                        help="""If update_learning_rate, decay learning rate by
                      this much if (i) perplexity does not decrease on the
                      validation set or (ii) epoch has gone past
                      start_decay_at""")

    opt = parser.parse_args()

    torch.backends.cudnn.deterministic = True
    # torch.cuda.set_device(0)
    torch.manual_seed(opt.seed)
    random.seed(opt.seed)
    torch.cuda.manual_seed(opt.seed)
    numpy.random.seed(opt.seed)

    try:
        os.makedirs(opt.save_model)
    except:
        pass

    print('Loading train set\n')
    train = torch.load(opt.data + '.train.pt')
    print('Loaded Train :')
    valid = torch.load(opt.data + '.valid.pt')

    print('Loaded datasets:')

    vocabs = torch.load(opt.data + '.vocab.pt')

    print(opt)
    model = S2SModel(opt, vocabs)

    trainer = Trainer(model)
    trainer.run_train_batched(train, valid, vocabs)


if __name__ == "__main__":
    main()
