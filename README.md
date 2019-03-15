# lang2code
Re-implementation of the paper Mapping Language to Code in Programmatic Context

This is not the official implementation of the paper.


## Install requirements

Create a virtual environment with python3 (optional).

`python3 -m venv langcode`

Activate the virtual environment

`source langcode/bin/activate`

Install the requirements

`pip install -r requirements.txt`

You are ready to go.

## Running

If you have a directory containing some .proto files, you can extract all the data recursively from the root dir.
Alternatively, if you want to use the CONCODE dataset you can skip the next command.

`python transform.py -root_dir=path/to/root/of/proto/files`

This will create train.json, test.json, valid.json files in the main project directory.

Once the json files are created (or downloaded from CONCODE), run:

`python build.py -train_file train.json -valid_file valid.json -test_file test.json -output_folder data`

This will create the corresponding `.dataset` files in the `data` directory. You can inspect those if you wish.

Next, we need to preprocess data. Run:

`python preprocess.py -train data/train.dataset -valid data/valid.dataset -save_data data/processed`

Finally, to train, run:

`python train.py -dropout 0.5 -data data/processed -save_model data/processed/ -epochs 30 -learning_rate 0.001 -seed 1123 -enc_layers 2 -dec_layers 2 -batch_size 20 -src_word_vec_size 512 -tgt_word_vec_size 512 -rnn_size 512 -decoder_rnn_size 1024`

The command above includes the same hyperparameters as in the original paper. This will result in trained models
being saved in `data/processed/`. Alternatively, already trained models can be put in this directory and tested (see next step). 

## Evaluating

To evaluate the model run:

`ipython predict.ipy -- -start 2 -end 2 -beam 3 -models_dir  data/processed/ -test_file data/valid.dataset -tgt_len 500`

This will evaluate the model on the weights of the 2nd epoch (that model should exist in `data/processed/`). 
Tweak start and end if needed to evaluate on more epochs, or other ones. 


## Dataset

Concode dataset can be downloaded from: https://drive.google.com/drive/folders/1kC6fe7JgOmEHhVFaXjzOmKeatTJy1I1W



