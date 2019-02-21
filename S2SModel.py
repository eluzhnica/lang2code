import torch
from torch.autograd import Variable
import torch.nn as nn

from Decoder import Decoder
from Encoder import Encoder
from Statistics import Statistics
from utils import bottle
from Beam import Beam
from CopyGenerator import CopyGenerator
from decoders import DecoderState


class S2SModel(nn.Module):
    def __init__(self, opt, vocabs):
        super(S2SModel, self).__init__()

        self.opt = opt
        self.vocabs = vocabs
        self.encoder = Encoder(vocabs, opt)
        self.decoder = Decoder(vocabs, opt)
        self.generator = CopyGenerator(self.opt.decoder_rnn_size, vocabs, self.opt)
        # self.cuda()

    def forward(self, batch):
        # initial parent states for Prod Decoder
        batch_size = batch['seq2seq'].size(0)
        batch['parent_states'] = {}
        for j in range(0, batch_size):
            batch['parent_states'][j] = {}
            batch['parent_states'][j][0] = Variable(torch.zeros(1, 1, self.opt.decoder_rnn_size),
                                                    requires_grad=False)

        context, context_lengths, enc_hidden = self.encoder(batch)

        dec_initial_state = DecoderState(enc_hidden, Variable(torch.zeros(batch_size, 1, self.opt.decoder_rnn_size),
                                                              requires_grad=False))

        output, attn, copy_attn = self.decoder(batch, context, context_lengths, dec_initial_state)

        del batch['parent_states']

        src_map = torch.zeros(0, 0)
        # print(src_map)
        # print(batch['concode_src_map_vars'].shape)
        src_map = torch.cat((src_map, batch['concode_src_map_vars']), 1)
        src_map = torch.cat((src_map, batch['concode_src_map_methods']), 1)

        scores = self.generator(bottle(output), bottle(copy_attn), src_map, batch)
        loss, total, correct = self.generator.computeLoss(scores, batch)

        return loss, Statistics(loss.data[0], total, correct, self.encoder.n_src_words)

    # This only works for a batch size of 1
    # def predict(self, batch, opt, vis_params):
    #     curr_batch_size = batch['seq2seq'].size(0)
    #     assert (curr_batch_size == 1)
    #     context, context_lengths, enc_hidden = self.encoder(batch)
    #     return self.decoder.predict(enc_hidden, context, context_lengths, batch, opt.beam_size, opt.max_sent_length,
    #                                 self.generator, opt.replace_unk, vis_params)
