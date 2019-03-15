import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.autograd import Variable
from utils import shiftLeft, bottle, unbottle


class ProdGenerator(nn.Module):
    def __init__(self, rnn_size, vocabs, opt):
        super(ProdGenerator, self).__init__()
        self.opt = opt
        self.mask = Variable(vocabs['mask'].float(), requires_grad=False)
        self.linear = nn.Linear(rnn_size, len(vocabs['next_rules']))  # only non unk rules
        self.linear_copy = nn.Linear(rnn_size, 1)
        self.tgt_pad = vocabs['next_rules'].stoi['<blank>']
        self.tgt_unk = vocabs['next_rules'].stoi['<unk>']
        self.vocabs = vocabs

    def forward(self, hidden, attn, src_map, batch):
        out = self.linear(hidden)
        # batch['nt'] contains padding.
        batch_by_tlen_, slen = attn.size()
        batch_size, slen_, cvocab = src_map.size()

        non_terminals = batch['nt'].contiguous().view(-1)
        masked_out = torch.add(out, torch.index_select(self.mask, 0, Variable(non_terminals, requires_grad=False)))
        prob = F.softmax(masked_out, dim=1)

        # Probability of copying p(z=1) batch.
        copy = F.sigmoid(self.linear_copy(hidden))

        # Probibility of not copying: p_{word}(w) * (1 - p(z))
        masked_copy = Variable(non_terminals.view(-1, 1).eq(self.vocabs['nt'].stoi['IdentifierNT']).float()) * copy
        out_prob = torch.mul(prob, 1 - masked_copy.expand_as(prob))  # The ones without IdentifierNT are left untouched
        mul_attn = torch.mul(attn, masked_copy.expand_as(attn))
        copy_prob = torch.bmm(mul_attn.view(batch_size, -1, slen), Variable(src_map, requires_grad=False))
        copy_prob = copy_prob.view(-1, cvocab)  # bottle it again to get batch_by_len times cvocab
        return torch.cat([out_prob, copy_prob], 1)  # batch_by_tlen x (out_vocab + cvocab)

    def computeLoss(self, scores, batch):

        batch_size = batch['seq2seq'].size(0)

        target = Variable(batch['next_rules'].contiguous().view(-1), requires_grad=False)
        align = Variable(batch['concode_next_rules_in_src_nums'].contiguous().view(-1), requires_grad=False)
        align_unk = batch['concode_vocab'][0].stoi['<unk>']

        offset = len(self.vocabs['next_rules'])

        out = scores.gather(1, align.view(-1, 1) + offset).view(-1).mul(
            align.ne(align_unk).float())  # all where copy is not unk
        tmp = scores.gather(1, target.view(-1, 1)).view(-1)

        out = out + 1e-20 + tmp.mul(target.ne(self.tgt_unk).float()) + \
              tmp.mul(align.eq(align_unk).float()).mul(target.eq(self.tgt_unk).float())  # copy and target are unks

        # Drop padding.
        loss = -out.log().mul(target.ne(self.tgt_pad).float()).sum()
        scores_data = scores.data.clone()
        target_data = target.data.clone()  # computeLoss populates this

        scores_data = self.collapseCopyScores(unbottle(scores_data, batch_size), batch)
        scores_data = bottle(scores_data)

        # Correct target copy token instead of <unk>
        # tgt[i] = align[i] + len(tgt_vocab)
        # for i such that tgt[i] == 0 and align[i] != 0
        # when target is <unk> but can be copied, make sure we get the copy index right
        correct_mask = target_data.eq(self.tgt_unk) * align.data.ne(align_unk)
        correct_copy = (align.data + offset) * correct_mask.long()
        target_data = (target_data * (1 - correct_mask).long()) + correct_copy

        pred = scores_data.max(1)[1]
        non_padding = target_data.ne(self.tgt_pad)
        num_correct = pred.eq(target_data).masked_select(non_padding).sum()

        return loss, non_padding.sum(), num_correct  # , stats

    def collapseCopyScores(self, scores, batch):
        """
        Given scores from an expanded dictionary
        corresponding to a batch, sums together copies,
        with a dictionary word when it is ambigious.
        """
        tgt_vocab = self.vocabs['next_rules']
        offset = len(tgt_vocab)
        for b in range(batch['seq2seq'].size(0)):
            src_vocab = batch['concode_vocab'][b]

            for i in range(1, len(src_vocab)):
                sw = "IdentifierNT-->" + src_vocab.itos[i]
                ti = tgt_vocab.stoi[sw] if sw in tgt_vocab.stoi else self.tgt_unk
                if ti != self.tgt_unk:
                    scores[b, :, ti] += scores[b, :, offset + i]
                    scores[b, :, offset + i].fill_(1e-20)
        return scores
