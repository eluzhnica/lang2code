import torch
from torch import nn
import torch.nn.functional as F


class EncoderOne(nn.Module):

    def __init__(self,
                 type_words,
                 embedding_dim,
                 name_words,
                 encoder_dim,
                 ):
        super(EncoderOne, self).__init__()

        self.type_embedding = nn.Embedding(len(type_words), embedding_dim)

        self.name_embedding = nn.Embedding(len(name_words), embedding_dim)

        self.name_lstm = nn.LSTM(embedding_dim, encoder_dim, bidirectional=True)
        self.name_state = None
        # self.type_lstm = nn.LSTM(type_embedding_dim, type_lstm_hidden_size, bidirectional=True)

        self.variable_lstm = nn.LSTM(embedding_dim, encoder_dim, bidirectional=True)
        self.variable_state = None
        self.method_lstm = nn.LSTM(embedding_dim, encoder_dim, bidirectional=True)
        self.method_state = None
        self.nl_lstm = nn.LSTM(embedding_dim, encoder_dim, bidirectional=True)
        self.nl_state = None
        self.init_hidden()

    def init_variable(self, variable):
        setattr(self, variable, (torch.zeros(2, 1, self.variable_lstm.hidden_size),
                                 torch.zeros(2, 1, self.variable_lstm.hidden_size)))

    def init_hidden(self):
        # Before we've done anything, we dont have any hidden state.
        # Refer to the Pytorch documentation to see exactly
        # why they have this dimensionality.
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        self.variable_state = (torch.zeros(2, 1, self.variable_lstm.hidden_size),
                               torch.zeros(2, 1, self.variable_lstm.hidden_size))
        self.method_state = (torch.zeros(2, 1, self.variable_lstm.hidden_size),
                             torch.zeros(2, 1, self.variable_lstm.hidden_size))
        self.nl_state = (torch.zeros(2, 1, self.variable_lstm.hidden_size),
                         torch.zeros(2, 1, self.variable_lstm.hidden_size))
        self.name_state = (torch.zeros(2, 1, self.variable_lstm.hidden_size),
                           torch.zeros(2, 1, self.variable_lstm.hidden_size))

    def forward(self, nl, methods, fields):
        embedded_nl = self.name_embedding(nl)
        nl_encoding, self.nl_state = self.nl_lstm(embedded_nl.view(len(nl), 1, -1), self.nl_state)

        environment_encodings = []
        for field in fields:
            field_type = field['type']
            field_name = field['name']
            print("Field name", field_name.shape)

            field_type_embedding = self.type_embedding(field_type)
            field_name_embedding = self.name_embedding(field_name)

            _, self.name_state = self.name_lstm(
                field_name_embedding.view(len(field_name_embedding), 1, -1),
                self.name_state
            )
            field_name_encoding = self.name_state[0]
            field_name_encoding = field_name_encoding.view(1, -1)
            print(field_name_encoding.shape)
            print(field_type_embedding.shape)
            self.init_variable('name_state')

            field_encoding = torch.stack((field_type_embedding, field_name_encoding))
            field_encoding, self.variable_state = self.variable_lstm(
                field_encoding.view(field_encoding.shape[0], 1, -1),
                self.variable_state
            )
            self.init_variable('variable_state')
            environment_encodings.append(field_encoding)

        for method in methods:
            method_type = method['type']
            method_name = method['name']

            method_type_embedding = self.type_embedding(method_type)
            method_name_embedding = self.name_embedding(method_name)

            _, self.name_state = self.name_lstm(
                method_name_embedding.view(len(method_name_embedding), 1, -1),
                self.name_state
            )
            method_name_encoding = self.name_state[0]
            method_name_encoding = method_name_encoding.view(1, -1)
            self.init_variable('name_state')

            method_encoding = torch.stack((method_type_embedding, method_name_encoding))
            method_encoding, self.method_state = self.method_lstm(
                method_encoding.view(method_encoding.shape[0], 1, -1),
                self.method_state
            )
            self.init_variable('method_state')
            environment_encodings.append(method_encoding)

        print("End")
        return nl_encoding, environment_encodings,


class DecoderOne(nn.Module):
    def __init__(self,
                 non_terminals_length,
                 rules_length,
                 embedding_dim,
                 decoder_rnn_size,
                 max_prod_rules
                 ):
        super(DecoderOne, self).__init__()

        self.nt_embedding = nn.Embedding(non_terminals_length, embedding_dim)
        self.rules_embedding = nn.Embedding(rules_length, embedding_dim)

        # self.decoder_lstm = nn.LSTM(3*embedding_dim + decoder_rnn_size, decoder_rnn_size)
        self.decoder_lstm = nn.LSTM(1*embedding_dim + decoder_rnn_size, decoder_rnn_size)
        self.decoder_state = None

        self.nl_attention = Attention(decoder_rnn_size)
        self.env_attention = Attention(decoder_rnn_size)

        # self.copy_linear = nn.Linear(3*decoder_rnn_size, decoder_rnn_size)
        self.attn_linear = nn.Linear(3*decoder_rnn_size, decoder_rnn_size)

        self.ct_weights = nn.Linear(decoder_rnn_size, max_prod_rules)

        self.copy_weights = nn.Linear(decoder_rnn_size, 1)

        self.init_hidden()

    def init_hidden(self):
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        self.decoder_state = (torch.zeros(1, 1, self.decoder_lstm.hidden_size),
                               torch.zeros(1, 1, self.decoder_lstm.hidden_size))

    def forward(self, init_prod, nt_to_rules, nl_encoding, env_encoding):
        init_prod_emb = self.nt_embedding(init_prod)

        # lstm_input = torch.cat((current_nt, last_prod))
        st, self.decoder_state = self.decoder_lstm(init_prod_emb, self.decoder_state)
        self.decoder_state = self.decoder_state[0]

        zt, z_attentions = self.nl_attention(nl_encoding, st)
        # zt = zt.view(zt.shape[2], 1, -1)

        et, e_attentions = self.env_attention(env_encoding, zt)
        # et = et.view(et.shape[2], 1, -1)

        conc = torch.cat((st, zt, et))
        ct = F.tanh(self.attn_linear(conc))
        next_prod_rule = self.ct_weights(ct)
        next_prod_rule_masked = next_prod_rule.where(prod_rules_masked(init_prod), next_prod_rule, -1000000)
        next_prod_rule = F.softmax(next_prod_rule_masked)

        copy_t = F.sigmoid(self.copy_weights(ct))

        res = copy_t*et + next_prod_rule*(1-copy_t)

        return res


class Attention(nn.Module):
    def __init__(self, dim):
        super(Attention, self).__init__()
        self.linear_out = nn.Linear(dim*2, dim)
        self.mask = None

    def set_mask(self, mask):
        """
        Sets indices to be masked

        Args:
            mask (torch.Tensor): tensor containing indices to be masked
        """
        self.mask = mask

    def forward(self, output, context):
        batch_size = output.size(0)
        hidden_size = output.size(2)
        input_size = context.size(1)
        # (batch, out_len, dim) * (batch, in_len, dim) -> (batch, out_len, in_len)
        attn = torch.bmm(output, context.transpose(1, 2))
        if self.mask is not None:
            attn.data.masked_fill_(self.mask, -float('inf'))
        attn = F.softmax(attn.view(-1, input_size), dim=1).view(batch_size, -1, input_size)

        # (batch, out_len, in_len) * (batch, in_len, dim) -> (batch, out_len, dim)
        mix = torch.bmm(attn, context)

        # concat -> (batch, out_len, 2*dim)
        combined = torch.cat((mix, output), dim=2)
        # output -> (batch, out_len, dim)
        output = F.tanh(self.linear_out(combined.view(-1, 2 * hidden_size))).view(batch_size, -1, hidden_size)

        return output, attn

