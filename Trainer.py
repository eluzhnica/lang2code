from Statistics import Statistics
import torch
import torch.optim as optim
from torch.nn.utils import clip_grad_norm
import torch.nn as nn
import time
import matplotlib.pyplot as plt


class Trainer:
    def __init__(self, model):
        self.model = model
        self.opt = model.module.opt if isinstance(model, nn.parallel.DistributedDataParallel) else model.opt
        self.start_epoch = 1

        self.lr = self.opt.learning_rate
        self.betas = [0.9, 0.98]
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr,
                                    betas=self.betas, eps=1e-9)
        self.train_scores = []
        self.valid_scores = []

        self.train_ppl = []
        self.valid_ppl = []

    def save_checkpoint(self, epoch, valid_stats):
        real_model = self.model

        model_state_dict = real_model.state_dict()
        self.opt.learning_rate = self.lr
        checkpoint = {
            'model': model_state_dict,
            'vocab': real_model.vocabs,
            'opt': self.opt,
            'epoch': epoch,
            'optim': self.optimizer.state_dict()
        }
        torch.save(checkpoint,
                   '%s_acc_%.2f_ppl_%.2f_e%d.pt'
                   % (self.opt.save_model + '/model', valid_stats.accuracy(),
                      valid_stats.ppl(), epoch))

    def run_train_batched(self, train_data, valid_data, vocabs):
        print(self.model.parameters)

        total_train = train_data.compute_batches(self.opt.batch_size, vocabs, self.opt.max_camel, 0, 1)

        total_valid = valid_data.compute_batches(
            10, vocabs, self.opt.max_camel, 0,
            1, randomize=False)

        print('Computed Batches. Total train={}, Total valid={}'.format(total_train, total_valid))

        for epoch in range(self.start_epoch, self.opt.epochs + 1):
            self.model.train()

            total_stats = Statistics()
            for idx, batch in enumerate(train_data.batches):
                loss, batch_stats = self.model.forward(batch)
                batch_size = batch['code'].size(0)
                loss.div(batch_size).backward()

                report_stats = Statistics()
                report_stats.update(batch_stats)
                total_stats.update(batch_stats)

                # clip_grad_norm(self.model.parameters(), self.opt.max_grad_norm)
                self.optimizer.step()
                self.optimizer.zero_grad()

                #report_stats.output(epoch, idx + 1, len(train_data.batches), total_stats.start_time)

            print('Train perplexity: %g' % total_stats.ppl())
            print('Train accuracy: %g' % total_stats.accuracy())

            self.train_scores.append(total_stats.accuracy())
            self.train_ppl.append(total_stats.ppl())

            self.model.eval() # set to evaluation mode so no gradients are accumulated
            valid_stats = Statistics()
            for idx, batch in enumerate(valid_data.batches):
                loss, batch_stats = self.model.forward(batch)
                valid_stats.update(batch_stats)

            print('Validation perplexity: %g' % valid_stats.ppl())
            print('Validation accuracy: %g' % valid_stats.accuracy())

            self.valid_scores.append(valid_stats.accuracy())
            self.valid_ppl.append(valid_stats.ppl())

            # plt.figure(1)
            # plt.plot(self.train_scores, label='train accuracy', color='red')
            # plt.plot(self.valid_scores, label='valid accuracy', color='blue')
            # plt.savefig('accuracy.png')
            #
            # plt.figure(2)
            # plt.plot(self.train_ppl, label='train perplexity', color='red')
            # plt.plot(self.valid_ppl, label='valid perplexity', color='blue')
            # plt.savefig('perplexity.png')

            print('Saving model')
            self.save_checkpoint(epoch, valid_stats)
            print('Model saved')
