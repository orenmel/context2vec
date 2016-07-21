#!/usr/bin/env python
"""
Learns context2vec's parametric model
"""
import argparse
import time
import sys

import numpy as np
from chainer import cuda
import chainer.links as L
import chainer.optimizers as O
import chainer.serializers as S
import chainer.computational_graph as C

from sentence_reader import SentenceReaderDir
from context2vec.common.context_models import BiLstmContext
from context2vec.common.defs import IN_TO_OUT_UNITS_RATIO, NEGATIVE_SAMPLING_NUM


#TODO: LOWER AS ARG
def dump_embeddings(filename, w, units, index2word):
    with open(filename, 'w') as f:
        f.write('%d %d\n' % (len(index2word), units))
        for i in range(w.shape[0]):
            v = ' '.join(['%f' % v for v in w[i]])
            f.write('%s %s\n' % (index2word[i], v))
            
def dump_comp_graph(filename, vs):
    g = C.build_computational_graph(vs)
    with open(filename, 'w') as o:
        o.write(g.dump())
        
        
def parse_arguments():

    parser = argparse.ArgumentParser()
    parser.add_argument('--indir', '-i',
                        default=None,
                        help='input corpus directory')
    parser.add_argument('--trimfreq', '-t', default=0, type=int,
                        help='minimum frequency for word in training')
    parser.add_argument('--ns_power', '-p', default=0.75, type=float,
                        help='negative sampling power')
    parser.add_argument('--dropout', '-o', default=0.0, type=float,
                        help='NN dropout')
    parser.add_argument('--wordsfile', '-w',
                        default=None,
                        help='word embeddings output filename')
    parser.add_argument('--modelfile', '-m',
                        default=None,
                        help='model output filename')
    parser.add_argument('--cgfile', '-cg',
                        default=None,
                        help='computational graph output filename (for debug)')
    parser.add_argument('--gpu', '-g', default=-1, type=int,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--unit', '-u', default=300, type=int,
                        help='number of units (dimensions) of one context word')
    parser.add_argument('--batchsize', '-b', type=int, default=100,
                        help='learning minibatch size')
    parser.add_argument('--epoch', '-e', default=10, type=int,
                        help='number of epochs to learn')
    parser.add_argument('--context', '-c', choices=['lstm'],
                        default='lstm',
                        help='context type ("lstm")')
    parser.add_argument('--deep', '-d', choices=['yes', 'no'],
                        default=None,
                        help='use deep NN architecture')
    
    args = parser.parse_args()
    
    if args.deep == 'yes':
        args.deep = True
    elif args.deep == 'no':
        args.deep = False
    else:
        raise Exception("Invalid deep choice: " + args.deep)
    
    print('GPU: {}'.format(args.gpu))
    print('# unit: {}'.format(args.unit))
    print('Minibatch-size: {}'.format(args.batchsize))
    print('# epoch: {}'.format(args.epoch))
    print('Context type: {}'.format(args.context))
    print('Deep: {}'.format(args.deep))
    print('Dropout: {}'.format(args.dropout))
    print('Trimfreq: {}'.format(args.trimfreq))
    print('NS Power: {}'.format(args.ns_power))
    print('')
       
    return args 
    


args = parse_arguments()

context_word_units = args.unit
lstm_hidden_units = IN_TO_OUT_UNITS_RATIO*args.unit
target_word_units = IN_TO_OUT_UNITS_RATIO*args.unit

if args.gpu >= 0:
    cuda.check_cuda_available()
    cuda.get_device(args.gpu).use()
xp = cuda.cupy if args.gpu >= 0 else np
    
reader = SentenceReaderDir(args.indir, args.trimfreq, args.batchsize)
print('n_vocab: %d' % (len(reader.word2index)-3)) # excluding the three special tokens
print('corpus size: %d' % (reader.total_words))

cs = [reader.trimmed_word2count[w] for w in range(len(reader.trimmed_word2count))]
loss_func = L.NegativeSampling(target_word_units, cs, NEGATIVE_SAMPLING_NUM, args.ns_power)

if args.context == 'lstm':
    model = BiLstmContext(args.deep, args.gpu, reader.word2index, context_word_units, lstm_hidden_units, target_word_units, loss_func, True, args.dropout)
else:
    raise Exception('Unknown context type: {}'.format(args.context))

optimizer = O.Adam()
optimizer.setup(model)

STATUS_INTERVAL = 1000000

for epoch in range(args.epoch):
    begin_time = time.time()
    cur_at = begin_time
    word_count = 0
    next_count = STATUS_INTERVAL
    accum_loss = 0.0
    last_accum_loss = 0.0
    last_word_count = 0
    print('epoch: {0}'.format(epoch))

    reader.open()    
    for sent in reader.next_batch():

        model.zerograds()
        loss = model(sent)
        accum_loss += loss.data
        loss.backward()
        del loss
        optimizer.update()

        word_count += len(sent)*len(sent[0]) # all sents in a batch are the same length
        accum_mean_loss = float(accum_loss)/word_count if accum_loss > 0.0 else 0.0

        if word_count >= next_count:        
            now = time.time()
            duration = now - cur_at
            throuput = float((word_count-last_word_count)) / (now - cur_at)
            cur_mean_loss = (float(accum_loss)-last_accum_loss)/(word_count-last_word_count)
            print('{} words, {:.2f} sec, {:.2f} words/sec, {:.4f} accum_loss/word, {:.4f} cur_loss/word'.format(
                word_count, duration, throuput, accum_mean_loss, cur_mean_loss))
            next_count += STATUS_INTERVAL
            cur_at = now
            last_accum_loss = float(accum_loss)
            last_word_count = word_count

    print 'accum words per epoch', word_count, 'accum_loss', accum_loss, 'accum_loss/word', accum_mean_loss
    reader.close()
    
if args.wordsfile != None:        
    dump_embeddings(args.wordsfile+'.targets', model.loss_func.W.data, target_word_units, reader.index2word)

if args.modelfile != None:
    S.save_npz(args.modelfile, model)
    
with open(args.modelfile + '.params', 'w') as f:
    f.write('model_file\t' + args.modelfile[args.modelfile.rfind('/')+1:]+'\n')
    f.write('words_file\t' + args.wordsfile[args.wordsfile.rfind('/')+1:]+'.targets\n')
    f.write('unit\t' + str(args.unit)+'\n')
    if args.deep:
        f.write('deep\tyes\n')
    else:
        f.write('deep\tno\n')
    f.write('drop_ratio\t' + str(args.dropout)+'\n')    
    f.write('#\t{}\n'.format(' '.join(sys.argv)))
    
    

