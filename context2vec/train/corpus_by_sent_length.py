'''
Converts a single large corpus file into a directory, in which for every sentence length k there is a separate file containing all sentences of that length. 
'''

import sys
import os
from collections import Counter
from context2vec.common.defs import SENT_COUNTS_FILENAME, WORD_COUNTS_FILENAME, TOTAL_COUNTS_FILENAME


def get_file(sub_files, corpus_dir, num_filename):
    if num_filename not in sub_files:
        full_file_name = corpus_dir + '/' + num_filename
        sub_files[num_filename] = open(full_file_name, 'w')        
    return sub_files[num_filename]
   

if __name__ == '__main__':
    
    if len(sys.argv) < 2:
        print("usage: %s <corpus-file> [max-sent-len]"  % (sys.argv[0]))
        sys.exit(1)
        
    corpus_file = open(sys.argv[1], 'r')
    if len(sys.argv) > 2:
        max_sent_len = int(sys.argv[2])
    else:
        max_sent_len = 128    
    print('Using maximum sentence length: ' + str(max_sent_len))
    
    corpus_dir = sys.argv[1]+'.DIR'
    os.makedirs(corpus_dir)
    sent_counts_file = open(corpus_dir+'/'+SENT_COUNTS_FILENAME, 'w')
    word_counts_file = open(corpus_dir+'/'+WORD_COUNTS_FILENAME, 'w')
    totals_file = open(corpus_dir+'/'+TOTAL_COUNTS_FILENAME, 'w')
    
    sub_files = {}
    sent_counts = Counter()
    word_counts = Counter()
    
    for line in corpus_file:
        words = line.strip().lower().split()
        wordnum = len(words)
        if wordnum > 1 and wordnum <= max_sent_len:
            num_filename = 'sent.' + str(wordnum)
            sub_file = get_file(sub_files, corpus_dir, num_filename)
            sub_file.write(line)
            sent_counts[num_filename] += 1
            for word in words:
                word_counts[word] += 1
               
    for sub_file in sub_files.values():
        sub_file.close()
        
    for num_filename, count in sent_counts.most_common():
        sent_counts_file.write(num_filename+'\t'+str(count)+'\n')
    
    for word, count in word_counts.most_common():
        word_counts_file.write(word+'\t'+str(count)+'\n')
    
    totals_file.write('total sents read: {}\n'.format(sum(sent_counts.values())))
    totals_file.write('total words read: {}\n'.format(sum(word_counts.values())))
    
    corpus_file.close()
    sent_counts_file.close()
    word_counts_file.close()
    totals_file.close()
    
    print('Done')