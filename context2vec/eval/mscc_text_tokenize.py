'''
Used to convert the Microsoft Sentence Completion Challnege (MSCC) learning corpus into a one-sentence-per-line format.
'''

import sys
from nltk.tokenize import word_tokenize, sent_tokenize


def write_paragraph_lines(paragraph_lines):
    paragraph_str = ' '.join(paragraph_lines)
    for sent in sent_tokenize(paragraph_str):
        if lowercase:
            sent = sent.lower()
        output_file.write(' '.join(word_tokenize(sent))+'\n')

lowercase = True

if len(sys.argv) < 2:
    sys.stderr.write("Usage: %s <input-filename> <output-filename>\n" % sys.argv[0])
    sys.exit(1)
    
input_file = open(sys.argv[1],'r')
output_file = open(sys.argv[2],'w')

paragraph_lines = []
for i, line in enumerate(input_file):
    if len(line.strip()) == 0 and len(paragraph_lines) > 0:
        write_paragraph_lines(paragraph_lines)        
        paragraph_lines = []
    else:
        paragraph_lines.append(line)
    
if len(paragraph_lines) > 0:
    write_paragraph_lines(paragraph_lines)
    
print('Read {} lines'.format(i))
                          
input_file.close()
output_file.close()
        
        
        
        
            