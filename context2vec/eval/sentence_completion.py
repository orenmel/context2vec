'''
Evaluates context2vec on the Microsoft Sentence Completion Challnege (MSCC)

questions file: *.machine_format.questions.txt
1a) I have seen it on him , and could [write] to it .
1b) I have seen it on him , and could [migrate] to it .
1c) I have seen it on him , and could [climb] to it .
1d) I have seen it on him , and could [swear] to it .
1e) I have seen it on him , and could [contribute] to it .

answers file: *.machine_format.answers.txt
1d) I have seen it on him , and could [swear] to it .
'''
import sys
import re
from nltk.tokenize import word_tokenize
from context2vec.common.defs import Toks
from context2vec.common.model_reader import ModelReader
import numpy as np

exp = re.compile('(.*)\[(.+)\](.*)')

def parse_input(line, word2index):
    
    line = line[line.find(' ')+1:].strip()
    if debug:
        print(line)
    segments = exp.match(line)
    if segments is not None:
        seg_left = segments.group(1)
        target_word = segments.group(2).lower()
        seg_right = segments.group(3)
        
        words_left = word_tokenize(seg_left)
        words_right = word_tokenize(seg_right)
        words = words_left + [target_word] + words_right
        if debug:
            print(words)
        target_pos = len(words_left)
        sent = []
        for word in words:
            word = word.lower()
            sent.append(word)
    else:
        raise Exception("Failed to parse line into segments")
    if debug:
        print(sent)
    return sent, target_pos, target_word


def answer_next_question(fd, model, w, word2index):
    
    best_sim = None
    best_answer = None
    context_v = None
    for _ in range(5):
        line = fd.readline()
        if not line:
            return None
        sent, target_pos, target_word = parse_input(line, word2index)
        if target_word == None:
            raise Exception("Can't find the target word.") 
        if len(sent) <= 1:
            raise Exception("Can't find context for target word.")
        target_v = w[word2index[sent[target_pos]]] if sent[target_pos] in word2index else w[Toks.UNK]
        
        if context_v is None: # all contexts are the same in the same question
            context_v = model.context2vec(sent, target_pos) 
            context_v = context_v / np.sqrt((context_v * context_v).sum())
        
        sim = target_v.dot(context_v)
        
        if debug:
            print('target_word', target_word)
            print('target_pos', target_pos)
            print('sim', sim)
        
        if best_sim is None or sim > best_sim:
            best_sim = sim
            best_answer = target_word
            
    return best_answer
            
def read_next_answer(fd, word2index):
    line = fd.readline()
    if not line:
        return None
    _, _, target_word = parse_input(line, word2index)
    if debug:
        print('\ngold target word', target_word)
        print('***************************')
    return target_word


if __name__ == '__main__':
    
    debug = False
    
    if len(sys.argv) < 4:
        sys.stderr.write("Usage: %s <questions-filename> <gold-filename> <results-filename> <model-params-filename>\n" % sys.argv[0])
        sys.exit(1)
        
    questions_fd = open(sys.argv[1],'r')
    gold_fd = open(sys.argv[2],'r')
    results_fd = open(sys.argv[3], 'w')    
    model_params_filename = sys.argv[4]        
    
    model_reader = ModelReader(model_params_filename)
    
    total_questions = 0
    correct = 0
    while True:      
        best_answer = answer_next_question(questions_fd, model_reader.model, model_reader.w, model_reader.word2index)
        gold_answer = read_next_answer(gold_fd, model_reader.word2index)
        if best_answer == None or gold_answer == None:
            break
        total_questions += 1
        if best_answer == gold_answer:
            correct += 1
            
    accuracy = float(correct) / total_questions
    print("Accuracy: {0}. Correct: {1}. Total: {2}".format(accuracy, correct, total_questions))
    results_fd.write("Accuracy: {0}. Correct: {1}. Total: {2}.\n".format(accuracy, correct, total_questions))
        
    questions_fd.close()
    gold_fd.close()
    results_fd.close()   
    