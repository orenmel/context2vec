# The context2vec toolkit

With this code you can:
* Use our pre-trained models to represent sentential contexts of target words and target words themselves with low-dimensional vector representations.
* Learn your own context2vec models with your choice of a learning corpus and hyperparameters.

Please cite the following paper if using the code:

**context2vec: Learning Generic Context Embedding with Bidirectional LSTM**  
Oren Melamud, Jacob Goldberger, Ido Dagan. CoNLL, 2016 [[pdf]](http://u.cs.biu.ac.il/~melamuo/publications/context2vec_conll16.pdf).

## Requirements

* Python 3.6
* Chainer 4.2 ([chainer](http://chainer.org/))
* NLTK 3.0 ([NLTK](http://www.nltk.org/))  - optional (only required for the AWE baseline and MSCC evaluation)

Note: Release 1.0 includes the original code that was used in the context2vec paper and has different dependencies (Python 2.7 and Chainer 1.7).

## Installation

* Download the code
* ```python setup.py install```

## Quick-start

* Download pre-trained context2vec models from [[here]](http://u.cs.biu.ac.il/~nlp/resources/downloads/context2vec/)
* Unzip a model into MODEL_DIR
* Run:
```
python context2vec/eval/explore_context2vec.py MODEL_DIR/MODEL_NAME.params
>> this is a [] book
```
* This will embed the entire sentential context 'this is a \_\_ book' and will output the top-10 target words whose embeddings are closest to that of the context.
* Use this as sample code to help you integrate context2vec into your own application.

## Training a new context2vec model

* CORPUS_FILE needs to contain your learning corpus with one sentence per line and tokens separated by spaces.
* Run:
```
python context2vec/train/corpus_by_sent_length.py CORPUS_FILE [max-sentence-length]
```
* This will create a directory CORPUS_FILE.DIR that will contain your preprocessed learning corpus
* Run:
```
python context2vec//train/train_context2vec.py -i CORPUS_FILE.DIR  -w  WORD_EMBEDDINGS -m MODEL  -c lstm --deep yes -t 3 --dropout 0.0 -u 300 -e 10 -p 0.75 -b 100 -g 0
```
* This will create WORD_EMBEDDINGS.targets file with your target word embeddings, a MODEL file, and a MODEL.params file. Put all of these in the same directory MODEL_DIR and you're done.
* See usage documentation for all run-time parameters.
  
NOTE:   
* The current code lowercases all corpus words
* Use of a gpu and mini-batching is highly recommended to achieve good training speeds

### Avoiding exploding gradients

Some users have noted that this configuration can cause exploding gradients
[(see issue #6)](https://github.com/orenmel/context2vec/issues/6). One option
is to turn down the learning rate, by reducing the Adam optimizer's alpha from
0.001 to something lower, e.g. by specifying `-a 0.0005`. As an extra safety
measure, you can enable gradient clipping which could be set to 5 by using the
very scientific method of using the value everyone else seems to be using `-gc
5`.

## Evaluation

### Microsoft Sentence Completion Challenge (MSCC)

* Download the train and test datasets from [[here]](https://www.microsoft.com/en-us/research/project/msr-sentence-completion-challenge/).
* Split the test files into dev and test if you wish to do development tuning.
* Download the pre-trained context2vec model for MSCC from [[here]](http://u.cs.biu.ac.il/~nlp/resources/downloads/context2vec/);
* Or alternatively train your own model as follows:
	- Run ```context2vec/eval/mscc_text_tokenize.py INPUT_FILE OUTPUT_FILE``` for every INPUT_FILE in the MSCC train set.
	- Concatenate all output files into one large learning corpus file.
	- Train a model as explained above.
* Run:  
```
python context2vec/eval/sentence_completion.py Holmes.machine_format.questions.txt Holmes.machine_format.answers.txt RESULTS_FILE MODEL_NAME.params
```


### Senseval-3

* Download the 'English lexical sample' train and test datasets from [[here]](http://web.eecs.umich.edu/~mihalcea/senseval/senseval3/data.html).
* Download the senseval scorer script(scorer2) from [[here]](http://web.eecs.umich.edu/~mihalcea/senseval/senseval3/scoring/scorer2.c) and build it.
* Train your own context2vec model or use one of the pre-trained models provided.
* For development runs do:
```
python context2vec/eval/wsd/wsd_main.py EnglishLS.train EnglishLS.train RESULTS_FILE MODEL_NAME.params 1
```
```
scorer2 RESULTS_FILE EnglishLS.train.key EnglishLS.sensemap
```
* For test runs do:
```
python context2vec/eval/wsd/wsd_main.py EnglishLS.train EnglishLS.test RESULTS_FILE MODEL_NAME.params 1
```
```
scorer2 RESULTS_FILE EnglishLS.test.key EnglishLS.sensemap
```



### Lexical Substitution

The code for the lexical substitution evaluation is included in a separate repository [[here]](https://github.com/orenmel/lexsub).

## Known issues

* All words are converted to lowercase.
* Using gpu and/or mini-batches is not supported at test time.


## License

Apache 2.0





