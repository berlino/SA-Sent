# Learning Latent Opinions for Aspect-level Sentiment Classification 

to appear in AAAI 2018

## Requirements

* python 2.7
* pytorch, nltk, stanford\_corenlp\_pywrapper, numpy, bs4
* glove for pre-trained vectors

## Steps

1. Download data from SemEval 2014 task 4
2. check the file path in reader.py and tokenizer.py, then run ./reader.py to generate the pkl file for training and these file will be located at the paths that are defined in config.py
3. run ./train.py > logs/1.log to start training. Turn on gpu option in config file if you have gpu resource. Make sure you redirect your output to log file since during the test step, it will 
generate opinion sequence for intepretation.


## Question?

Feel free to open github issues or email me.
