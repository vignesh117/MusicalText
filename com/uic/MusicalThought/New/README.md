__author__ = 'vignesh'

Instructions to run the code
============================

Configurations
===============

1. Place the data in a desired folder
2. Edit the config.cfg file as follows
    i) Change the 'trainfile' configuration to point to the trainfile
    ii) Change the 'testfile' configuration to point to the testfile
3. Edit the configuration 'classifier' to point to the stanfordNER CRF classifier
4. Edit the configuration 'tagger' to point to the StanforedNER tagger

Running
=======

1. Run the test.py. This will generate the train and test sentences and serialize them in the same directory
2. Run the ExtracFeatures.py. This will extract features for the train and test sentences and dump them in the
    same directory
3. Run LearnAndClassify.py. This will run all the classifiers on the data and print out the performance report
    on the screeen


Tools used
==========

1. Sklearn ( Python )
2. Numpy ( Python )
3. Stanford NER tagger
4. Stanford Dependency parser