# Variational Hierarchical User-based Conversation Model
Implementation of Variational Hierarchical User-based Conversation Model (VHUCM) in EMNLP-IJCNLP 2019

## Environment
We run this code in this environment.

- Hardware
    - Intel(R) Xeon(R) CPU E5-2640 v4 @ 2.40GHz
    - GeForce GTX 1080 Ti 11GB

- Software
    - Python 3.7.4
    - PyTorch 1.3.0
    - NumPy 1.16.4
    - CUDA 9.0

## Run the code
### Training a model
To train the model, run the ```RunTrain.sh``` file.
To understand the meaning of arguments, please see the ```config.py``` file.


### Generating responses 
To generate responses from trained model, run the ```RunExportTestSamples.sh``` file. 
It outputs txt files that have a set of input conversation context, generated responses, and ground truth response. 


### Evaluating a model by generated responses
To evaluate the model, run the ```RunEval.sh``` file.
It outputs the score of BLUE, ROUGE, and the length of responses.



## Data
In the paper, we build and use Twitter conversation corpus. 
We prepared to release the corpus and submitted a part of the data to EMNLP submission site.
However, we got comments from other researchers about the privacy issues.
We took the comments and prepare the opening the data to the research community such as removing personally identifiable information and taking Institutional Review Board approval.

We use [Cornell Movie Dialogs Corpus](https://www.cs.cornell.edu/~cristian/Cornell_Movie-Dialogs_Corpus.html) to show the availability of the implementation.


## Reference
- https://github.com/ctr4si/A-Hierarchical-Latent-Structure-for-Variational-Conversation-Modeling
- https://github.com/jiweil/Neural-Dialogue-Generation
- https://github.com/OpenXAIProject/Neural-Conversation-Models
