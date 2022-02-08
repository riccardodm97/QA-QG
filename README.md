# Question Answering - Question Generation 

Question Answering (QA) and Question Generation (QG) are two Natural Language Processing (NLP) and machine comprehension (MC) tasks, which have gained popularity over the past few years. They are fascinating yet challenging problems that have an instrinsic connection and can be regarded as dual tasks. 

The focus of QA task is to build a system which is capable to automatically produce an answer given a question relative to passage of text provided in a natural language form. The QG problem on the other hand aims at at generating questions given a specific answer related to a text passage. Both this tasks are clearly tough for machines, requiring both understanding of natural language and ability to contextualize.

We made use of the Stanford Question Answering Dataset (SQuAD) v1.1, for both problems. SQuAD is a reading comprehension dataset made up of 100,000+ questions on 500+ articles posed by crowd workers on a collection of Wikipedia articles, where the answer to each question is a text segment, or span, from the relevant reading passage.

To tackled these tasks we developed and implemented multiple neural architectures and took advantage of the two most popular ones in the NLP field: Recurrent Neural Networks (RNN) and Transformers.

### QA example 

> <b>Context</b> : Canadian football has 12 players on the field per team rather than 11; the field is roughly 10 yards wider, and 10 yards longer between end-zones that are themselves 10 yards deeper; and a team has only three downs to gain 10 yards, which results in less offensive rushing than in the American game. In the Canadian game all players on the defending team, when a down begins, must be at least 1 yard from the line of scrimmage. 
>
> <b>Question</b> : How far away from the line of scrimmage must Canadian football defenders be?
>
> <b>Correct Answer</b> : 1 yard
> 
> <b>Model Answer</b> : at least 1 yard


## Usage 

### Setup 

The following packages are needed to be able to run the code:

```
- numpy 
- pandas == 1.3.5
- pyarrow == 6.0.0
- gensim == 4.0.1
- torch == 1.10.1
- datasets == 1.17.0
- transformers == 4.15.0
```
### Train
To train the models on the dataset present in the 'data' folder, run the following script : 

```
python train.py -t 'task' -m 'model' -d 'dataset' 
```
With arguments :
```
 -t   the task to perform (Question Answering or Question Generation); options: ['qa', 'qg']
 -m   the model to be trained ; options: ['DrQA','Bert','Electra','BaseQG','BertQG','RefNetQG']
 -d   the name of the json file which contains the dataset
 -l   do not use this option as it requires an access token to be able to log on WandB platform 
```
### QA Inference
To test a trained model for the Question Answering task run :
```
python compute_answers.py 'path_to_json_file'
```
where the json file is test set formatted as as the official SQuAD training set. The script generates a prediction file in json format that will be stored in the 'data' folder. 
### Evaluation
To evaluate the performance of the trained model run : 
```
python evaluate.py 'path_to_ground_truth' 'path_to_predictions_file'
```
where the predictions file is the one produced by the 'compute_answers.py' script

## Authors 
- [Riccardo de Matteo](https://github.com/riccardodm97) 
- [Marco Cucé](https://github.com/Marco97-exe)
- [Giacomo Berselli](https://github.com/JackBerselli)
