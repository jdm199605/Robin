##Overview
This is the repository of source code for paper `Robust Table Integration in Data Lakes: From Integrable Set
Discovery to Multi-Tuple Conflict Resolution`.  
Authors: Daomin JI, Hui Luo, Zhifeng Bao, J. Shane Culpepper

##Benchmarks
For the used benchmarks, please visit [this link](https://zenodo.org/record/8313382） to download our benchmarks. The whole benmarks are created based on two data pools, `Real` and `Join`, from a VLDB 23 Paper, `Integrating Data Lakes`. For more information about this paper, please visit [ALITE](https://github.com/northeastern-datalab/alite).  
`Real` has 11 datasets and `Join` has 28 datasets. We shortly term those datasets as R1-R11 (from `Real`) and J1-J28 (fro `Join`). In original benchmarks, each dataset has several input tables to be integrated. In our paper, those input tables are first integrated into a single table T by outer-union as input of our tasks. 
###How to use our Benchmarks
The benchmarks are divided into two tasks, `integrable set discovery` and `multi-tuple conflict resolution`.
####Integrable dataset discovery
According to the way to inject noises, the benmarks are divided into three cases, balanced noises, TE-heavy, and SE-heavy. More information about the setting can be refered to Sec. 6.1 in our paper.
Each dataset has two files, a csv file representing the input table T, and a txt file representing the ground-truth indicating all the integrable datasets in T.
The ground-truth file has several rows, each of which corresponding to an integrable set that contains the IDs about the tuples in T.
####Conflict Resolution
We use a json file for each dataset, which contains three kinds of information:  
* The incompleting information of a resulting tuple, which contains the non-missing attribute information (Please refer to Sec. 5 in our paper for more information).  
* Conflicted Attribute, which is the attribute we need to fill
* Candidate values, which contains a set of candidate values.
* Ground-truth, which is the right answer from the ground-truth.
* 
##Methods
We have implemented two main methods in this paper, `SSACL` and `ICLCF`, for the two tasks, respectively, with an additional method using DFS algorithm to find integrable sets (Please refer to Sec. 4 for more information).
## How to run the codes
### Dependency
At first, please run `pip install requirements.txt` to install all the packages required for the models.
### SSACL
Go to SSACL directory, run `python train.py` to train SSACL with default parameter settings. You also can specify the following key parameters as well as other common parameters (not listed in the following), such as learning rate and batch size.  
* pos: the number of positive instances
* neg: the number of negative instances
run `Eval.py` to use the trained SSACL model to predict the integrability for any tuples in table T on a certain dataset. Then a DFS algorithm will automatically run to get all the integrable sets. This task is evaluted by F1 and similarity (Please refer to Sec. 6.2 for more information).
### ICLCF
Go to ICLCF directory, since this is a train-free method, you can directly run `python eval.py` to make prediction on a specific dataset. The evalution metric is accuracy (Please refer to Sec. 6.2 for  more informtation).
## Contact
Thanks for your interest in this work. If you have any questions, please do not hesitate to send an email to `daomin.ji@student.rmit.edu.au`. 