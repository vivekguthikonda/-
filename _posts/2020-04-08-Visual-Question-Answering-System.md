---
published: false
---

![]({{ "/assets/img/vqa/int.jpg" | relative_url}})

In this case study, we will compare various deep learning models which perform visual question answering.

### Introduction:

VQA is a task of free-form and open-ended Visual Question Answering (VQA). Given an image and a natural language question about the image, the task is to provide an accurate natural language answer. Mirroring real-world scenarios, such as helping the visually impaired, both the questions and answers are open-ended. Visual questions selectively target different areas of an image, including background details and underlying context. As a result, a system that succeeds at VQA typically needs a more detailed understanding of the image and complex reasoning than a system producing generic image captions. 

Moreover, VQA is amenable to automatic evaluation, since many open-ended answers contain only a few words or a closed set of answers that can be provided in a multiple-choice format. They have provided a dataset containing 0.25M images, 0.76M questions, and 10M answers (www.visualqa.org).

### Problem Statement:

Given the image and the question related to it, output the answer with decent accuracy. (Using top most 1000 answer labels and filtering the questions accordingly)

### Dataset:

Dataset source : www.visualqa.org/download

2017 dataset:

Training annotations 2017 v2.0* : 4,437,570 answers Validation annotations 2017 v2.0* : 2,143,540 answers.

Training questions 2017 v2.0* : 443,757 questions Validation questions 2017 v2.0* : 214,354 questions.

Data format: Image folders containing .jpg images and json files consisting questions and answers.

### Real world Objectives and constraints
- Predict answers with decent accuracy.
- Output should be given within few seconds.

### Mapping the problem to Deep Learning Problem:
#### Type of Problem:
It is a multi class classification problem using deep learning and computer vision techniques, which outputs only one label of class at end in form of answer in english.

#### Performance Metric:
Accuracy (Multi-Choice) : Its the ratio of no. of correct answers predicted to no. of total predictions. Multi-Choice is, any value predicted should be in the multichoice answers for that question .
Accuracy = no. of correct predictions / no. of total predictions.


### Data Cleaning:

After downloading the dataset, we will have .json files like v2_mscoco_train2014_annotations.json, v2_mscoco_val2014_annotations.json, v2_OpenEnded_mscoco_train2014_questions.json,
v2_OpenEnded_mscoco_val2014_questions.json. Load those json files into json lists using json.load() and extract the required columns like image_id, question, question_id, multiple_choice_answer, answers and answer_type for both train and test sets.

![]({{ "/assets/img/vqa/data_c1.jpg" | relative_url}})
![]({{ "/assets/img/vqa/data_c2.jpg" | relative_url}})

### Exploratory Data Analysis:


  Every train sample and val sample has 
  - image_id says about the file name where image is located,
  - question,
  - answer_type like yes/no, number and other,
  - one open end answer,
  - upto 10 multi choice answers.
 
  Now, Lets see the distribution of both train and val datasets to find out whether they have same distributions or not.
 
![]({{ "/assets/img/vqa/disbc.jpg" | relative_url}})

The below image has distributions for both train and val sets:
![]({{ "/assets/img/vqa/disb.jpg" | relative_url}})

By seeing the image, we can say that both train and val sets datasets are very imbalance but they have similar distribution which means we can val set to validate the models.

Now lets see maximum words that a question has:
![]({{ "/assets/img/vqa/length.jpg" | relative_url}})

Since the maximum words in question are 22, we will take max length of 25 words for every question.

### Data Preparation:

#### Image Features:
In this case study, we are going to compare three architectures from research papers arXiv:1505.00468 [cs.CL] , arXiv:1704.03162 [cs.CV], arXiv:1606.00061 [cs.CV].
So, first we will extract the image features from pretrained models and store them in .h5 files for easy retrieval while training models.
![]({{ "/assets/img/vqa/img_feats.jpg" | relative_url}})

#### Questions:

Now we encode questions. Create question vocab by collecting unique words and sorting them in alphabetical order. Also \<PAD\> and \<UNK\> tokens for padding and unknown words. 

![]({{ "/assets/img/vqa/create_vocab.jpg" | relative_url}})

For each question, convert word to id using vocab dict. if the length is below 25, then add padding else trim it.
![]({{ "/assets/img/vqa/encode_qs.jpg" | relative_url}})

#### Answers:

Similarly for answers also, we create answer_vocab by using only 1000 top answers.  
![]({{ "/assets/img/vqa/create_ans_v.jpg" | relative_url}})

Now using the above vocab, we encode answers. First collect all answers from multiple answers and set on them, if a answer is not present in vocab, then replace it with \<UNK\>. For pytorch dataloader the all inputs should have same dimensions. So we will pad multi answers with -11 for remaining out of 10 answers.

![]({{ "/assets/img/vqa/encode_ans.jpg" | relative_url}})

#### Creating pytorch Dataset and DataLoader:

To feed the data to pytoch model, we need a dataloader which uses dataset and transforms it to required batch_sized samples.
Each sample contains image features tensor, encoded question tensor, answer tensor and multi_answer tensor.

###### Dataset:
![]({{ "/assets/img/vqa/vqa_dataset.jpg" | relative_url}})

###### Dataloader:
![]({{ "/assets/img/vqa/dataloader.jpg" | relative_url}})

#### Deep Learning Models:

##### DeeperLSTM-Q + norm I(arXiv:1505.00468 [cs.CL]):
Now we are going to implement the same model from paper in pytorch framework.

![]({{ "/assets/img/vqa/model1.jpg" | relative_url}})

###### Model Definition:
In this model, we embed question to 300 dim vectors and pass it to fully connected layer. This embeded question is sent to 2 layered LSTM which outputs 512d tensor. The obtained 2 hidden and 2 cell states tensors are concatenated to form 2048 tensor. This question tensor is passed through fullyconnected layer which outputs final question feature tensor. The image features are l2 normalized are passed through another dense layer to get final image features. 

These both question and image features are point wise multiplied and passed through 2 layered MLP with dropout 0.5.
Lastly we add softmax layer top of them.

Most weights in the model are initialized using he_normal weight initialization.

![]({{ "/assets/img/vqa/model1def.jpg" | relative_url}})

Now, we will initilize model by adding optimizer, onecycle_scheduler.
- NLLLoss is used because we have already used softmax in last layer.
- Adamax optimizer is used with initial lr = 0.001
- onecyclic scheduler with max_lr = 0.005 for 20 epochs.

![]({{ "/assets/img/vqa/model1def1.jpg" | relative_url}})

Now, the model is defined, next we will train it.

###### Model training:

In pytorch, for training, we using model.train() to enable training mode. optimizer.zero_grad() must be used before feeding the batch to model. As we have to check answer with multiple answers, here we modify the answer tensor by replacing the wrong output answers with possible multi answers and find loss. After that backpropagate the loss using loss.backward() followed by optimizer.step() and onecycle_scheduler.step().

Model is earlystopped after the loss decrease for 2 epochs.
![]({{ "/assets/img/vqa/model1def2.jpg" | relative_url}})

###### Plots obtained after training:

![]({{ "/assets/img/vqa/plot1.jpg" | relative_url}})


##### DeeperLSTM-Q + norm I with attention(arXiv:1704.03162 [cs.CV]):
































  
  




















  
  




























