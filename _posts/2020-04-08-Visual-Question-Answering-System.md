---
published: true
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
![]({{ "/assets/img/vqa/vqadataset.jpg" | relative_url}})

###### Dataloader:
![]({{ "/assets/img/vqa/dataloader.jpg" | relative_url}})

### Deep Learning Models:

##### DeeperLSTM-Q + norm I(arXiv:1505.00468 [cs.CL]):
Now we are going to implement the same model from paper in pytorch framework.

![]({{ "/assets/img/vqa/model1.jpg" | relative_url}})

###### Model Definition:
In this model, we embed question to 300 dim vectors and pass it to fully connected layer. This embeded question is sent to 2 layered LSTM which outputs 512d tensor. The obtained 2 hidden and 2 cell states tensors are concatenated to form 2048 tensor. This question tensor is passed through fullyconnected layer which outputs final question feature tensor. The image features are l2 normalized are passed through another dense layer to get final image features. 

These both question and image features are point wise multiplied and passed through 2 layered MLP with dropout 0.5.
Lastly we add softmax layer top of them.

Most weights in the model are initialized using he_normal weight initialization. GELU is used as activation function because its known to work well with NLP problems.

![]({{ "/assets/img/vqa/model1def.jpg" | relative_url}})

Now, we will initilize model by adding optimizer, onecycle_scheduler.
- NLLLoss is used because we have already used softmax in last layer.
- Adamax optimizer is used with initial lr = 0.001
- onecyclic scheduler with max_lr = 0.005 for 20 epochs.

![]({{ "/assets/img/vqa/model1def1.jpg" | relative_url}})

Now, the model is defined, next we will train it.

###### Model training:

In pytorch, for training, we use model.train() to enable training mode. optimizer.zero_grad() must be used before feeding the batch to model. As we have to check answer with multiple answers, here we modify the answer tensor by replacing the wrong output answers with possible multi answers and find loss. After that backpropagate the loss using loss.backward() followed by optimizer.step() and onecycle_scheduler.step().

Model is earlystopped after the loss decrease for 2 epochs.
![]({{ "/assets/img/vqa/model1def2.jpg" | relative_url}})

###### Plots obtained after training:

![]({{ "/assets/img/vqa/plot1.jpg" | relative_url}})


##### DeeperLSTM-Q + norm I with attention(arXiv:1704.03162 [cs.CV]):

We will add attention mechanism from paper to the above model.

![]({{ "/assets/img/vqa/att.jpg" | relative_url}})

###### Model Definition:
In this model, we embed question to 512 dim vectors and pass it to fully connected layer. This embeded question is sent to 2 layered LSTM which outputs 1024d tensor.  The obtained last cell state tensor of 1024d is passed to a dense layer to  get final question feature. That question feature is attended to image feature of dimension 1024 x 14 x 14. The obtained weighted image is concatenated with question feature and passed through 2 layered MLP with softmax layer at last.

![]({{ "/assets/img/vqa/model2def1.jpg" | relative_url}})

###### Attention definition:

Here in attention, we pass 1024x14x14 image feature to 1x1 kernel sized conv2d layer which outputs 1024x14x14. Question feature is expanded as 1024d to 1024x14x14 repeating in all spatial dimensions similar to image features. Now, we will add both obtained image and question features. This added feature sent to glimpse convolution layer which looks those features for 2 glimpses and outputs 2x14x14 feature tensor which is our attention tensor. Now, we will multiply this attention tensor with image features for 2 glimpses. The obtained tensor is averaged across 14x14 and finally we will get 2048d tensor.

![]({{ "/assets/img/vqa/model2def2.jpg" | relative_url}})

###### Model training:
Similarly to model1, we also train this model.

###### Plots obtained after training:
![]({{ "/assets/img/vqa/plot2.jpg" | relative_url}})

##### Hierarchical Question Image Co-Attention Model(arXiv:1606.00061 [cs.CV]):

Here, we will implement the hierarchical co attention model from paper[here](https://arxiv.org/pdf/1606.00061).
![]({{ "/assets/img/vqa/model3.jpg" | relative_url}})

###### Model Definition:

First for embedding question, we will get unigrams, bigrams, traigrams using respective convolution layers and maxpool them. This maxpooled feature is passed to 3 layered LSTM. 

Now, We will define layers as per the paper.
![]({{ "/assets/img/vqa/model3def1.jpg" | relative_url}})

The forward function of model will look:
![]({{ "/assets/img/vqa/model3def2.jpg" | relative_url}})

Parallel co-attention:
The paper formulated co-attention mechanism as follow:
![]({{ "/assets/img/vqa/model3form.jpg" | relative_url}})
The implementation of mechanism will look like:
![]({{ "/assets/img/vqa/model3def3.jpg" | relative_url}})

###### Model Training:

Now, we will initilize model by adding optimizer, onecycle_scheduler.
- NLLLoss is used because we have already used softmax in last layer.
- Adamax optimizer is used with initial lr = 0.0003
- onecyclic scheduler with max_lr = 0.0012 for 50 epochs.
![]({{ "/assets/img/vqa/model3def4.jpg" | relative_url}})

This model also trained similar to above two models.

###### Plots obtained after training:
![]({{ "/assets/img/vqa/plot3.jpg" | relative_url}})


#### Using Pretrained embeddings for word embeddings(for model 2):
###### GloVe Embeddings:

We extracted question vocab features from glove.42B.300d model availble online and stored the vectors in pickle and loaded the 
weights in embedding layer. The question are encoded similar to previous models but with new glove vocab.
![]({{ "/assets/img/vqa/glove.jpg" | relative_url}})

###### Bert Embeddings:

Here, we tokenize the question using BertTokenizer and encode it as following:

![]({{ "/assets/img/vqa/bert1.jpg" | relative_url}})

The Embedding layer with loaded with bert embedding weights as follows:

![]({{ "/assets/img/vqa/bert2.jpg" | relative_url}})

The extra 2 models are trained using these glove and bert embeddings and results are noted.

### Comparing All Models:

#### VQADemo:
Lets compare with an image and set of questions:
###### Image:
![]({{ "/assets/img/vqa/demo_img.jpg" | relative_url}})
###### Question:
![]({{ "/assets/img/vqa/demo.jpg" | relative_url}})

#### Metric Comparision:
![]({{ "/assets/img/vqa/comp.jpg" | relative_url}})

By seeing vqademo and metric comparision, we can say model2 with glove embedding performed better than rest.

### Future Work:

- As future work, Hierarchical co attention model should be trained for more epochs as I found that it is not that much overfitted as other models.
- There are many complex deep learning models in internet which perform much better than these. So, we can try implementing and tuning them.

### Epilogue:

In this case study, we have trained, tuned and compared different deep learning models based for this visual question answering problem. All models are implemented in pytorch framework. 

The total code and case study is available at my github: [here](https://github.com/vivekguthikonda/VisualQA/blob/master/VQA_notebook.ipynb)

My linkedin profile : [here](https://www.linkedin.com/in/vivek-guthikonda-a08074173?lipi=urn%3Ali%3Apage%3Ad_flagship3_profile_view_base_contact_details%3Buzmi4oz%2BSeaFV8F0MiWfVA%3D%3D)

### References:

- https://arxiv.org/pdf/1505.00468
- https://arxiv.org/abs/1704.03162
- https://arxiv.org/abs/1606.00061
-https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html#sphx-glr-beginner-blitz-cifar10-tutorial-py
