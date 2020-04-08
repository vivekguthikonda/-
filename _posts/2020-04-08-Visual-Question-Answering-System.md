---
published: false
---

![]({{ "/assets/img/vqa/int.jpg" | relative_url}})

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




























