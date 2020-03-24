---
published: true
---
![intro]({{ "/assets/img/mpst/image.jpg" | relative_url}})

### Introduction:

Social Tagging of movies reveals a wide range of heterogeneous information about movies, like the genre, plot structure, soundtracks, metadata, visual and emotional experiences. Such information can be valuable in building automatic systems to create tags for movies. Automatic tagging systems can help recommendation engines to improve the retrieval of similar movies as well as help viewers to know what to expect from a movie in advance. 

In this case study, we are using MPST dataset of 14k movie plot synopses which has 71 tagset describing multi-label association with synopses. The dataset is hosted by cryptexcode in kaggle datasets page is [here](https://www.kaggle.com/cryptexcode/mpst-movie-plot-synopses-with-tags) and the paper related to the dataset is [here](https://www.aclweb.org/anthology/L18-1274).

### Problem Statement:

Suggest the tags based on the plot synopses of movies given.

### Dataset:

Contains IMDB id, title, plot synopsis, tags for the movies. There are 14,828 movies' data in total. The split column indicates where the data instance resides in the Train/Dev/Test split.

![image2]({{ "/assets/img/mpst/image2.jpg" | relative_url}})


### Real world Objectives and Constraints:
- Predict as many tags as possible with high precision and recall.
- Incorrect tags could impact movie search results generated based on tags.
- No strict latency constraints.

### Mapping the problem to Machine Learning problem:

##### Type of Machine Learning Problem:
It is a multi-label classification problem
Multi-label Classification: Multilabel classification assigns to each sample a set of target labels. This can be thought as predicting properties of a data-point that are not mutually exclusive, such as topics that are relevant for a document. A movie on MPST dataset might be about any of horror, comedy, romantic etc. at the same time or none of these.

### Performance Metric:

Micro-Averaged F1-Score (Mean F Score) : The F1 score can be interpreted as a weighted average of the precision and recall, where an F1 score reaches its best value at 1 and worst score at 0. The relative contribution of precision and recall to the F1 score are equal. 

![map]({{ "/assets/img/mpst/map.jpg" | relative_url}})
![mar]({{ "/assets/img/mpst/mar.jpg" | relative_url}})
![maf]({{ "/assets/img/mpst/maf.jpg" | relative_url}})

### Data Cleaning:

Any dataset at first will contain many nan or null entries and duplicate items. So, we will remove them now.

##### Checking for NaN or null entries:
![nan]({{ "/assets/img/mpst/checking_nan.jpg" | relative_url}})
So, the dataset has no null/NaN entries.

##### Checking for duplicates:
![dup]({{ "/assets/img/mpst/duplicates.jpg" | relative_url}})

Let's see what are those duplicate rows!

![dup_head]({{ "/assets/img/mpst/duplicate_head.jpg" | relative_url}})

Now, lets see how those rows are duplicated.

![dup_1]({{ "/assets/img/mpst/duplicate1.jpg" | relative_url}})

The titles appeared to be same but the plot and tags are different. So its time to check plot synopsis duplication.

![dup_2]({{ "/assets/img/mpst/duplicate2.jpg" | relative_url}})

Now, we have to remove the duplicate rows which have both 'title' and 'plot_synopsis' as same.

![dup_3]({{ "/assets/img/mpst/duplicate3.jpg" | relative_url}})

Thats it. We have done data cleaning of the dataset.

##### Preprocessing tags:
- We split the tags at whitespace and remove the commas.
- Replace spaces between each tag with '_'.
- Encode tags using one hot encoding.

![tagpp]({{ "/assets/img/mpst/tag_pp.jpg" | relative_url}})

### Exploratory Data Analysis:

To get the clarity on the distribution of the dataset, we have to plot the data using visualization tools available.

##### Distribution of Tags:

Lets plot the distribution of tags using barplot from seaborn library.

![tag_code]({{ "/assets/img/mpst/tag_code.jpg" | relative_url}})

and the output is:

![tag_dist]({{ "/assets/img/mpst/tag_dist.png" | relative_url}})

By looking at the plot, we can say that data is very imbalanced.

##### WordCloud of Most Frequent Tags:

![wordcloud]({{ "/assets/img/mpst/wordcloud.png" | relative_url}})

### Text Preprocessing:

##### Preprocessing the plot_synopses:

- Remove name tags like Dr., Mr., Mrs., Miss, Master, etc.
- Remove stopwords.
- Remove special characters.

![pre1]({{ "/assets/img/mpst/preprocessing1.jpg" | relative_url}})

- Replace all person names as 'person'.
- Convert every word to lowercase.

![pre11]({{ "/assets/img/mpst/preprocessing11.jpg" | relative_url}})

![pre2]({{ "/assets/img/mpst/preprocessing2.jpg" | relative_url}})

- While preprocessing itself, we find the sentiment features using sentic.SenticPhrase library which include: 14 mood tags features like ['#interest', '#admiration', '#sadness', '#disgust', '#joy', '#anger', '#fear', '#surprise'] and 3 basic sentiments like negative, neutral, positive intensity features from SentimentIntensityAnalyzer in nltk.sentiment.vader library.

- Stem all the words using krovetzstemmer.

![pre3]({{ "/assets/img/mpst/preprocessing3.jpg" | relative_url}})

### Data Preparation:

For the data to be ready for training, we need to encode numericals to one hot vectors and drop the unneccesary columns like 'title' and 'imdb_id'.

![dp]({{ "/assets/img/mpst/data_prep.jpg" | relative_url}})

##### Train-Test split:

For the validating our models, we train them on train split and validate them using test split.

![split]({{ "/assets/img/mpst/train_test.jpg" | relative_url}})

##### Text Featurization:

- All features are extracted using scikit-learn libraries and MinMax Normalized before training.

- BoW (Bag Of Words) features: we use max_features = 25000 (found this optimal value which works better with 71 tags) and ngram_range = (1,5).

![bow]({{ "/assets/img/mpst/bow.jpg" | relative_url}})
									*Image: Bag of words*

- Tfidf features: we use min_df = 5 (min. document frequency for a word), sublinear_tf = True (option which normalizes the features), max_features = 25000 and ngram_range = (1,5).

![tfidf]({{ "/assets/img/mpst/tfidf.jpg" | relative_url}})
									*Image: Tfidf*

- pretrained-Glove average word2vec features (300dim) and tfidf weighted word2vec features.

![w2v]({{ "/assets/img/mpst/w2v.jpg" | relative_url}})
									*Image: Glove avg-w2v*


### Machine Learning Models:

We define various machine learning models like LogisticRegression, LinearSVM, Complement Naive Bayes(As it known for handling unbalanced numerical count features) inside of OneVsRestClassifier which trains a specified model for every label present in tagset.

NOTE: set class_weight = 'balanced' for models where ever available in libraries.

![lr]({{ "/assets/img/mpst/lrmodel.jpg" | relative_url}})
							*Image: Logistic Regression (Tfidf features)*

##### Training the models:
- We train models using sentiment features and taking one of above text featurizations seperately for each model.
- We also train models using mix of all above featurizations.
- I have tuned the parameters manually so that I can test wide range of parameters by hand.
- Also train the models with Top3 and Top4 tags as Movie databases show mostly 3 or 4 tags with movie.

![lr]({{ "/assets/img/mpst/lrmodel.jpg" | relative_url}})
							*Image: Logistic Regression (Mixed Features)*


#### Comparing all models:

##### On 71 tags:

![71tags]({{ "/assets/img/mpst/71tags.jpg" | relative_url}})

##### On Top3 and Top4 tags:

![34tags]({{ "/assets/img/mpst/34tags.jpg" | relative_url}})

#### Future Work:

- As future work, we can engineer more features from text and try to increase the micro-f1 score.

- As the text preprocessing plays an important role in getting rich features, we can preprocess the text in more creative ways to extract more performance from models.

- I have also tested the models with embeddings of the plot_synopsis obtained from pretrained Deep learning models like BERT(786d), Elmo (1024d), etc. but the dimensionality of those vectors is not sufficient to classify this 71 tags tagset. If possible, we can try to optimize those features for more score.

#### Epilogue:

- we have covered many of the Machine learning technique in classifying the text data. 

We have used MPST dataset which has 14,828 plot synopses and with 71 tagset. We have proprocessed tags and plot_synopses extracting useful features. We have analysed various machine learning models trained upon those features. 

The best performance is shown by LogisticRegression with all w2v, bow and tfidf features together giving **0.4101** f1-micro for 71 tags.

The performance of Top3 Tags LR model is **0.6102** f1-micro.

The performance of Top4 Tags LR model is **0.5874** f1-micro.

The link to the notebook of this case study at github: [here](https://github.com/vivekguthikonda/MPST/blob/master/MPST.ipynb)


#### References:

- [www.aclweb.org/anthology/L18-1274](www.aclweb.org/anthology/L18-1274)
- [www.appliedaicourse.com](www.appliedaicourse.com)
