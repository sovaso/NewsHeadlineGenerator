# News_Headline_Generator
NLP project to generate headline based on a new using text summarization technique and Tranformer model (paper of Transformer model)

## Required Libraries and frameworks
* Tensorflow 2.4.1
* NumPy 1.19.2
* Pandas 1.1.3
* Flask 1.1.2
* Flask-Cors 3.0.10
* Angular 8

## Dataset
Dataset (Inshorts dataset -> [link](https://www.kaggle.com/shashichander009/inshorts-news-data)) consists of about 55000 pairs of news and their headlines for the training and test purposes. This dataset is split on train and test datasets in ratio 99:1.

## How to train
Navigate to the `News_Headline_Generator/backend/code` and then run `python train_main.py` in terminal

## How to run
Backend: Navigate to the `News_Headline_Generator/backend/code`and then run `python train_main.py` in terminal and it will run on http://127.0.0.1:5000/ <br>
Frontend: Navigate to the `News_Headline_Generator/frontend/news-headline-generator`and then run `ng serve` in terminal and it will run on http://127.0.0.1:4200/

## Checkpoint download
Link to download checkpoints for the model can be found >> [here](https://e.pcloud.link/publink/show?code=kZYK4XZHUze0rPof4V1ggCSYvKwkfFJKvsV) << <br>
When you download it make sure to put your checkpoints folder in `News_Headline_Generator/backend`
