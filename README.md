# Stance Detection baseline

This is a baseline implementation of stance detection for the [Fake News Challenge](http://www.fakenewschallenge.org/).

The initial objective of this project is to develop a natural language processing model to classify two bodies of text (a headline and article body) as **related** or **unrelated**. The model will use a Naive Bayes' classifier.

## Requirements
Python 2.7

## Installation
Make sure to use the Python 2.7 version of all commands (`python` and `pip`)
```bash
git clone git@github.com:StanceDetection/stance_detection.git \
&& cd stance_detection \
&& pip install -r requirements.txt
```

## Running
```bash
python stance_detection.py
```
