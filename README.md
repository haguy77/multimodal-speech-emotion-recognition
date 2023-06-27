# Multimodal Speech Emotion Recognition and Ambiguity Resolution

## Overview
Identifying emotion from speech is a non-trivial task pertaining to the ambiguous definition of emotion itself. In this work, we build light-weight multimodal machine learning models and compare it against the heavier and less interpretable deep learning counterparts. For both types of models, we use hand-crafted features from a given audio signal. Our experiments show that the light-weight models are comparable to the deep learning baselines and even outperform them in some cases, achieving state-of-the-art performance on the IEMOCAP dataset.

The hand-crafted feature vectors obtained are used to train two types of models:

1. ML-based: Logistic Regression, SVMs, Random Forest, eXtreme Gradient Boosting and Multinomial Naive-Bayes.
2. DL-based: Multi-Layer Perceptron, LSTM Classifier

This project was carried by Haguy Idelchik & Avihay Peretz as a course project for the course Natural Language Processing taught by Dr. Nava Shaked & Mr. Yuri Yurchenko at the Holon Institue of Technology. For a more detailed explanation, please check the [report](https://arxiv.org/abs/1904.06022).

## Datasets
The [IEMOCAP](https://link.springer.com/content/pdf/10.1007%2Fs10579-008-9076-6.pdf) dataset was used for all the experiments in this work. Please refer to the [report](https://arxiv.org/abs/1904.06022) for a detailed explanation of pre-processing steps applied to the dataset.
For our project we acquired the IEMOCAP full release dataset from [Kaggle](https://www.kaggle.com/datasets/dejolilandry/iemocapfullrelease).

## Requirements
All the experiments have been tested using the following libraries:
- xgboost==0.82
- torch==1.0.1.post2
- scikit-learn==0.20.3
- numpy==1.16.2
- jupyter==1.0.0
- pandas==0.24.1
- librosa==0.7.0

To avoid conflicts, it is recommended to setup a new python virtual environment to install these libraries. Once the env is setup, run `pip install -r requirements.txt` to install the dependencies.

## Instructions to run the code
1. Clone this repository by running `git clone git@github.com:haguy77/multimodal-speech-emotion-recognition`.
2. Go to the root directory of this project by running `cd multimodal-speech-emotion-recognition/` in your terminal.
3. Start a jupyter notebook by running `jupyter notebook` from the root of this project.
4. Run `1_extract_emotion_labels.ipynb` to extract labels from transriptions and compile other required data into a csv.
5. Run `2_build_audio_vectors.ipynb` to build vectors from the original wav files and save into a pickle file
6. Run `3_extract_audio_features.ipynb` to extract 8-dimensional audio feature vectors for the audio vectors
7. Run `4_prepare_data.ipynb` to preprocess and prepare audio + video data for experiments
8. It is recommended to train `LSTMClassifier` before running any other experiments for easy comparsion with other models later on:
  - Change `config.py` for any of the experiment settings. For instance, if you want to train a speech2emotion classifier, make necessary changes to `lstm_classifier/s2e/config.py`. Similar procedure follows for training text2emotion (`t2e`) and text+speech2emotion (`combined`) classifiers.
  - Run `python lstm_classifier.py` from `lstm_classifier/{exp_mode}` to train an LSTM classifier for the respective experiment mode (possible values of `exp_mode: s2e/t2e/combined`)
9. Run `5_audio_classification.ipynb` to train ML classifiers for audio
10. Run `5.1_sentence_classification.ipynb` to train ML classifiers for text
11. Run `5.2_combined_classification.ipynb` to train ML classifiers for audio+text

**Note:** Make sure to include correct model paths in the notebooks as not everything is relative right now and it needs some refactoring

## Results
Accuracy, F-score, Precision and Recall has been reported for the different experiments.

**Audio**

Models | Accuracy | F1 | Precision | Recall
---|---|---|---|---
RF | **93.3** | 59.7 | 62.1 | 59.5
XGB | 92.7 | 58.7 | 61.5 | 57.4
SVM | 67.2 | 15.3 | 33.3 | 17.6
MNB | 66.6 | 13.8 | 43.8 | 16.9
LR | 67.1 | 15.7 | 32.5 | 17.9
MLP | 84.6 | 39.7 | 57.1 | 36.8
LSTM | 66.4 | 13.3 | 11.1 | 16.7
ARE (4-class) | 56.3 | - | 54.6 | -
E1 (4-class) | 56.2 | 45.9 | 67.6 | 48.9
E1 | 92.2 | 57.7 | 64.2 | 54.8
**WAV2VEC2** (4-class) | 73.8 | **74.4** | **76.4** | **81.1**

E1: Ensemble (RF + XGB + MLP)

**Text**

Models | Accuracy | F1 | Precision | Recall
---|---|---|---|---
RF | 90.5 | 67.1 | 75.2 | 63.0
XGB | 54.1 | 47.8 | 75.8 | 40.9
SVM | 90.5 | 68.6 | 72.9 | 65.2
MNB | 89.1 | 62.6 | 73.9 | 57.4
LR | 89.6 | 64.6 | 74.7 | 59.1
**MLP** | **90.6** | 68.7 | 72.5 | **65.9**
LSTM | 67.1 | 13.4 | 11.2 | 16.7
TRE (4-class) | **65.5** | - | 63.5 | -
E1 (4-class) | 63.1 | 61.4 | **67.7** | 59.0
**E2** | **90.6** | **69.0** | **76.0** | 64.6
DISTILROBERTA | 20 | 21.7 | 31.6 | 31.6

E2: Ensemble (RF + XGB + MLP + MNB + LR)
E1: Ensemble (RF + XGB + MLP)

**Audio + Text**

Models | Accuracy | F1 | Precision | Recall
---|---|---|---|---
RF | 94.6 | 68.0 | 72.7 | 66.7
XGB | 91.5 | 57.7 | 72.0 | 52.6
SVM | 90.0 | 67.9 | 72.9 | 64.2
MNB | 88.6 | 62.4 | 74.5 | 57.0
MLP | 95.0 | 73.0 | 74.2 | 72.0
LR | 89.2 | 64.9 | 75.1 | 59.2
LSTM | 90.2 | 68.3 | 71.9 | 65.5
MDRE (4-class) | 75.3 | - | 71.8 | -
E1 (4-class) | 70.3 | 67.5 | 73.2 | 65.5
E2 | 94.8 | 73.3 | 78.0 | 70.1
**LF1** | 95.8 | 79.0 | **87.2** | 74.3
**LF2** | **96.1** | **79.4** | 86.6 | **75.5**

LF1: Late Fusion Approach 1 (Audio E1 + Text E2)
LF2: Late Fusion Approach 2 (Audio RF + Text E2)

For more details, please refer to the [report](https://arxiv.org/abs/1904.06022), and to the [survey article](https://www.sciencedirect.com/science/article/pii/S0167639322000413/pdfft?md5=fe042a81eb4b13f10d5f48f13b03209b&pid=1-s2.0-S0167639322000413-main.pdf)

## Citation
If you find this work useful, please cite:

```
@article{sahu2019multimodal,
  title={Multimodal Speech Emotion Recognition and Ambiguity Resolution},
  author={Sahu, Gaurav},
  journal={arXiv preprint arXiv:1904.06022},
  year={2019}
}
```

```
@article{atmaja2022surveyonbimodal,
  title={Survey on Bimodal Speech Emotion Recognition from Acoustic and Linguistic Information Fusion},
  author={Atmaja, Bagus Tris, Akira Sasou, Masato Akagi},
  journal={Speech Communication},
  year={2022}
}
```
