# Face-Recognition-Using-SiameseNetwork_TripletLoss

<!-- ![Project Image](project_image.png) Replace with your project image if applicable -->

This is a face recognition project that utilizes Siamese Network architecture and Triplet Loss to perform accurate and robust face recognition tasks. Face detection is done by the dnn module present in opencv. The Siamese Network enables us to learn a feature representation for faces and compare the similarity between two face images efficiently. The Triplet Loss function aids in training the network to enforce that the distance between an anchor face and a positive face (same identity) is smaller than the distance between the anchor and a negative face (different identity).

## Features

- Robust face recognition with high accuracy.
- Efficient face comparison using Siamese Network architecture.
- Triplet Loss training for improved performance on face embeddings.

## Table of Contents <!-- Update with your project-specific content -->

- [Getting Started](#getting-started)
- [Usage](#usage)
- [Data preparation](#data-preparation)
- [Training and Evaluation](#training-and-evaluation)
- [Examples](#examples)

## Getting Started

These instructions will help you set up the project and get it running on your local machine.


1. Clone the repository:

```bash
git clone https://github.com/your_username/Face-Recognition-Using-SiameseNetwork_TripletLoss.git
```

2. Navigate to the project directory:
```bash
cd Face-Recognition-Using-SiameseNetwork_TripletLoss
```

3. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Usage
This project can be used as it is, by running 
```bash
py app.py
```
or 
```bash
python app.py
```
or 
```bash
python3 app.py
```
depending upon the python version.

Or the model can be trained again with some tinkering or modification.

For recognition of faces other than the ones already given( imgs of me and some peers :)  ), images with faces can be added using the file upload option in the website.
The name for the images should be in the format "name.jpg".

## Data preparation
The siamese network has been trained on the lfw-dataset for achieving one-shot learning. 
The lfw-dataset has not been included in this repository due to size constraints. It can be added to the artifacts directory if the model is going to be trained again.

## Model architecture
Will be added soon

## Training and Evaluation
Will be added soon

## Examples
Will be added soon

This is an ongoing project, hence will be modified frequently.
