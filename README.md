# landmark-recognition-challenge
Source code used for the google landmark recognition challenge on kaggle [19th place]

# Google landmark recognition challenge (on kaggle)
## Finetuning the Xception CNN with a generalized mean pool (and custom loss function)

### Google landmark recognition challenge

The kaggle competition was about the classification of about 15000 landmark categories from images, see
https://www.kaggle.com/c/landmark-recognition-challenge for details.

Main challenges:

- large number of categories
- imbalanced training set (only a few training images for some categories)
- very hard test set (only a small fraction of images actually depict landmarks)
- GAP (global average precision) metrics (i.e. confidence scores are very important)

### Full solution and validation result

My full solution consists of two stages, the first stage is given by the NN classifier in `landmarks-xception.ipynb`, which proposes a single landmark and a confidence score per image. This resulted in a (public/private) GAP of 0.145/0.149 and would have corresponded to the 34th place. (On an easy dev set comparable to the training data without non-landmark images, this model has a GAP of about 0.96).

As a second step I used Google DELF features (https://arxiv.org/abs/1612.06321) to rescore every image by comparing it to (up to) 32 landmark images of the proposed category, the maximal number of matching features after geometric verification is used as a DELF-based score. The source code for the DELF prediction and analysis was developed based on the examples in the tensorflow repository.

The total confidence for the prediction is then computed by a weighted average of NN and DELF confidence, so that the NN and the DELF confidences contribute roughly half for images with typical DELF-based scores (wheras very high DELF scores dominate the average).

The full model lead to a GAP of (public/private) 0.211/0.192, which resulted in the 19th place (out of 483).

### Finetuning a pretrained Xception-CNN with a generalized mean pool

Here, we finetune a pretrained Xception deep convolutional neural network (input resolution: 299x299) using the keras library with tensorflow backend for the customizations described below. A classifier with 15000 outputs and a sigmoid activation is used as the final layer. The latter has the advantage that the model can more naturally reject non-landmark images.

The model was trained on the about 1,200,000 landamrk images plus about 200,000 non-landmark images from various sources. I first trained the classifier of the `top_model`, and then included some additional layers of the network (see the comment in the code). 

#### Generalized average pool

A generalized average pooling layer has been shown to improve landmark recognition performance (https://arxiv.org/pdf/1711.02512.pdf). The advantage seems to be that the network can better suppress non-relevant features. The exponent p=2.2 was learned during training. 

#### Reweighted loss function

I slightly changed the standard binary cross entropy loss function by sorting the top predictions on each batch and increased the binary crossentropy loss proportional to the rank. This way, wrong predictions with a high confidence are suppressed. This worked well for a toy model, but I could not afford the computational power to compare it to a reference network trained without this feature. Thus, it could well be that this modification has no effect or that it even slowes down learning, but clearly it didn't hurt the overall model performance.

#### batch_GAP

To better supervise training, I implemented a custom metric `batch_GAP`, which calculates the GAP on each batch.

#### 22 crops at prediction

At prediction, I used several (22 for the scores given above) crops of each image and calculated the image category and confidence by a simple voting procedure. This significantly improves performance (by about 10%), because of the large number of output categories and the fact that most of the test images do not depict any landmark (and it is clearly computationally cheaper than training additional models).
