# FacialEmotionRecognition_FER2013
Detects and classifies emotions using facial images. The goal is to, with the use of Computer Vision and deep learning algorithms, analyze facial expressions and predict emotional states with respect to this.  Model is trained and evaluated on a dataset (FER-2013) which includes facial images categorized into different emotions

The FER-2013 dataset was used to train the algorithm and test the algorithm.
This dataset consists of 48x48 grayscale images of faces. These faces are automatically registered, so the face is more or less centered and takes up similar space.
FER-2013 holds two folders, one for train, and one for test, each folder has further sub-folders named after every emotion.
(Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral)

The training set includes 28,709 examples.
The testing set includes 3,589 examples.
Total of 35,887 images.
It is split into 80% for training, 10% for validation, and 10% for testing.

This is highly influential in the field and has been used in many research studies to compare the performance of different facial emotion recognition algorithms and models.

With its challenges, such as dealing with the illumination, face orientation, and quality of the image.

By combining the sequential structure, convolutional layers, pooling layers, and fully connected layers, the model learns to extract relevant spatial and temporal information from the input images. This allows the model to understand and differentiate between different facial expressions associated with various emotions, enabling it to classify the input images into the appropriate emotion categories.

The model architecture is designed to leverage the power of deep learning to automatically learn and extract features from the images, rather than relying on handcrafted features. This enables the model to adapt and generalize well to different facial expressions and emotions, making it effective for emotion detection tasks.


In conclusion, the computer vision emotion detector project successfully developed a deep learning model using Convolutional Neural Networks (CNNs) to recognize emotions from facial expressions. The model underwent hyperparameter tuning and achieved decent accuracy in emotion classification. However, there are limitations related to lighting, pose, and image quality. Future improvements could explore advanced techniques like RNNs or Transformers to capture temporal dependencies and enhance emotion recognition in sequential data. 
Overall, the computer vision emotion detector semester project provided valuable insights into the application of deep learning techniques for emotion recognition from facial expressions. It served as a foundation for further research and development in the field of affective computing, with the potential to contribute to various domains, including human-computer interaction, psychology, and healthcare.
