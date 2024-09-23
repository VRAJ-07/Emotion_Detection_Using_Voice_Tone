# Emotion Recognition from Speech

## Overview

This project is focused on emotion recognition from speech using a Long Short-Term Memory (LSTM) neural network. The dataset consists of audio files representing different emotions, and the aim is to classify the audio files into categories such as *happy*, *sad*, *angry*, *fear*, *disgust*, *neutral*, and *ps* (perhaps "pleasant surprise").

The process includes loading the dataset, performing exploratory data analysis (EDA), extracting Mel Frequency Cepstral Coefficients (MFCC) features from audio, and training an LSTM model for classification.

## Project Structure

The project is structured into several key steps:

1. **Loading Dataset**: 
   - The dataset is composed of various speech files categorized by emotions.
   - The paths to audio files and their corresponding emotion labels are extracted and stored in a pandas DataFrame.
  
2. **Exploratory Data Analysis (EDA)**:
   - A count plot of the distribution of emotions in the dataset is plotted using Seaborn to understand the balance of the data.
   - Waveforms and spectrograms of audio files are visualized for each emotion using `librosa`.

3. **Feature Extraction**:
   - For each audio file, MFCC (Mel Frequency Cepstral Coefficients) features are extracted. MFCCs are widely used for speech analysis as they effectively capture the characteristics of the sound.
   - These features are stored and will be used as input to the LSTM model.

4. **Modeling**:
   - An LSTM model is created using Keras. The LSTM architecture is well-suited for sequence data like audio signals.
   - The model consists of several layers: LSTM, Dropout, and Dense layers, followed by a softmax layer for classification.
   - The model is trained using categorical cross-entropy loss and Adam optimizer. 

5. **Training and Evaluation**:
   - The model is trained over several epochs, with validation data being used to monitor accuracy and loss. Metrics like validation accuracy are tracked to assess the model's performance.

## Key Steps

### 1. Dataset Loading
- The audio dataset is loaded, and the paths and emotion labels are extracted and stored in a DataFrame for further processing.

### 2. Exploratory Data Analysis (EDA)
- The dataset is analyzed visually by plotting the count of samples per emotion category.
- The waveforms and spectrograms for a few example audio files are plotted to explore the differences between emotions.

### 3. Feature Extraction
- MFCC features are extracted from each audio file to serve as input features for the LSTM model. The MFCC features capture essential sound characteristics.
- These features are stored in a NumPy array.

### 4. Model Creation
- An LSTM model is built using Keras with the following structure:
  - LSTM layer with 256 units
  - Dropout layers to prevent overfitting
  - Dense layers for classification
  - Softmax output layer for multi-class emotion classification

### 5. Training the Model
- The model is trained for 50 epochs with a validation split of 20%.
- The training process includes monitoring both training and validation accuracy and loss.
  
## Results
- The model achieves high accuracy during training and validation. Accuracy can be further improved by tuning hyperparameters or experimenting with the architecture.

## Dependencies

- Python 3.x
- pandas
- numpy
- seaborn
- matplotlib
- librosa
- IPython
- Keras
- TensorFlow
- scikit-learn

## How to Run

1. **Install the dependencies**:
   You can install the required libraries using pip:
   ```bash
   pip install pandas numpy seaborn matplotlib librosa tensorflow scikit-learn
   ```

2. **Prepare the Dataset**:
   - Ensure the dataset is available in the correct format, with audio files organized by emotion.
   - Update the script to point to the correct directory for the dataset.

3. **Run the Script**:
   - Execute the script to load the dataset, extract features, and train the model.

4. **Model Training**:
   - The model will be trained for 50 epochs. You can modify the batch size, number of epochs, or architecture for experimentation.

## Future Work

- Experiment with different audio features (e.g., Chroma, Mel-spectrogram) to improve performance.
- Implement data augmentation techniques to handle imbalanced data.
- Deploy the trained model as a web service or mobile app for real-time emotion recognition.
