# tf_in_working
This repo is for Practice of Tensorflow codes, the list below contains the basic layout of the all the implementations of the basic tensorflow functions.

This repo also contains the codes of various courses based on tensorflow library such as "Tensorflow in Practice Specialization", as well as my kaggle codes submissions.

### List Of the Folders:
 
## 1) Kaggle: 
   - Titanic-1:  
      A binary classification problem, with basic concepts of normalization by sklearn, data clearning,                     
      getting stated with kaggle competetions,                         
      with almost 72% accuracy.                            

## 2) tf_basic: 
   - basic-intro.ipynb:                                     
      Getting started with tf, a linear regression problem with function like model, compile, fit etc.
      
   - fashion-minist-dataset.ipynb :  
      Importing inbuild fashion-minist-dataset of tensorflow, a multi-label classification problem, with callback when accuracy fall below the certain value.
      
   - stop-at-limit-callback.ipynb:        
     Stoping the epoch if the accuracy or loss reach certain limit.
     
   - reducing_overfit.ipynb:               
     Comparision of tiny, small, medium and large models                 
     then imporving large model with **L2 regularization and Dropout**                
     Used **Early stopping as callback**
     
   - regression.ipynb
     
## 3) nlp: 
   - imdb-review:         
       i) pre_trained_corpus.ipynb - pre trained subwords of the reviews and there position in subword dict as a token, using these tokens to generate sequences and the sending to the embedding layer and GlobalAveragePooling1D layer 
       
       ii) review.ipynb - imbd dataset in colab, concepts of tokenizers, pad_sequenc and how to download file from a google colab, generating dimentional projection files "vecs.tsv" and "meta.tsv" for projecting it.
   - layers:                 
       i) imdb-conv.ipynb - nlp by Convolutional layer         
       ii) imdb-gru.ipynb - GRU implementation             
       iii) imdb-multi-LSTM.ipynb             
       iv) imdb-single-LSTM.ipynb         
       v) sarcasm-LSTM.ipynb - link of sarcasm dataset is available in ipynb               
       vi) sarcasm-conv.ipynb
  
   - sarcasm/sarcasm.py        
       Concept of tokenizer, pad_sequence, and text to sequence process
  
  - text-sampleling - it is producing random words by giving integer/no input.                            
       i) poetry1 - one hot vector, corpus, word index, bidirectional LSTM, given little input text then predict next few texts.                   
       ii) poetry2 - using larger corpus to train both are with adam optimizer and categorical_crossentropy loss.
  
  - tf-docs:                      
       i) Word embedding: Sentimental classifier with IMDB dataset, **Learning embeddings from scratch, subword, wordindex**, ploting embedding graphs!                                     
       ii) text_generation.ipynb: **character-based language generator**                                           
       iii) nmt_with_Attention  :**BahdanauAttention** with GRU, **encoder, decoder and evaluate function for attention**   
       iv) transformer.ipynb: machine transalation, **self-Attention, Multi head attention, Scaled dot product attention** with evaluted function, adam with lr scheduler, checkpoint callback! 
       
  - encodingapi.py: **basic concept of word-index, corpus, tokenizer, padding and text to sequence convertion using tf**
  
## 4) time-series analysis and prediction
   - time-series.ipynb:                   
      Concepts of **Trend and Seasonality, seasonal pattern, Noise, autocorrelation**
   
   - data-prepossing.ipynb:                      
      Creating Windows for sending data into model for traing and forcasting the graph.
      
   - prediction-without-ml.ipynb:                           
      concept of **Naive Forecast, moving_average_forecast** improving by  **seasonality differencing**.
      
   - simple-ann-predication.ipynb:                   
      Concepts of **windowed dataset**, training on window of the graph, with optimizer SGD and loss "mse", forcasting with the simple ann model.
      
   - dnn-prediction.ipynb:                             
      predicting with Deep neural networks, 2 layers of 10 neutrons, with activation="linear" layer at the end.
      
   - simple-rnn.ipynb:        
   **Lambda Layer** and **Huber** loss and 2 layered **Simple RNN** with 40 neurons
   
   - LSTMs-prediction.ipynb:               
   **Bidirectional LSTM** of 32 units
   
   - conv-lstm.ipynb:                  
   **1D conv** layer with followed by birectional LSTM with SGD and momentum                
   **LearningRateScheduler** callback
   
   - Sunspot Database: 
   Conv + LSTM + NN
