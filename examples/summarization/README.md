# Text summarization with CNN Dailymail dataset
## Preprocessing
 - Preprocessing is a long task and has scope for better parallelization.
 - The notebook has the code necessary for generating the preprocessed data.
 - Since it is a long task, the results are made available to download from 
 Google Drive.
 - When I tried, it crashed on Google Colab but one could work to optimize it.

## Training
 - Downloads preprocessed data from Google Drive.
 - Builds the model using components from `copynet-tf`.
 - Can be run from Google Colab.