# rul-prediction-fed
Prediction of Remaining Useful Life under both centralised and federated learning environments, using the FD001 - FD004 Turbofan Engine Degradation datasets.

- There are 5 main folders:
  - FATE: Contains the files needed to submit a job to FATE. First, set the appropriate data distribution in all files. Then, run python upload.py to upload all the files and python homo_sbt.py in order to perform the training and testing of the Federated Gradient Decision Tree Algorithm.
  - data_analysis: Contains the data analysis performed to all four datasets, which include feature selection, scaling, smoothing, PCA, condition extraction, etc.
  - federated-learning: This folder contains the FL-data folder with all the different partitions of the data for both the FATE library and the Flower simulation, and the code necessary to run the Flower Federated experiments. In order to run the experiments, open various terminal tabs: one for the server (python server.py) and the others for the different clients (python client.py --partition << # of partition >>).
  - rul_prediction_ml: Contains the different centralised baseline models and the histograms with all the Federated Learning experiment results ran using Flower.
  - TED: Contains the raw dataset files (FD001 - FD004).
