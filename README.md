**PREDICTION OF TYROSINASE INHIBITORY PEPTIDES USING MACHINE LEARNING APPROACH**

**Introduction**

Tyrosinase is an enzyme vital for the production of melanin and is linked to an individual's susceptibility to hyperpigmentation disorders, which involves the development of darker, discolored patches of skin. Tyrosinase inhibitory peptides (TIPs) are small peptides with a length of 3-20 amino acid residues, demonstrating tyrosinase inhibitory activity. These findings have significance for pharmaceutical and clinical research, since they can be used as drugs based on the TIPs in the treatment of disorders of hyperpigmentation. It is, however, expensive and time-consuming to identify the possible TIPs through experiments. Therefore, this calls for a need for computational methods that will properly and at lower costs identify TIPs. In this study, our core approach will be to look into the application of machine learning algorithms to train a computational model, able to classify between TIPs and non-TIPs using only peptide sequence information. Thus, more effective and affordable tools are offered for the description of potential inhibitors against tyrosinase.


**Data**

We have obtained data from Proteins & Peptides Mining Lab, which is open access for everyone on the website (https://pmlabstack.pythonanywhere.com/dataset_TIPred). Proteins & Peptides Mining Lab provides 4 datasets that include information about protein sequences only with the following size distribution:
![image](https://github.com/DKodirova/Prediction-of-Tyrosinase-Inhibitory-Peptides-Using-Machine-Learning-Approach/assets/141365455/eddbf68c-1f56-4fbe-8afa-e3cbedfc786a)

However, the sequence information alone is not enough as a data to predict if it is TIP-inhibitory or not. Therefore, extraction of new features and adding them to the initial dataset  was needed. For this we used PseAAC - General Software (http://www.csbio.sjtu.edu.cn/bioinf/PseAAC/) created by Shanghai Jiao Tong University, where given the sequence information and input parameters, one can obtain pseudo amino acid composition information. 

However, the amount of features and importance of features we can extract also depends on sequence size, which ranges from 2 to 20 for these datasets. Based on the distribution curves, it was decided to divide all datasets into short and long, where short include sequences of length 2 and long include length of 3 and more. This division resulted in a total of 8 sequence datasets for each of which multiple feature sets will be generated. 


**Features**

PseAAC Software allowed us to generate 3 features sets using the following input parameters:
![image](https://github.com/DKodirova/Prediction-of-Tyrosinase-Inhibitory-Peptides-Using-Machine-Learning-Approach/assets/141365455/60fb8396-7711-4954-a69a-a27daef25fc1)

![image](https://github.com/DKodirova/Prediction-of-Tyrosinase-Inhibitory-Peptides-Using-Machine-Learning-Approach/assets/141365455/1d841d57-6ad9-4052-8826-a760ff981be9)


**Model Selection**

For this problem classification algorithm was necessary because we are deciding between positive or negative outcomes of TIPs. One of the classifiers is Support Vector Machine (SVM) Algorithm, which was chosen as a core model for prediction. It uses a technique called the kernel trick to transform data and then based on these transformations it finds an optimal boundary between the possible outputs. This model works best for finding complex relationships, which is the case of the TIP-inhibitory problem. 

We used an algorithm by LIBSVM (https://www.csie.ntu.edu.tw/~cjlin/libsvm/) python package that offers an already implemented SVM algorithm. 


**Software Instructions**

To run the program for prediction simply call “python main.py” on the terminal. It will read all of the datasets in the “data” folder, the preparation of which was discussed earlier. Then, it applies the SVM model and returns a prediction report called “report.csv” and actual prediction values stored in the “predictions” folder.


**Results and Discussion**

All performance evaluation metrics point that for all feature sets, performance of long sequence datasets was better with accuracy within 90-91% across all feature sets. The main factor that might have affected the lower performance of short sequence datasets is the dataset size. Shorter sequence datasets still included much less number of sequences than the longer ones, resulting in fewer data points. This makes it difficult for the model to find the relationships within the dataset and classify correctly. Therefore, accuracy and error is much lower for short sequence datasets. 
![image](https://github.com/DKodirova/Prediction-of-Tyrosinase-Inhibitory-Peptides-Using-Machine-Learning-Approach/assets/141365455/e6a1ac8b-a179-4394-8252-3ff3f3e611b0)

**Future Improvements**

Future advancements to consider to further improve the algorithm performance start from obtaining bigger datasets, especially for short sequence datasets. Another thing to consider is feature selection. Even though all of the feature sets yielded similar performance, picking the best predicting/correlating features from each set would improve the performance. These are the changes related to data and features, however, some improvements for the algorithm could be delivered too. For example, using a multi-layer approach for the algorithms is something to consider, as the previous related researches practiced this approach successfully. 


