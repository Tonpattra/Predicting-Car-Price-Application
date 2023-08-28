# HW1: Predicting Car Price
The dataset is look like this
![image](https://github.com/Tonpattra/Machine-Learning/assets/89975216/fbdf466b-0b33-4c6b-a244-c57194895dfe)

## Task 1 :
Preparing the datasets - Download the Car Price dataset from Google classroom. Perform
loading, EDA, preprocessing, model selection, · · · , inference. Grade will be given based on the how well
you adhere to best practices. There are some important coding considerations:

• For the feature owner, map First owner to 1, ..., Test Drive Car to 5
• For the feature fuel, remove all rows with CNG and LPG because CNG and LPG use a different
mileage system i.e., km/kg which is different from kmfeaturepl for Diesel and Petrol
• For the feature mileage, remove “kmpl” and convert the column to numerical type (e.g., float).
Hint: use df.mileage.str.split
• For the feature engine, remove “CC” and convert the column to numerical type (e.g., float)
• Do the same for max power
• For the feature brand, take only the first word and remove the rest
• Drop the feature torque, simply because Chaky’s company does not understand well about it
• You will found out that Test Drive Cars are ridiculously expensive. Since we do not want to
involve this, we will simply delete all samples related to it.

Show the results in Task 1:
![image](https://github.com/Tonpattra/Machine-Learning/assets/89975216/1a876cff-1ece-43ad-bab9-b6d89dbe2e39)








Hint: use df.mileage.str.split :
```
st123012	Todsavad Tangtortan
st123459	Anjana Tissera
st122053	Wanchanok Sunthorn
st123010	Tonson Praphabkul
st122876	Aiman Lameesa
```

## Dependencies :
- mne library 

## Components :
- 01 ETL
    - select 16-channels out of 32-channels
    - notch filter power line
    - filter
- 02 EDA
    - Feature Extraction (Alpha Beta Gamma) 
        - Power spectral density (PSD)
        - Asymmetry
- 03 ML Model
    - SVM
    - NB
    - KNN
    - LR
- 04 DL Model
    - CNN1D
    - LSTM

## Result :
- Out of the ML models we trained, Support Vector Machine has higher test accuracy.
- Generally, When Relative Gamma frequency band is included as a feature, Model testing accuracy is high.
- Deep Learning Models didn’t result with good accuracy.
- Accuracy of testing Akkaradet’s dataset
    - Low accuracy compared to our test dataset

## Conclusion :
- Hypothesis 1:  Some of the frequency band are more significant
    - The results show that Relative Gamma has a higher significance in classifying chronic stress
- Hypothesis 2 : Chronic Stress can be classified based on EEG data
    - Model accuracy varies with the dataset
    - Our experiment doesn't completely support this hypothesis.


## Limitation :
 - Bias Stress
 - No Consultant
 - Small Dataset

## Future Work :
 - Recording Post Examination
    - Collecting the same participants after post-examination

## Reference :
- Saeed, S. M. U., Anwar, S. M., Khalid, H., Majid, M., & Bagci, U. (2020). EEG based classification of long-term stress using psychological labeling. Sensors, 20(7), 1886.

- Minguillon, J., Lopez-Gordo, M. A., & Pelayo, F. (2016). Stress assessment by prefrontal relative gamma. Frontiers in computational neuroscience, 10, 101.

- Zhang, Y., Wang, Q., Chin, Z. Y., & Ang, K. K. (2020, July). Investigating different stress-relief methods using Electroencephalogram (EEG). In 2020 42nd Annual International Conference of the IEEE Engineering in Medicine & Biology Society (EMBC) (pp. 2999-3002). IEEE.
