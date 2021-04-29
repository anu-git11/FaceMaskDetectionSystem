# FACE MASK DETECTION SYSTEM with FACE MATCHING FROM SQL DATABASE

As of January 28, 2020, severe acute respiratory syndrome coronavirus 2 (SARS-CoV-2) has infected more than 100 million individuals worldwide and caused more than 2.1 million deaths.

Artificial Intelligence (AI) based on Machine learning and Deep Learning can help to fight Covid-19 in many ways. Machine learning allows researchers and clinicians evaluate vast quantities of data to forecast the distribution of COVID-19, to serve as an early warning mechanism for potential pandemics, and to classify vulnerable populations. The provision of healthcare needs funding for emerging technology such as artificial intelligence, IoT, big data and machine learning to tackle and predict new diseases. To better understand infection rates and to trace and quickly detect infections, the AI’s power is being exploited to address the Covid-19 pandemic. Many countries require by law that individuals wear masks in public. In other instances, mask-wearing is required by private institutions, such as grocery stores or shopping malls. These rules and laws were developed as an action to the exponential growth in cases and deaths in many areas. However, the process of monitoring large groups of people is becoming more difficult. The monitoring process involves the detection of anyone who is not wearing a face mask.


Therefore, we have created a face mask face detection model that is based on computer vision and deep learning. The proposed model can be integrated with surveillance cameras to impede the COVID-19 transmission by allowing the detection of people who are wearing masks not wearing face masks. The model is a integration between of deep learning and classical machine learning techniques with YOLO v5, OpenCV, CUDA DNN, Pytorch and Torchvision. 

## **HIGH LEVEL ARCHITECTURE DIAGRAM FOR THE SYSTEM:**

![HighLevelARCHMASKITOR](https://user-images.githubusercontent.com/63171468/116589665-de47e780-a8ea-11eb-848f-e80adae01498.png)

This architecture diagram clearly shows the process of training the model (highlighted in green), and the working of inferencing engine using the trained model along with its interactions with the SQL database to store and retrieve the detections for Face-Matching and Reports generation. (*The images used in the above diagram are from google image search*)

## **DATA DESCRIPTION**

The dataset used to train YOLOv5 model can be downloaded from:
https://github.com/anu-git11/MaskData . 
This dataset consists of ~7000 images of masked and non-masked people. 
It was created from open source images and was labelled in yolo format using the open source labelling tool :
**LabelImg (https://github.com/tzutalin/labelImg)** 

## **TRAINING**

The YOLOv5x algorithm was trained on Google COLAB with our dataset for various image sizes, with 620px giving the best results i.e. A detection Accuracy of 97%, Precision 0.95, and a Recall of 0.9. The trained model is stored in the Models folder of the MASKITORAlgorithm.


