#Elif Çalışkan 2016400183
import cv2 as cv
import numpy as np
from sklearn import svm
from sklearn.cluster import KMeans
import os
from sklearn.model_selection import GridSearchCV
from sklearn.metrics.pairwise import chi2_kernel
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.svm import SVC

dirNames = np.zeros((0))
descriptions = np.zeros((0,128))
labels = np.zeros((0))
clusterCount = 50
filePaths = np.zeros((0))
sift = cv.xfeatures2d.SIFT_create()

for root, dirs, files in os.walk("./train"):
    for dir in dirs:
        dirNames = np.append(dirNames, [dir])
        print(dirNames)
        for root1, dirs1, files1 in os.walk("./train/"+dir):
            for file in files1:
                path = "./train/"+dir+ "/" +file
                filePaths = np.append(filePaths, path)
                img = cv.resize(cv.imread(path), (150, 150), cv.INTER_AREA)
                kp, des = sift.detectAndCompute(img, None)
                descriptions = np.append(descriptions, des, 0)
                labels = np.append(labels, dir)

kmeans = KMeans(n_clusters=clusterCount)
kmeans.fit(descriptions)

image_descriptions = np.zeros((filePaths.size, clusterCount))
c = 0
for path in filePaths:
    img = cv.resize(cv.imread(path), (150, 150), cv.INTER_AREA)
    kp, des = sift.detectAndCompute(img, None)
    img_des = kmeans.predict(des)
    for i in range(len(img_des)):
        image_descriptions[c][img_des[i]] += 1
    c+=1


print(image_descriptions)
svc_classifier = SVC(kernel=chi2_kernel, class_weight = 'balanced')
svc_classifier = svc_classifier.fit(image_descriptions, labels)

test_labels = np.zeros((0))
test_descriptions = np.zeros((0, 50))

for root, dirs, files in os.walk("./test"):
    for dir in dirs:
        for root1, dirs1, files1 in os.walk("./test/"+dir):
            for file in files1:
                path = "./test/"+dir+ "/" +file
                img = cv.resize(cv.imread(path), (150, 150), cv.INTER_AREA)
                sift = cv.xfeatures2d.SIFT_create()
                kp, des = sift.detectAndCompute(img, None)
                arr = np.zeros(50)
                img_des = kmeans.predict(des)
                test_labels = np.append(test_labels, dir)
                for i in range(len(img_des)):
                    arr[img_des[i]] += 1
                test_descriptions = np.append(test_descriptions, [arr], 0)

print(test_descriptions)
print("x test predict")
test_predict = svc_classifier.predict(test_descriptions)

print("labels")
print(test_labels)
print("test_predict")
print(test_predict)
confusion_matrix = confusion_matrix(test_labels, test_predict)
print("confusionM")
print(confusion_matrix)
print(classification_report(test_labels, test_predict))
print("score")
score = svc_classifier.score(test_descriptions, test_labels)
print(score)

parameter_candidates = [
  {'C': [1, 10, 100, 1000], 'kernel': ['linear']},
  {'C': [1, 10, 100, 1000], 'kernel': [chi2_kernel]},
]
print("grid")
#Create a classifier object with the classifier and parameter candidates
clf = GridSearchCV(estimator=svm.SVC(), param_grid=parameter_candidates, n_jobs=-1).fit(image_descriptions, labels)
print("best score ", clf.best_score_)
print("C ", clf.best_estimator_.C)
print("kernel ", clf.best_estimator_.kernel)
classifier = svm.SVC(C=clf.best_estimator_.C, kernel=clf.best_estimator_.kernel).fit(image_descriptions, labels)

# Train the classifier on data1's feature and target data
print("new_test")
new_test_pred = classifier.predict(test_descriptions)
print(new_test_pred)
#print("confusionM")
#print(confusion_matrix(test_labels, new_test_pred))
#print(classification_report(test_labels, new_test_pred))
print("score2")
score2 = classifier.score(test_descriptions, test_labels)
print(score2)
training_pred = classifier.predict(image_descriptions)
print("scpre3")
print(classifier.score(image_descriptions, labels))
#print(confusion_matrix(labels, training_pred))
#print(classification_report(labels, training_pred))

