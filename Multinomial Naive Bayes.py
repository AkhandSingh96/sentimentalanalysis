
# Fit and transform the training data to a document-term matrix using CountVectorizer
countVect = CountVectorizer() 
X_train_countVect = countVect.fit_transform(X_train_cleaned)
print("Number of features : %d \n" %len(countVect.get_feature_names())) #6378 
print("Show some feature names : \n", countVect.get_feature_names()[::1000])
y_train=np.array(y_train)
# Train MultinomialNB classifier
y_train=y_train.astype('int')
#print(y_train.shape)
mnb = MultinomialNB()
mnb.fit(X_train_countVect, y_train)

def modelEvaluation(predictions):
    print ("\nAccuracy on validation set: {:.4f}".format(accuracy_score(y_test, predictions)))
    print("\nClassification report : \n", metrics.classification_report(y_test, predictions))
    print("\nConfusion Matrix : \n", metrics.confusion_matrix(y_test, predictions))

def modelTrainEvaluation(predictions):
    '''
    Print model evaluation to predicted result 
    '''
    print ("\nAccuracy on train set: {:.4f}".format(accuracy_score(y_train, predictions)))
    #print("\nAUC score : {:.4f}".format(roc_auc_score(y_train, predictions)))
    print("\nClassification report : \n", metrics.classification_report(y_train, predictions))
    print("\nConfusion Matrix : \n",metrics.confusion_matrix(y_train, predictions) )

# Evaluate the model on validaton set
predictions = mnb.predict(countVect.transform(X_train_cleaned))
modelTrainEvaluation(predictions)
predictions = mnb.predict(countVect.transform(X_test_cleaned))
modelEvaluation(predictions)
