from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction.text import CountVectorizer

from experiment_baseplate import load_split_data, get_text_data

X_train, y_train, X_validate, y_validate, X_test, y_test = load_split_data()

#Text processing
MAX_INPUT_LENGTH = 1000

tockenizer = CountVectorizer(analyzer="word")
tockenizer.fit(get_text_data())

def tockenize(data):
    n_data = tockenizer.transform(data)
    # if(len(n_data)>MAX_INPUT_LENGTH):
    #     ndata = ndata[:MAX_INPUT_LENGTH]
    # else:
    return n_data
        

# Create a decision tree classifier
clf = DecisionTreeClassifier()

# Train the classifier on the training data
print("Training...")
clf.fit(tockenize(X_train), y_train)
print("Training finished...")

# Predict the class labels for the testing data
#y_pred = clf.predict(tockenize(X_text))

# Evaluate the accuracy of the classifier
accuracy = clf.score(tockenize(X_validate), y_validate)

# Print the accuracy
print("Accuracy:", accuracy)

# print(clf.predict(tockenize(["You are shit", "You are in the shit", "You are an eye sore", "I don't like you", "I enjoy your company", "You are biatch", "You biiitch"])))