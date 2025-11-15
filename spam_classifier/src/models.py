from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

def train_models(features, labels):
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

    # Train Naive Bayes
    nb_model = MultinomialNB()
    nb_model.fit(X_train, y_train)
    nb_pred = nb_model.predict(X_test)
    
    print("Naive Bayes Model Evaluation:")
    print(classification_report(y_test, nb_pred))
    print("Accuracy:", accuracy_score(y_test, nb_pred))

    # Train Logistic Regression
    logreg_model = LogisticRegression(max_iter=1000)
    logreg_model.fit(X_train, y_train)
    logreg_pred = logreg_model.predict(X_test)
    
    print("\nLogistic Regression Model Evaluation:")
    print(classification_report(y_test, logreg_pred))
    print("Accuracy:", accuracy_score(y_test, logreg_pred))

    return nb_model, logreg_model
