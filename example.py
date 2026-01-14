from flask import Flask, render_template, request, redirect, url_for, flash, g
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report


app = Flask(__name__)
app.config['SECRET_KEY'] = '1234567890'  # Required for flash messages and Flask-WTF

df = pd.read_csv('reviews1.csv')

    # print("Error: reviews.csv not found.")
    # print("Please replace 'reviews.csv' with your dataset file path.")
    # print("Ensure the file has 'review_text' and 'is_fake' columns.")
    # exit()

# Handle missing values if any
df.dropna(inplace=True)

# 2. Split Data into Training and Testing Sets
X = df['review_text']
y = df['is_fake']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# 3. Vectorize Text Data (TF-IDF)
# Convert text reviews into a numerical format (features) using TF-IDF
tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

# 4. Train a Machine Learning Model (Logistic Regression)
# Logistic Regression is a common and effective classifier for text data
model = LogisticRegression(max_iter=1000)
model.fit(X_train_tfidf, y_train)

# 5. Evaluate the Model
y_pred = model.predict(X_test_tfidf)
accuracy = accuracy_score(y_test, y_pred)
# print(f"Model Accuracy: {accuracy:.3f}")
# print("\nClassification Report:")
# print(classification_report(y_test, y_pred))

# review_text_tfidf=""
# tfidf_vectorizer=""
user_input=""
@app.route('/', methods=['GET', 'POST'])
def submit_form():
    if request.method == 'POST':
        # Access form data using request.form.get() or request.form[]
        user_input = request.form.get('user_input_name')  # 'user_input_name' matches the HTML input's name attribute

        if not user_input:
            flash('Input is required!', 'error')  # Use flash messages for feedback
            return render_template('form.html')
        print(user_input)    

        # Process the data (e.g., save to a database, perform logic)
        try:
            df = pd.read_csv('reviews1.csv')
        except FileNotFoundError:
            print("Error: reviews.csv not found.")
            print("Please replace 'reviews.csv' with your dataset file path.")
            print("Ensure the file has 'review_text' and 'is_fake' columns.")
            exit()

        # Handle missing values if any
        # df.dropna(inplace=True)
        #
        # # 2. Split Data into Training and Testing Sets
        # X = df['review_text']
        # y = df['is_fake']
        # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
        #
        # # 3. Vectorize Text Data (TF-IDF)
        # # Convert text reviews into a numerical format (features) using TF-IDF
        # g.tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_df=0.8)
        # X_train_tfidf = g.tfidf_vectorizer.fit_transform(X_train)
        # X_test_tfidf = g.tfidf_vectorizer.transform(X_test)
        #
        # # 4. Train a Machine Learning Model (Logistic Regression)
        # # Logistic Regression is a common and effective classifier for text data
        # model = LogisticRegression(max_iter=1000)
        # model.fit(X_train_tfidf, y_train)
        #
        # # 5. Evaluate the Model
        # y_pred = model.predict(X_test_tfidf)
        # accuracy = accuracy_score(y_test, y_pred)
        # print(f"Model Accuracy: {accuracy:.3f}")
        # print("\nClassification Report:")
        # print(classification_report(y_test, y_pred))

        processed_data = user_input.upper()
        review_text_tfidf =tfidf_vectorizer.transform([user_input])
        # Predict the class
        prediction = model.predict(review_text_tfidf)
        print(prediction)


        if prediction[0] == 0:
            processed_data= "Fake Review"
        else:
            processed_data= "Real Review"


        # Redirect the user to another page after successful submission
        flash(processed_data, 'success')
        #return redirect(url_for('confirmation', data=processed_data))

    # For GET requests, just render the form
    return render_template('form.html')


@app.route('/confirmation/<data>')
def confirmation(data):
    return f"Output: {data}"




@app.route('/predict_review_authenticity')
def predict_review_authenticity(review_text):
    # Vectorize the new text using the *same* fitted vectorizer
    review_text_tfidf = tfidf_vectorizer.transform([review_text])
    # Predict the class
    prediction = model.predict(review_text_tfidf)

    if prediction[0] == 0:
        return "Fake Review"
    else:
        return "Real Review"


    # Example usage:
    new_review_1 = "This is the best product ever, absolutely fantastic, a must buy!"
    new_review_2 = "The product worked as described, but I had some issues with the delivery."

    print(f"\nReview 1 prediction: {predict_review_authenticity(new_review_1)}")
    print(f"Review 2 prediction: {predict_review_authenticity(new_review_2)}")
    return "1"
if __name__ == '__main__':
    app.run(debug=True)


# 1. Load the Dataset
# Assume 'reviews.csv' has two columns: 'review_text' and 'is_fake' (0 or 1)



# 6. Predict on New Data
