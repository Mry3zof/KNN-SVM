import numpy as np
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score


def load_data(dataset_name):
    # Load and preprocess the dataset
    dataset = load_dataset(dataset_name)
    df = dataset['train'].to_pandas()

    missing_count = df.isnull().sum()
    total_cells = np.prod(df.shape)
    print(f"Total cells: {total_cells}")
    print(f"Missing values: {missing_count.sum()}")

    return df


class DataPreprocessor:
    def __init__(self, data):
        self.data = data

    def clean_data(self):
        # Remove rows with missing values
        self.data = self.data.dropna(axis=0)

    def extract_features_labels(self):
        features = self.data["transcription"]
        labels = self.data["sample_name"]
        return features, labels

    def split_data(self, features, labels):
        X_train, X_test, y_train, y_test = train_test_split(
            features, labels, test_size=0.2, shuffle=True, random_state=7
        )
        return X_train, X_test, y_train, y_test

    def transform_data(self, X_train, X_test):
        # transform the text data to vector so it can be used in the model
        vectorizer = TfidfVectorizer()
        X_train_transformed = vectorizer.fit_transform(X_train)
        X_test_transformed = vectorizer.transform(X_test)
        return X_train_transformed, X_test_transformed


class KNNModel:
    def __init__(self, k=3):
        self.k = k
        self.model = KNeighborsClassifier(n_neighbors=self.k)

    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        return self.model.predict(X_test)

    def evaluate(self, y_test, y_pred):
        accuracy = accuracy_score(y_test, y_pred)
        print(f"KNN Accuracy: {accuracy:.4f}")


class SVMModel:
    def __init__(self, kernel='linear', C=1.0):
        """
        Initialize the SVM model with specified kernel and regularization parameter C.
        """
        self.kernel = kernel
        self.C = C
        self.model = SVC(kernel=self.kernel, C=self.C)

    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        return self.model.predict(X_test)

    def evaluate(self, y_test, y_pred):
        accuracy = accuracy_score(y_test, y_pred)
        print(f"SVM Accuracy: {accuracy:.4f}")


def evaluate_model(model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    model.evaluate(y_test, y_pred)


def main():
    dataset_name = "DataFog/medical-transcription-instruct"

    # Load and preprocess data
    df = load_data(dataset_name)
    preprocessor = DataPreprocessor(df)
    preprocessor.clean_data()
    features, labels = preprocessor.extract_features_labels()
    X_train, X_test, y_train, y_test = preprocessor.split_data(features, labels)
    X_train_transformed, X_test_transformed = preprocessor.transform_data(X_train, X_test)

    # take a portion of the data so it doesn't overfit
    X_train_small = X_train_transformed[:8000]
    y_train_small = y_train[:8000]

    knn_model = KNNModel(k=3)
    svm_model = SVMModel(kernel='linear', C=1.0)

    print("Evaluating KNN Model...")
    evaluate_model(knn_model, X_train_small, y_train_small, X_test_transformed, y_test)

    print("Evaluating SVM Model...")
    evaluate_model(svm_model, X_train_small, y_train_small, X_test_transformed, y_test)


if __name__ == "__main__":
    main()

