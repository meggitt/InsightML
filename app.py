import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import accuracy_score, mean_squared_error, confusion_matrix
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder
import numpy as np

# Function to handle missing values
def handle_missing_values(df, option):
    if option == "Ignore":
        return df.dropna()
    elif option == "Fill with NaN":
        return df.fillna(np.nan)

# Function to get model
def get_model(algorithm_name, task_type, params=None):
    if task_type == "classification":
        if algorithm_name == "Random Forest":
            return RandomForestClassifier(**params) if params else RandomForestClassifier()
        elif algorithm_name == "Logistic Regression":
            return LogisticRegression(**params) if params else LogisticRegression()
        elif algorithm_name == "Support Vector Machine":
            return SVC(**params) if params else SVC()
        elif algorithm_name == "Decision Tree":
            return DecisionTreeClassifier(**params) if params else DecisionTreeClassifier()
        elif algorithm_name == "K-Nearest Neighbors":
            return KNeighborsClassifier(**params) if params else KNeighborsClassifier()
        elif algorithm_name == "Naive Bayes":
            return GaussianNB(**params) if params else GaussianNB()
        elif algorithm_name == "Gradient Boosting":
            return GradientBoostingClassifier(**params) if params else GradientBoostingClassifier()
    elif task_type == "regression":
        if algorithm_name == "Random Forest":
            return RandomForestRegressor(**params) if params else RandomForestRegressor()
        elif algorithm_name == "Linear Regression":
            return LinearRegression(**params) if params else LinearRegression()
        elif algorithm_name == "Support Vector Machine":
            return SVR(**params) if params else SVR()
        elif algorithm_name == "Decision Tree":
            return DecisionTreeRegressor(**params) if params else DecisionTreeRegressor()
        elif algorithm_name == "K-Nearest Neighbors":
            return KNeighborsRegressor(**params) if params else KNeighborsRegressor()
        elif algorithm_name == "Ridge Regression":
            return Ridge(**params) if params else Ridge()
        elif algorithm_name == "Lasso Regression":
            return Lasso(**params) if params else Lasso()
        elif algorithm_name == "Gradient Boosting":
            return GradientBoostingRegressor(**params) if params else GradientBoostingRegressor()

# Function to display confusion matrix
def display_confusion_matrix(cm, labels, dataset_name):
    st.subheader(f"{dataset_name}")
    st.write("Confusion Matrix:")
    cm_df = pd.DataFrame(cm, index=labels, columns=labels)
    cm_df['All'] = cm_df.sum(axis=1)
    cm_df.loc['All'] = cm_df.sum()
    st.write(cm_df)

# Main Streamlit app
def main():
    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.write("Data Preview:")
            st.write(df.head())
        except Exception as e:
            st.error(f"Error loading file: {e}")
            return
        
        # Handle missing values
        try:
            missing_value_option = st.radio("How to handle missing values?", ('Ignore', 'Fill with NaN'))
            df = handle_missing_values(df, missing_value_option)
        except Exception as e:
            st.error(f"Error handling missing values: {e}")
            return
        
        # Checkbox to select all columns
        if st.checkbox("Select all columns for modeling"):
            selected_columns = df.columns.tolist()
        else:
            try:
                selected_columns = st.multiselect("Select columns for modeling (at least 2)", df.columns, default=df.columns[:2].tolist())
            except Exception as e:
                st.error(f"Error selecting columns: {e}")
                return
        
        if len(selected_columns) < 2:
            st.error("Please select at least two columns.")
            return
        
        # Choose target column
        try:
            target_column = st.selectbox("Select the target column", selected_columns)
        except Exception as e:
            st.error(f"Error selecting target column: {e}")
            return
        
        # Create a dictionary to store LabelEncoders
        encoders = {}
        
        # Encode categorical variables
        try:
            for col in df.select_dtypes(include=['object']).columns:
                if col != target_column:
                    le = LabelEncoder()
                    df[col] = le.fit_transform(df[col])
                    encoders[col] = le
        except Exception as e:
            st.error(f"Error encoding categorical variables: {e}")
            return
        
        # Determine if target is categorical or continuous
        try:
            target = df[target_column]
            task_type = "classification" if target.dtype == 'int64' or target.nunique() < 20 else "regression"
        except Exception as e:
            st.error(f"Error determining task type: {e}")
            return
        
        # Select algorithm based on task type
        if task_type == "classification":
            algorithm_options = ["Random Forest", "Logistic Regression", "Support Vector Machine", "Decision Tree", "K-Nearest Neighbors", "Naive Bayes", "Gradient Boosting"]
        else:
            algorithm_options = ["Random Forest", "Linear Regression", "Support Vector Machine", "Decision Tree", "K-Nearest Neighbors", "Ridge Regression", "Lasso Regression", "Gradient Boosting"]
        
        try:
            algorithm = st.selectbox("Choose an algorithm", algorithm_options)
        except Exception as e:
            st.error(f"Error selecting algorithm: {e}")
            return
        
        # Parameters option
        try:
            use_best_params = st.radio("Parameters option", ('Choose best parameters', 'Select your own parameters'))
        except Exception as e:
            st.error(f"Error selecting parameters option: {e}")
            return
        
        params = {}
        if use_best_params == 'Select your own parameters':
            try:
                if algorithm == "Random Forest":
                    if task_type == "classification":
                        params['n_estimators'] = st.number_input("Number of estimators", min_value=1, value=100)
                    else:
                        params['n_estimators'] = st.number_input("Number of estimators", min_value=1, value=100)
                elif algorithm == "Logistic Regression":
                    params['C'] = st.number_input("Inverse of regularization strength (C)", value=1.0)
                elif algorithm == "Support Vector Machine":
                    params['C'] = st.number_input("Regularization parameter (C)", value=1.0)
                    params['kernel'] = st.selectbox("Kernel", ['linear', 'rbf'])
                elif algorithm == "Decision Tree":
                    params['max_depth'] = st.number_input("Max depth", min_value=1, value=3)
                elif algorithm == "K-Nearest Neighbors":
                    params['n_neighbors'] = st.number_input("Number of neighbors", min_value=1, value=5)
                elif algorithm == "Ridge Regression":
                    params['alpha'] = st.number_input("Regularization strength (alpha)", value=1.0)
                elif algorithm == "Lasso Regression":
                    params['alpha'] = st.number_input("Regularization strength (alpha)", value=1.0)
            except Exception as e:
                st.error(f"Error setting parameters: {e}")
                return
        
        try:
            model = get_model(algorithm, task_type, params if use_best_params == 'Select your own parameters' else None)
        except Exception as e:
            st.error(f"Error getting model: {e}")
            return
        
        # Train-test split options
        try:
            split_option = st.radio("Choose train-test split method", ('Random', 'Shuffle'))
            train_size = st.slider("Train size (percentage)", 20, 99, 80)  # Ensuring a minimum train size of 70%
            test_size = 100 - train_size
            train_size = train_size / 100
            test_size = test_size / 100
            split_ratio = f"Train {train_size * 100:.0f}% / Test {test_size * 100:.0f}%"
            st.write(split_ratio)
        except Exception as e:
            st.error(f"Error selecting train-test split options: {e}")
            return
        
        # Split data
        try:
            X = df[selected_columns].drop(columns=[target_column])
            y = df[target_column]
        
            if split_option == 'Random':
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
            elif split_option == 'Shuffle':
                df = df.sample(frac=1).reset_index(drop=True)
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
        except Exception as e:
            st.error(f"Error splitting data: {e}")
            return
        
        # Train model
        try:
            if use_best_params == 'Choose best parameters':
                if algorithm == "Random Forest":
                    param_grid = {'n_estimators': [50, 100, 200]}
                elif algorithm == "Logistic Regression":
                    param_grid = {'C': [0.1, 1.0, 10.0]}
                elif algorithm == "Support Vector Machine":
                    param_grid = {'C': [0.1, 1.0, 10.0], 'kernel': ['linear', 'rbf']}
                elif algorithm == "Decision Tree":
                    param_grid = {'max_depth': [3, 5, 7]}
                elif algorithm == "K-Nearest Neighbors":
                    param_grid = {'n_neighbors': [3, 5, 7]}
                elif algorithm == "Gradient Boosting":
                    param_grid = {'n_estimators': [50, 100, 200]}
                else:
                    param_grid = {}
                
                grid_search = GridSearchCV(model, param_grid, cv=3)
                grid_search.fit(X_train, y_train)
                model = grid_search.best_estimator_
                st.write("Best parameters found:")
                st.write(model)
            
            model.fit(X_train, y_train)
        except Exception as e:
            st.error(f"Error training model: {e}")
            return
        
        # Validation accuracy or MSE
        try:
            y_train_pred = model.predict(X_train)
            if task_type == "classification":
                train_score = accuracy_score(y_train, y_train_pred)
                st.write(f"Validation Accuracy: {train_score:.2f}")
                # Display confusion matrix
                train_cm = confusion_matrix(y_train, y_train_pred)
                display_confusion_matrix(train_cm, np.unique(y_train), "Training Set")
            else:
                train_score = mean_squared_error(y_train, y_train_pred, squared=False)
                st.write(f"Validation RMSE: {train_score:.2f}")
        except Exception as e:
            st.error(f"Error evaluating model on training data: {e}")
            return
        
        # Test accuracy or MSE
        try:
            if test_size > 0:
                y_test_pred = model.predict(X_test)
                if task_type == "classification":
                    test_score = accuracy_score(y_test, y_test_pred)
                    st.write(f"Test Accuracy: {test_score:.2f}")
                    # Display confusion matrix
                    test_cm = confusion_matrix(y_test, y_test_pred)
                    display_confusion_matrix(test_cm, np.unique(y_test), "Validation Set")
                else:
                    test_score = mean_squared_error(y_test, y_test_pred, squared=False)
                    st.write(f"Test RMSE: {test_score:.2f}")
        except Exception as e:
            st.error(f"Error evaluating model on test data: {e}")
            return

        # Upload a test file for checking testing accuracy
        st.header("Upload a test file for checking testing accuracy:")
        test_uploaded_file = st.file_uploader("Upload your test CSV file", type=["csv"])
        if test_uploaded_file is not None:
            try:
                test_df = pd.read_csv(test_uploaded_file)
                # Ensure the test data only contains the selected columns and the target column
                test_df = test_df[selected_columns]
                # Apply the same LabelEncoder to the test data
                for col in test_df.select_dtypes(include=['object']).columns:
                    if col in encoders:
                        test_df[col] = encoders[col].transform(test_df[col])
                test_X = test_df.drop(columns=[target_column])
                test_y = test_df[target_column]
                test_pred = model.predict(test_X)
                if task_type == "classification":
                    test_accuracy = accuracy_score(test_y, test_pred)
                    st.write(f"Testing Accuracy: {test_accuracy:.2f}")
                    # Display confusion matrix
                    test_cm = confusion_matrix(test_y, test_pred)
                    display_confusion_matrix(test_cm, np.unique(test_y), "Uploaded Test Set")
                else:
                    test_rmse = mean_squared_error(test_y, test_pred, squared=False)
                    st.write(f"Testing RMSE: {test_rmse:.2f}")
            except Exception as e:
                st.error(f"Error evaluating model on uploaded test data: {e}")
                return

        # Cross-validation scores
        try:
            if task_type == "classification":
                cv_scores = cross_val_score(model, X, y, cv=10, scoring='accuracy')
            else:
                cv_scores = cross_val_score(model, X, y, cv=10, scoring='neg_root_mean_squared_error')
            st.subheader("Cross-validation scores: ")
            st.write(f"{cv_scores}")
            st.subheader("Average cross-validation score: ")
            st.write(f"{np.mean(cv_scores)}")
        except Exception as e:
            st.error(f"Error calculating cross-validation scores: {e}")
            return

if __name__ == "__main__":
    try:
        main()
    except:
        st.write(f"Something went wrong, please make sure your data is correct and you are selecting the right parameters")
