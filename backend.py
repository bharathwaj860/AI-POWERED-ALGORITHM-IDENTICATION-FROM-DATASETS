import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier, MLPRegressor
from xgboost import XGBClassifier, XGBRegressor
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score, f1_score, precision_score, recall_score
from sklearn.preprocessing import LabelEncoder, StandardScaler, RobustScaler
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif, f_regression, mutual_info_classif, mutual_info_regression
from scipy import stats
import optuna
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter


def analyze_dataset(df, target_column, verbose=True):
    """
    Analyze dataset to extract detailed characteristics.
    
    Parameters:
    - df: pandas DataFrame
    - target_column: string, name of the target column
    - verbose: bool, whether to print analysis results
    
    Returns:
    - dictionary with dataset characteristics
    """
    # Make a copy to avoid modifying the original dataframe
    df_copy = df.copy()
    
    # Basic dataset info
    num_samples, num_features = df_copy.shape
    num_features = num_features - 1  # Excluding target column
    
    # Detect target type
    if df_copy[target_column].dtype == 'object':
        target_type = "classification"
        # Encode categorical target for further analysis
        df_copy[target_column] = LabelEncoder().fit_transform(df_copy[target_column])
    else:
        # Numerical target - check if continuous or discrete
        unique_values = df_copy[target_column].nunique()
        if unique_values <= 10:
            target_type = "classification"  # Treating as classification if <= 10 unique values
        else:
            target_type = "regression"
    
    # Analyze columns by type
    cat_columns = df_copy.select_dtypes(include=['object', 'category']).columns.tolist()
    num_columns = df_copy.select_dtypes(include=['number']).columns.tolist()
    if target_column in num_columns:
        num_columns.remove(target_column)
    
    num_categorical = len(cat_columns)
    num_numerical = len(num_columns)
    
    # Check for class imbalance
    is_imbalanced = False
    if target_type == "classification":
        class_counts = df_copy[target_column].value_counts(normalize=True)
        is_imbalanced = class_counts.max() > 0.7
        minority_class_ratio = class_counts.min()
    else:
        minority_class_ratio = None
    
    # Check for missing values
    missing_values = df_copy.isnull().sum()
    has_missing_values = missing_values.sum() > 0
    missing_ratio = missing_values.sum() / (num_samples * num_features) if num_features > 0 else 0
    
    # Check for outliers in numerical columns
    outlier_ratios = {}
    for col in num_columns:
        if col != target_column:
            # Use IQR method to detect outliers
            Q1 = df_copy[col].quantile(0.25)
            Q3 = df_copy[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers = ((df_copy[col] < lower_bound) | (df_copy[col] > upper_bound)).sum()
            outlier_ratios[col] = outliers / num_samples
    
    has_outliers = any(ratio > 0.05 for ratio in outlier_ratios.values())
    
    # Prepare for correlation analysis
    # Encode categorical columns
    for col in cat_columns:
        df_copy[col] = LabelEncoder().fit_transform(df_copy[col])
    
    # Correlation analysis
    correlations = {}
    if num_features > 1:
        corr_matrix = df_copy.corr().abs()
        target_correlations = corr_matrix[target_column].drop(target_column).sort_values(ascending=False)
        
        # Identify highly correlated features
        feature_correlations = corr_matrix.drop(target_column, axis=0).drop(target_column, axis=1)
        high_corr_features = []
        for i in range(len(feature_correlations.columns)):
            for j in range(i+1, len(feature_correlations.columns)):
                if feature_correlations.iloc[i, j] > 0.8:
                    high_corr_features.append((feature_correlations.columns[i], feature_correlations.columns[j]))
        
        correlations = {
            'target_correlations': target_correlations,
            'high_corr_features': high_corr_features
        }
    
    # Feature importance using mutual information
    feature_importance = {}
    if target_type == "classification":
        mi_func = mutual_info_classif
    else:
        mi_func = mutual_info_regression
    
    # Calculate mutual information for all features
    X = df_copy.drop(columns=[target_column])
    y = df_copy[target_column]
    
    try:
        mi_scores = mi_func(X, y, random_state=42)
        feature_importance = dict(zip(X.columns, mi_scores))
    except:
        # Fallback if mutual info fails
        feature_importance = {}
    
    # Dimensionality characteristics
    dimensionality_ratio = num_features / num_samples if num_samples > 0 else 0
    high_dimensionality = dimensionality_ratio > 0.3
    
    # Skewness analysis for numerical columns
    skewness = {}
    for col in num_columns:
        if col != target_column:
            skewness[col] = stats.skew(df_copy[col].dropna())
    
    has_skewed_features = any(abs(s) > 1 for s in skewness.values())
    
    dataset_info = {
        "num_samples": num_samples,
        "num_features": num_features,
        "num_categorical": num_categorical,
        "num_numerical": num_numerical,
        "target_type": target_type,
        "is_imbalanced": is_imbalanced,
        "minority_class_ratio": minority_class_ratio,
        "has_missing_values": has_missing_values,
        "missing_ratio": missing_ratio,
        "has_outliers": has_outliers,
        "outlier_ratios": outlier_ratios,
        "correlations": correlations,
        "feature_importance": feature_importance,
        "high_dimensionality": high_dimensionality,
        "dimensionality_ratio": dimensionality_ratio,
        "has_skewed_features": has_skewed_features,
        "skewness": skewness
    }
    
    if verbose:
        print("\n=== Dataset Analysis ===")
        print(f"Samples: {num_samples}, Features: {num_features}")
        print(f"Categorical features: {num_categorical}, Numerical features: {num_numerical}")
        print(f"Target type: {target_type}")
        if target_type == "classification":
            print(f"Class imbalance: {'Yes' if is_imbalanced else 'No'}")
            if is_imbalanced:
                print(f"Minority class ratio: {minority_class_ratio:.2f}")
        print(f"Missing values: {'Yes' if has_missing_values else 'No'} (ratio: {missing_ratio:.2f})")
        print(f"Outliers: {'Yes' if has_outliers else 'No'}")
        print(f"High dimensionality: {'Yes' if high_dimensionality else 'No'} (ratio: {dimensionality_ratio:.2f})")
        print(f"Skewed features: {'Yes' if has_skewed_features else 'No'}")
        
        if correlations and 'target_correlations' in correlations and not correlations['target_correlations'].empty:
            print("\nTop 5 features by correlation with target:")
            print(correlations['target_correlations'].head(5))
    
    return dataset_info


def recommend_algorithm(dataset_info, verbose=True):
    """
    Recommend ML algorithm based on comprehensive dataset analysis.
    
    Parameters:
    - dataset_info: dictionary with dataset characteristics
    - verbose: bool, whether to print reasoning
    
    Returns:
    - tuple: (recommended algorithm name, explanation)
    """
    target_type = dataset_info["target_type"]
    
    # Initialize algorithm scores
    algorithm_scores = {}
    
    if target_type == "classification":
        algorithms = [
            "logistic_regression", "random_forest_classifier", "svm_classifier", 
            "xgboost_classifier", "gradient_boosting_classifier", "knn_classifier",
            "naive_bayes", "mlp_classifier", "decision_tree_classifier"
        ]
    else:  # regression
        algorithms = [
            "linear_regression", "ridge_regression", "lasso_regression", "elastic_net",
            "random_forest_regressor", "svm_regressor", "xgboost_regressor",
            "gradient_boosting_regressor", "knn_regressor", "mlp_regressor",
            "decision_tree_regressor"
        ]
    
    for algo in algorithms:
        algorithm_scores[algo] = 0
    
    # Analyze dataset characteristics and adjust scores
    reasons = []
    
    # Sample size considerations
    if dataset_info["num_samples"] < 200:
        if target_type == "classification":
            algorithm_scores["naive_bayes"] += 2
            algorithm_scores["logistic_regression"] += 2
            algorithm_scores["knn_classifier"] += 1
            algorithm_scores["decision_tree_classifier"] += 1
            reasons.append("Small sample size favors simpler models like Naive Bayes and Logistic Regression.")
        else:
            algorithm_scores["linear_regression"] += 2
            algorithm_scores["ridge_regression"] += 2
            algorithm_scores["decision_tree_regressor"] += 1
            reasons.append("Small sample size favors simpler models like Linear Regression and Ridge.")
    elif dataset_info["num_samples"] < 1000:
        if target_type == "classification":
            algorithm_scores["logistic_regression"] += 1
            algorithm_scores["random_forest_classifier"] += 1
            algorithm_scores["gradient_boosting_classifier"] += 1
            reasons.append("Medium sample size works well with both simple and ensemble methods.")
        else:
            algorithm_scores["ridge_regression"] += 1
            algorithm_scores["random_forest_regressor"] += 1
            algorithm_scores["gradient_boosting_regressor"] += 1
            reasons.append("Medium sample size works well with both simple and ensemble methods.")
    else:  # Large sample size
        if target_type == "classification":
            algorithm_scores["random_forest_classifier"] += 2
            algorithm_scores["gradient_boosting_classifier"] += 2
            algorithm_scores["xgboost_classifier"] += 2
            algorithm_scores["mlp_classifier"] += 1
            reasons.append("Large sample size benefits complex models like Random Forest, Gradient Boosting, and XGBoost.")
        else:
            algorithm_scores["random_forest_regressor"] += 2
            algorithm_scores["gradient_boosting_regressor"] += 2
            algorithm_scores["xgboost_regressor"] += 2
            algorithm_scores["mlp_regressor"] += 1
            reasons.append("Large sample size benefits complex models like Random Forest, Gradient Boosting, and XGBoost.")
    
    # Feature count considerations
    if dataset_info["num_features"] < 10:
        if target_type == "classification":
            algorithm_scores["logistic_regression"] += 1
            algorithm_scores["naive_bayes"] += 1
            reasons.append("Few features work well with simple models.")
        else:
            algorithm_scores["linear_regression"] += 1
            algorithm_scores["ridge_regression"] += 1
            reasons.append("Few features work well with simple models.")
    elif dataset_info["num_features"] > 50:
        if target_type == "classification":
            algorithm_scores["random_forest_classifier"] += 1
            algorithm_scores["xgboost_classifier"] += 1
            algorithm_scores["gradient_boosting_classifier"] += 1
            reasons.append("Many features benefit from models with built-in feature selection like Random Forest and boosting methods.")
        else:
            algorithm_scores["random_forest_regressor"] += 1
            algorithm_scores["xgboost_regressor"] += 1
            algorithm_scores["lasso_regression"] += 1
            algorithm_scores["elastic_net"] += 1
            reasons.append("Many features benefit from models with built-in feature selection or regularization.")
    
    # Handle class imbalance
    if dataset_info["is_imbalanced"] and target_type == "classification":
        algorithm_scores["random_forest_classifier"] += 1
        algorithm_scores["xgboost_classifier"] += 2
        algorithm_scores["gradient_boosting_classifier"] += 2
        algorithm_scores["logistic_regression"] -= 1
        algorithm_scores["naive_bayes"] -= 1
        reasons.append("Class imbalance favors ensemble methods and penalizes probabilistic models.")
    
    # Handle missing values
    if dataset_info["has_missing_values"]:
        if target_type == "classification":
            algorithm_scores["random_forest_classifier"] += 1
            algorithm_scores["xgboost_classifier"] += 1
            algorithm_scores["gradient_boosting_classifier"] += 1
        else:
            algorithm_scores["random_forest_regressor"] += 1
            algorithm_scores["xgboost_regressor"] += 1
            algorithm_scores["gradient_boosting_regressor"] += 1
        reasons.append("Missing values are better handled by tree-based methods.")
    
    # Handle outliers
    if dataset_info["has_outliers"]:
        if target_type == "classification":
            algorithm_scores["random_forest_classifier"] += 1
            algorithm_scores["xgboost_classifier"] += 1
            algorithm_scores["gradient_boosting_classifier"] += 1
            algorithm_scores["svm_classifier"] -= 1
            algorithm_scores["logistic_regression"] -= 1
        else:
            algorithm_scores["random_forest_regressor"] += 1
            algorithm_scores["xgboost_regressor"] += 1
            algorithm_scores["gradient_boosting_regressor"] += 1
            algorithm_scores["svm_regressor"] -= 1
            algorithm_scores["linear_regression"] -= 1
        reasons.append("Outliers are better handled by tree-based methods and can negatively impact linear models.")
    
    # Handle high dimensionality
    if dataset_info["high_dimensionality"]:
        if target_type == "classification":
            algorithm_scores["random_forest_classifier"] += 1
            algorithm_scores["xgboost_classifier"] += 1
            algorithm_scores["svm_classifier"] += 1
            algorithm_scores["logistic_regression"] -= 1
        else:
            algorithm_scores["random_forest_regressor"] += 1
            algorithm_scores["xgboost_regressor"] += 1
            algorithm_scores["ridge_regression"] += 1
            algorithm_scores["lasso_regression"] += 1
            algorithm_scores["elastic_net"] += 1
            algorithm_scores["linear_regression"] -= 1
        reasons.append("High dimensionality favors models with built-in feature selection or regularization.")
    
    # Handle skewed features
    if dataset_info["has_skewed_features"]:
        if target_type == "classification":
            algorithm_scores["random_forest_classifier"] += 1
            algorithm_scores["xgboost_classifier"] += 1
            algorithm_scores["gradient_boosting_classifier"] += 1
            algorithm_scores["logistic_regression"] -= 1
        else:
            algorithm_scores["random_forest_regressor"] += 1
            algorithm_scores["xgboost_regressor"] += 1
            algorithm_scores["gradient_boosting_regressor"] += 1
            algorithm_scores["linear_regression"] -= 1
        reasons.append("Skewed features are better handled by tree-based methods.")
    
    # Select the highest scoring algorithm
    recommended_algo = max(algorithm_scores, key=algorithm_scores.get)
    
    # Create explanation
    explanation = f"Recommended algorithm: {recommended_algo}\n\nReasoning:\n"
    for reason in reasons:
        explanation += f"- {reason}\n"
    
    if verbose:
        print("\n=== Algorithm Recommendation ===")
        print(explanation)
        print("\nTop 3 algorithms by score:")
        sorted_algos = sorted(algorithm_scores.items(), key=lambda x: x[1], reverse=True)[:3]
        for algo, score in sorted_algos:
            print(f"{algo}: {score}")
    
    return recommended_algo, explanation


def handle_missing_values(df, strategy='auto'):
    """
    Automatically fills missing values in numerical and categorical columns.
    
    Parameters:
    - df: pandas DataFrame
    - strategy: str, 'auto', 'mean', 'median', 'most_frequent', or 'knn'
    
    Returns:
    - pandas DataFrame with missing values filled
    """
    # Make a copy to avoid modifying the original dataframe
    df_copy = df.copy()
    
    # Drop columns with too many missing values
    threshold = 0.4 * len(df_copy)
    df_copy = df_copy.dropna(thresh=threshold, axis=1)
    
    # Get column types
    num_cols = df_copy.select_dtypes(include=['number']).columns.tolist()
    cat_cols = df_copy.select_dtypes(exclude=['number']).columns.tolist()
    
    # Auto-detect best strategy for numerical columns
    if strategy == 'auto':
        if len(num_cols) > 0:
            # Check for outliers to determine if median is better than mean
            has_outliers = False
            for col in num_cols:
                if df_copy[col].notnull().sum() > 0:  # Check if there are non-null values
                    Q1 = df_copy[col].quantile(0.25)
                    Q3 = df_copy[col].quantile(0.75)
                    IQR = Q3 - Q1
                    outlier_ratio = ((df_copy[col] < (Q1 - 1.5 * IQR)) | (df_copy[col] > (Q3 + 1.5 * IQR))).mean()
                    if outlier_ratio > 0.05:
                        has_outliers = True
                        break
            
            num_strategy = 'median' if has_outliers else 'mean'
        else:
            num_strategy = 'mean'  # Default
    else:
        num_strategy = strategy
    
    # Fill numerical columns
    if num_cols:
        if num_strategy == 'knn':
            imputer = KNNImputer(n_neighbors=5)
            df_copy.loc[:, num_cols] = imputer.fit_transform(df_copy[num_cols])
        else:
            imputer = SimpleImputer(strategy=num_strategy)
            df_copy.loc[:, num_cols] = imputer.fit_transform(df_copy[num_cols])
    
    # Fill categorical columns
    if cat_cols:
        cat_imputer = SimpleImputer(strategy='most_frequent')
        df_copy.loc[:, cat_cols] = cat_imputer.fit_transform(df_copy[cat_cols])
    
    return df_copy


def preprocess_data(X_train, X_test, y_train, target_type, dataset_info=None):
    """
    Preprocess features with scaling, feature selection, and dimensionality reduction.
    
    Parameters:
    - X_train: Training features
    - X_test: Test features
    - y_train: Training target values
    - target_type: 'classification' or 'regression'
    - dataset_info: Dictionary with dataset characteristics
    
    Returns:
    - X_train_processed, X_test_processed: Processed feature sets
    """
    # Handle different preprocessing based on dataset characteristics
    if dataset_info and dataset_info["has_outliers"]:
        # Use RobustScaler for datasets with outliers
        scaler = RobustScaler()
    else:
        scaler = StandardScaler()
    
    # Feature selection method
    if dataset_info and dataset_info["high_dimensionality"]:
        # Use more aggressive feature selection for high-dimensional data
        k = min(int(X_train.shape[1] * 0.5), X_train.shape[1])
    else:
        k = min(int(X_train.shape[1] * 0.8), X_train.shape[1])
    
    if k < 1:
        k = 1  # Ensure at least one feature is selected
    
    if target_type == "classification":
        selector = SelectKBest(f_classif, k=k)
    else:
        selector = SelectKBest(f_regression, k=k)
    
    # Apply PCA for dimensionality reduction
    apply_pca = False
    if dataset_info:
        # Only apply PCA for high-dimensional data with many numerical features
        if dataset_info["high_dimensionality"] and dataset_info["num_numerical"] > 10:
            apply_pca = True
    
    # Scale features
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Select features
    X_train_selected = selector.fit_transform(X_train_scaled, y_train)
    X_test_selected = selector.transform(X_test_scaled)
    
    # Apply PCA if needed
    if apply_pca:
        pca = PCA(n_components=0.95)  # Keep 95% variance
        X_train_processed = pca.fit_transform(X_train_selected)
        X_test_processed = pca.transform(X_test_selected)
        return X_train_processed, X_test_processed
    else:
        return X_train_selected, X_test_selected


def encode_categorical_features(df):
    """
    Encode categorical features using appropriate encoding techniques.
    """
    df_encoded = df.copy()
    cat_cols = df_encoded.select_dtypes(include=['object', 'category']).columns.tolist()
    
    for col in cat_cols:
        unique_values = df_encoded[col].nunique()
        if unique_values <= 10:  # Low cardinality
            le = LabelEncoder()
            df_encoded[col] = le.fit_transform(df_encoded[col])
        else:  # High cardinality
            # For high cardinality, you might want to use target encoding or other methods
            # Here's a simple approach - convert to category codes
            df_encoded[col] = df_encoded[col].astype('category').cat.codes
    
    return df_encoded


# Define all possible models
MODEL_MAPPING = {
    # Classification models
    "logistic_regression": LogisticRegression(max_iter=1000),
    "random_forest_classifier": RandomForestClassifier(),
    "svm_classifier": SVC(probability=True),
    "decision_tree_classifier": DecisionTreeClassifier(),
    "gradient_boosting_classifier": GradientBoostingClassifier(),
    "xgboost_classifier": XGBClassifier(),
    "knn_classifier": KNeighborsClassifier(),
    "naive_bayes": GaussianNB(),
    "mlp_classifier": MLPClassifier(max_iter=500),
    
    # Regression models
    "linear_regression": LinearRegression(),
    "ridge_regression": Ridge(),
    "lasso_regression": Lasso(),
    "elastic_net": ElasticNet(),
    "random_forest_regressor": RandomForestRegressor(),
    "svm_regressor": SVR(),
    "decision_tree_regressor": DecisionTreeRegressor(),
    "gradient_boosting_regressor": GradientBoostingRegressor(),
    "xgboost_regressor": XGBRegressor(),
    "knn_regressor": KNeighborsRegressor(),
    "mlp_regressor": MLPRegressor(max_iter=500)
}

def create_ensemble_model(X_train, y_train, top_models, is_classification):
    """
    Create a voting ensemble from the best performing models.
    
    Parameters:
    - X_train: Training features
    - y_train: Training target values
    - top_models: List of (model, score) tuples
    - is_classification: Boolean indicating if this is a classification task
    
    Returns:
    - Trained ensemble model
    """
    from sklearn.ensemble import VotingClassifier, VotingRegressor
    
    # Extract models from tuples and give them names
    named_models = [(f"model_{i}", model) for i, (model, _) in enumerate(top_models)]
    
    if is_classification:
        # For classifiers, we can use soft voting (using predicted probabilities)
        try:
            ensemble = VotingClassifier(named_models, voting='soft')
        except:
            # Fallback to hard voting if soft voting isn't supported by all models
            ensemble = VotingClassifier(named_models, voting='hard')
    else:
        # For regressors, we use standard voting
        ensemble = VotingRegressor(named_models)
    
    # Train the ensemble model
    ensemble.fit(X_train, y_train)
    
    return ensemble



def tune_hyperparameters(model, X_train, y_train, is_classification, n_trials=10):
    """
    Performs hyperparameter tuning using Bayesian optimization (Optuna).
    
    Parameters:
    - model: sklearn estimator
    - X_train: Training features
    - y_train: Training target values
    - is_classification: bool, True if classification task
    - n_trials: int, number of optimization trials
    
    Returns:
    - model with optimized hyperparameters
    """
    # Define the objective function based on model type
    def objective(trial):
        params = {}
        
        # Linear models
        if isinstance(model, (LogisticRegression, LinearRegression, Ridge, Lasso, ElasticNet)):
            params["fit_intercept"] = trial.suggest_categorical("fit_intercept", [True, False])
            
            if isinstance(model, LogisticRegression):
                params["C"] = trial.suggest_float("C", 0.01, 10.0, log=True)
                params["solver"] = trial.suggest_categorical("solver", ["liblinear", "saga"])
            
            if isinstance(model, (Ridge, Lasso, ElasticNet)):
                params["alpha"] = trial.suggest_float("alpha", 0.01, 10.0, log=True)
            
            if isinstance(model, ElasticNet):
                params["l1_ratio"] = trial.suggest_float("l1_ratio", 0.1, 0.9)
        
        # Tree-based models
        elif isinstance(model, (RandomForestClassifier, RandomForestRegressor, 
                               DecisionTreeClassifier, DecisionTreeRegressor)):
            params["max_depth"] = trial.suggest_int("max_depth", 3, 20)
            params["min_samples_split"] = trial.suggest_int("min_samples_split", 2, 20)
            
            if isinstance(model, (RandomForestClassifier, RandomForestRegressor)):
                params["n_estimators"] = trial.suggest_int("n_estimators", 50, 300)
                params["max_features"] = trial.suggest_categorical("max_features", ["sqrt", "log2", None])
        
        # Boosting models
        elif isinstance(model, (GradientBoostingClassifier, GradientBoostingRegressor, 
                               XGBClassifier, XGBRegressor)):
            params["learning_rate"] = trial.suggest_float("learning_rate", 0.01, 0.3, log=True)
            params["max_depth"] = trial.suggest_int("max_depth", 3, 10)
            params["n_estimators"] = trial.suggest_int("n_estimators", 50, 300)
            
            if isinstance(model, (XGBClassifier, XGBRegressor)):
                params["subsample"] = trial.suggest_float("subsample", 0.5, 1.0)
                params["colsample_bytree"] = trial.suggest_float("colsample_bytree", 0.5, 1.0)
        
        # SVM models
        elif isinstance(model, (SVC, SVR)):
            params["C"] = trial.suggest_float("C", 0.1, 100.0, log=True)
            params["kernel"] = trial.suggest_categorical("kernel", ["linear", "rbf", "poly"])
            params["gamma"] = trial.suggest_categorical("gamma", ["scale", "auto"])
        
        # KNN models
        elif isinstance(model, (KNeighborsClassifier, KNeighborsRegressor)):
            params["n_neighbors"] = trial.suggest_int("n_neighbors", 3, 20)
            params["weights"] = trial.suggest_categorical("weights", ["uniform", "distance"])
            params["algorithm"] = trial.suggest_categorical("algorithm", ["auto", "ball_tree", "kd_tree", "brute"])
        
        # Neural Network models
        elif isinstance(model, (MLPClassifier, MLPRegressor)):
            params["hidden_layer_sizes"] = (trial.suggest_int("hidden_layer_size", 10, 100),)
            params["alpha"] = trial.suggest_float("alpha", 0.0001, 0.01, log=True)
            params["learning_rate_init"] = trial.suggest_float("learning_rate_init", 0.001, 0.1, log=True)
            params["max_iter"] = 1000  # Ensure enough iterations
        
        # Gaussian Naive Bayes doesn't have key hyperparameters to tune
        
        # Apply parameters to the model
        if params:
            model.set_params(**params)
        
        # Perform cross-validation
        try:
            scores = cross_val_score(
                model, X_train, y_train, cv=3, 
                scoring='accuracy' if is_classification else 'neg_mean_squared_error'
            )
            return np.mean(scores)
        except Exception as e:
            # Return a poor score if there's an error
            return -1e6 if not is_classification else 0.0
    
    # Create and run the optimization study
    direction = "maximize" if is_classification else "minimize"
    study = optuna.create_study(direction=direction)
    study.optimize(objective, n_trials=n_trials)
    
    # Set the best parameters
    best_params = study.best_trial.params
    if best_params:
        model.set_params(**best_params)
    
    return model


def train_and_evaluate(df, target_column, algorithm_name=None, dataset_info=None):
    """
    Trains and evaluates the model based on classification or regression.
    
    Parameters:
    - df: pandas DataFrame
    - target_column: string, name of the target column
    - algorithm_name: string, name of the algorithm to use (optional)
    - dataset_info: dictionary with dataset characteristics (optional)
    
    Returns:
    - tuple: (primary_score, additional_metrics, trained_model)
    """
    # Ensure target column exists
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in the DataFrame.")
    
    # Encode categorical features
    df_encoded = encode_categorical_features(df)
    
    # Split data
    X = df_encoded.drop(columns=[target_column])
    y = df_encoded[target_column]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Determine problem type
    is_classification = y.nunique() <= 10
    target_type = "classification" if is_classification else "regression"
    
    # If algorithm is not specified, recommend one
    if algorithm_name is None:
        if dataset_info is None:
            dataset_info = analyze_dataset(df, target_column, verbose=False)
        algorithm_name, _ = recommend_algorithm(dataset_info, verbose=False)
    
    # Preprocess the data
    X_train_processed, X_test_processed = preprocess_data(X_train, X_test, y_train, target_type, dataset_info)
    
    # Get the model
    if algorithm_name in MODEL_MAPPING:
        model = MODEL_MAPPING[algorithm_name]
    else:
        raise ValueError(f"Algorithm '{algorithm_name}' not found in available models.")
    
    # Tune hyperparameters
    try:
        tuned_model = tune_hyperparameters(model, X_train_processed, y_train, is_classification, n_trials=10)
    except Exception as e:
        print(f"Hyperparameter tuning failed: {e}")
        tuned_model = model  # Use the default model if tuning fails
    
    # Train the model
    tuned_model.fit(X_train_processed, y_train)
    
    # Make predictions
    y_pred = tuned_model.predict(X_test_processed)
    
    # Evaluate performance based on problem type
    if is_classification:
        # For classification
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        
        primary_score = accuracy
        additional_metrics = {
            'accuracy': accuracy,
            'f1_score': f1,
            'precision': precision,
            'recall': recall
        }
        
        print("\n=== Classification Results ===")
        print(f"Algorithm: {algorithm_name}")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
    else:
        # For regression
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)
        
        primary_score = -rmse  # Negative because higher is better in optimization
        additional_metrics = {
            'rmse': rmse,
            'mse': mse,
            'r2': r2
        }
        
        print("\n=== Regression Results ===")
        print(f"Algorithm: {algorithm_name}")
        print(f"RMSE: {rmse:.4f}")
        print(f"MSE: {mse:.4f}")
        print(f"R²: {r2:.4f}")
    
    return primary_score, additional_metrics, tuned_model

def interpret_model(model, X, feature_names):
    """
    Provide interpretation of model predictions using SHAP values or feature importance.
    
    Parameters:
    - model: Trained model
    - X: Feature matrix
    - feature_names: List of feature names
    
    Returns:
    - Dictionary with interpretation results
    """
    interpretation = {}
    
    # Try direct feature importance first (works for tree-based models)
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        # Create feature importance dictionary
        sorted_features = [(feature_names[i], float(importances[i])) for i in indices]
        interpretation['feature_importance'] = sorted_features
        
        # Calculate cumulative importance and identify key features
        cumulative_importance = 0
        key_features = []
        for feature, importance in sorted_features:
            cumulative_importance += importance
            key_features.append(feature)
            if cumulative_importance > 0.8:  # Cover 80% of importance
                break
        
        interpretation['key_features'] = key_features
        interpretation['num_key_features'] = len(key_features)
        
    # For linear models, extract coefficients
    elif hasattr(model, 'coef_'):
        if model.coef_.ndim == 1:
            # Linear regression, logistic regression, etc.
            coefficients = model.coef_
            # Get absolute values and sort
            abs_coeffs = np.abs(coefficients)
            indices = np.argsort(abs_coeffs)[::-1]
            
            # Create coefficient dictionary
            sorted_features = [(feature_names[i], float(coefficients[i])) for i in indices]
            interpretation['feature_coefficients'] = sorted_features
            
            # Identify key features with highest absolute coefficients
            key_features = [feature_names[i] for i in indices[:min(5, len(feature_names))]]
            interpretation['key_features'] = key_features
            interpretation['num_key_features'] = len(key_features)
    
    return interpretation

def engineer_features(df, target_column):
    """
    Automatically generate new features that might improve model performance.
    
    Parameters:
    - df: pandas DataFrame
    - target_column: string, name of the target column
    
    Returns:
    - DataFrame with additional engineered features
    """
    df_result = df.copy()
    
    # Select only numerical columns
    num_cols = df_result.select_dtypes(include=['number']).columns.tolist()
    if target_column in num_cols:
        num_cols.remove(target_column)
    
    # Skip if too few numerical columns
    if len(num_cols) < 2:
        return df_result
    
    # Feature creation counters
    created_features = 0
    max_features = min(10, len(num_cols) * 2)  # Limit new features
    
    # 1. Create polynomial features for highly correlated features with target
    if target_column in df.columns and len(num_cols) > 0:
        # Ensure target column is numerical for correlation calculation
        if df[target_column].dtype in ['object', 'category']:
            # If target is categorical, skip polynomial feature creation
            pass
        else:
            # Compute correlations only for numerical columns
            correlations = df[num_cols + [target_column]].corr()[target_column].abs().sort_values(ascending=False)
            top_correlated = correlations.index.tolist()
            if target_column in top_correlated:
                top_correlated.remove(target_column)
            
            # Use top 3 correlated features
            for col in top_correlated[:3]:
                if created_features < max_features:
                    df_result[f'{col}_squared'] = df[col] ** 2
                    created_features += 1
    
    # 2. Create interaction features between top numerical features
    if len(num_cols) >= 2:
        for i in range(min(3, len(num_cols))):
            for j in range(i+1, min(4, len(num_cols))):
                if created_features < max_features:
                    df_result[f'{num_cols[i]}_x_{num_cols[j]}'] = df[num_cols[i]] * df[num_cols[j]]
                    created_features += 1
    
    # 3. Create ratio features for appropriate numerical columns
    if len(num_cols) >= 2:
        for i in range(min(2, len(num_cols))):
            for j in range(i+1, min(3, len(num_cols))):
                if created_features < max_features and (df[num_cols[j]] != 0).all():
                    df_result[f'{num_cols[i]}_div_{num_cols[j]}'] = df[num_cols[i]] / df[num_cols[j]].replace(0, 1)
                    created_features += 1
    
    return df_result

def run_automl(df, target_column):
    print("\n========= AutoML Pipeline Started =========\n")
    
    # Step 1: Handle missing values
    df_clean = handle_missing_values(df)
    print("\n✓ Missing values handled")
    
    # Step 2: Feature Engineering (NEW)
    df_engineered = engineer_features(df_clean, target_column)
    print("\n✓ Feature engineering completed")
    print(f"  - Added {df_engineered.shape[1] - df_clean.shape[1]} new features")
    
    # Step 3: Analyze dataset
    dataset_info = analyze_dataset(df_engineered, target_column)
    print("\n✓ Dataset analysis completed")
    
    # Step 4: Recommend algorithm
    recommended_algo, explanation = recommend_algorithm(dataset_info)
    print("\n✓ Algorithm recommendation completed")
    
    # Step 5: Train and evaluate the model (now includes ensemble options)
    primary_score, metrics, model = train_and_evaluate(
        df_engineered, target_column, None, dataset_info  # Pass None to try multiple algorithms
    )
    print("\n✓ Model training and evaluation completed")
    
    # Step 6: Model interpretation (NEW)
    try:
        feature_names = df_engineered.drop(columns=[target_column]).columns.tolist()
        interpretation = interpret_model(model, df_engineered.drop(columns=[target_column]), feature_names)
        
        if interpretation and 'key_features' in interpretation:
            print("\n=== Model Interpretation ===")
            print("Key influential features:")
            for i, feature in enumerate(interpretation['key_features'][:5], 1):
                print(f"{i}. {feature}")
        
        print("\n✓ Model interpretation completed")
    except Exception as e:
        print(f"\nWarning: Model interpretation failed with error: {e}")
    
    print("\n========= AutoML Pipeline Completed =========\n")
    
    return model, metrics, dataset_info, interpretation if 'interpretation' in locals() else None


# Example usage
if __name__ == "__main__":

    df = pd.read_csv('/Users/tunguturiuday/Desktop/algorithm identifier project/data/Wholesale customers data.csv')
    model, metrics, dataset_info, interpretation = run_automl(df, 'Delicassen')