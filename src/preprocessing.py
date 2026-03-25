"""
Preprocessing Module for Fraud Detection
Handles scaling, transformation, and final preprocessing steps
"""

from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.impute import SimpleImputer
import numpy as np


class Preprocessor:
    """
    Preprocessing pipeline for fraud detection models
    """
    
    def __init__(self, scaler_type='robust'):
        self.scaler_type = scaler_type
        self.scaler = None
        self.imputer = None
        
    def get_scaler(self):
        """
        Get appropriate scaler based on type
        """
        if self.scaler_type == 'standard':
            return StandardScaler()
        elif self.scaler_type == 'minmax':
            return MinMaxScaler()
        elif self.scaler_type == 'robust':
            return RobustScaler()
        else:
            return StandardScaler()
    
    def fit_transform(self, X_train, X_test=None):
        """
        Fit scaler on training data and transform both train and test
        """
        self.scaler = self.get_scaler()
        
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        if X_test is not None:
            X_test_scaled = self.scaler.transform(X_test)
            return X_train_scaled, X_test_scaled
        
        return X_train_scaled
    
    def transform(self, X):
        """
        Transform data using fitted scaler
        """
        if self.scaler is None:
            raise ValueError("Scaler not fitted. Call fit_transform first.")
        
        return self.scaler.transform(X)
    
    def handle_imbalance_smote(self, X_train, y_train, random_state=42):
        """
        Apply SMOTE (Synthetic Minority Over-sampling Technique)
        
        SMOTE creates synthetic samples of the minority class by:
        1. Selecting a minority class sample randomly
        2. Finding its k nearest neighbors (default k=5)
        3. Randomly selecting one of the k neighbors
        4. Generating a new sample along the line between the two
        
        This helps the model learn better decision boundaries for
        the minority class without simply duplicating existing samples.
        """
        try:
            from imblearn.over_sampling import SMOTE
        except ImportError:
            print("imbalanced-learn not installed. Install with: pip install imbalanced-learn")
            return X_train, y_train
        
        print("\n--- Applying SMOTE ---")
        print(f"Before SMOTE:")
        print(f"  Class 0 (Legitimate): {sum(y_train == 0)}")
        print(f"  Class 1 (Fraud): {sum(y_train == 1)}")
        
        smote = SMOTE(random_state=random_state)
        X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
        
        print(f"\nAfter SMOTE:")
        print(f"  Class 0 (Legitimate): {sum(y_resampled == 0)}")
        print(f"  Class 1 (Fraud): {sum(y_resampled == 1)}")
        
        return X_resampled, y_resampled
    
    def handle_imbalance_undersampling(self, X_train, y_train, random_state=42):
        """
        Apply Random Under Sampling
        
        Randomly removes samples from the majority class to match
        the minority class count. Simple but loses information.
        
        Pros:
        - Simple to implement
        - Reduces training time
        
        Cons:
        - Loses valuable information from majority class
        - May create biased sample if random selection is unlucky
        """
        from imblearn.under_sampling import RandomUnderSampler
        
        print("\n--- Applying Random Under Sampling ---")
        print(f"Before Under Sampling:")
        print(f"  Class 0 (Legitimate): {sum(y_train == 0)}")
        print(f"  Class 1 (Fraud): {sum(y_train == 1)}")
        
        rus = RandomUnderSampler(random_state=random_state)
        X_resampled, y_resampled = rus.fit_resample(X_train, y_train)
        
        print(f"\nAfter Under Sampling:")
        print(f"  Class 0 (Legitimate): {sum(y_resampled == 0)}")
        print(f"  Class 1 (Fraud): {sum(y_resampled == 1)}")
        
        return X_resampled, y_resampled
    
    def handle_imbalance_smoteenn(self, X_train, y_train, random_state=42):
        """
        Apply SMOTE + ENN (Edited Nearest Neighbors)
        
        Combines oversampling with cleaning:
        - SMOTE creates synthetic minority samples
        - ENN removes samples that are misclassified by NN
        
        This results in a cleaner, more distinct decision boundary
        """
        from imblearn.combine import SMOTEENN
        
        print("\n--- Applying SMOTE + ENN ---")
        print(f"Before SMOTEENN:")
        print(f"  Class 0 (Legitimate): {sum(y_train == 0)}")
        print(f"  Class 1 (Fraud): {sum(y_train == 1)}")
        
        smoteenn = SMOTEENN(random_state=random_state)
        X_resampled, y_resampled = smoteenn.fit_resample(X_train, y_train)
        
        print(f"\nAfter SMOTEENN:")
        print(f"  Class 0 (Legitimate): {sum(y_resampled == 0)}")
        print(f"  Class 1 (Fraud): {sum(y_resampled == 1)}")
        
        return X_resampled, y_resampled


class CostSensitiveLearning:
    """
    Cost-sensitive learning to handle class imbalance
    
    Instead of changing the data distribution, we modify the
    learning algorithm to penalize misclassification differently
    for each class.
    
    In fraud detection:
    - False Negative (missed fraud): HIGH cost (financial loss)
    - False Positive (blocked legitimate): LOW cost (customer inconvenience)
    
    We assign higher weight to minority class to make the model
    more sensitive to detecting fraud.
    """
    
    def __init__(self):
        pass
    
    def compute_class_weights(self, y_train):
        """
        Compute balanced class weights
        
        Weight = n_samples / (n_classes * n_samples_for_class)
        
        This gives higher weight to the minority class
        """
        from sklearn.utils.class_weight import compute_class_weight
        
        classes = np.unique(y_train)
        weights = compute_class_weight('balanced', classes=classes, y=y_train)
        
        class_weights = dict(zip(classes, weights))
        
        print("\n--- Class Weights ---")
        for cls, weight in class_weights.items():
            label = "Legitimate" if cls == 0 else "Fraud"
            print(f"  Class {cls} ({label}): {weight:.4f}")
        
        return class_weights
    
    def get_sample_weights(self, y_train, class_weights=None):
        """
        Generate sample weights for training
        """
        if class_weights is None:
            class_weights = self.compute_class_weights(y_train)
        
        sample_weights = np.array([class_weights[y] for y in y_train])
        
        return sample_weights


def compare_sampling_methods(X_train, y_train):
    """
    Compare different sampling methods
    
    This helps understand which approach works best for the dataset
    """
    preprocessor = Preprocessor()
    
    results = {
        'Original': len(y_train),
    }
    
    try:
        from imblearn.over_sampling import SMOTE
        smote = SMOTE(random_state=42)
        X_res, y_res = smote.fit_resample(X_train, y_train)
        results['SMOTE'] = len(y_res)
    except:
        pass
    
    try:
        from imblearn.under_sampling import RandomUnderSampler
        rus = RandomUnderSampler(random_state=42)
        X_res, y_res = rus.fit_resample(X_train, y_train)
        results['Under Sampling'] = len(y_res)
    except:
        pass
    
    print("\n--- Sampling Method Comparison ---")
    for method, count in results.items():
        print(f"{method}: {count} samples")
    
    return results


if __name__ == "__main__":
    from src.data_pipeline import main
    
    pipeline, X_train, X_test, y_train, y_test = main()
    
    preprocessor = Preprocessor(scaler_type='robust')
    X_train_scaled, X_test_scaled = preprocessor.fit_transform(X_train, X_test)
    
    print("\n--- Comparing Sampling Methods ---")
    compare_sampling_methods(X_train_scaled, y_train)
