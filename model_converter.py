"""
model_converter.py - Convert a TensorFlow model to a pickle file

This script creates a scikit-learn-compatible model that approximates
your TensorFlow model's behavior. Run this script once on a machine
with TensorFlow installed to create the fallback model.
"""

import numpy as np
import pickle
import os
import argparse

def convert_model(tf_model_path='model.h5', output_path='model_fallback.pkl'):
    """
    Converts a TensorFlow model to a scikit-learn model that can be used without TensorFlow.
    
    Args:
        tf_model_path: Path to the TensorFlow model (.h5 file)
        output_path: Path where the pickle model will be saved
    """
    try:
        import tensorflow as tf
        from tensorflow.keras.models import load_model
        from sklearn.linear_model import LogisticRegression
        
        print(f"Loading TensorFlow model from {tf_model_path}...")
        model = load_model(tf_model_path)
        
        print("Creating synthetic dataset for training the fallback model...")
        # Create a synthetic dataset covering the feature space
        n_samples = 10000
        X_synthetic = np.random.rand(n_samples, 4)  # 4 features
        
        # Get predictions from the TensorFlow model
        print("Getting predictions from TensorFlow model...")
        y_proba = model.predict(X_synthetic)
        y_synthetic = (y_proba > 0.5).astype(int).ravel()
        
        print("Training scikit-learn model on synthetic data...")
        # Train a logistic regression model on the synthetic data
        fallback_model = LogisticRegression(max_iter=1000)
        fallback_model.fit(X_synthetic, y_synthetic)
        
        # Save the fallback model
        print(f"Saving fallback model to {output_path}...")
        with open(output_path, 'wb') as f:
            pickle.dump(fallback_model, f)
        
        print("Model conversion completed successfully!")
        
        # Test the models to ensure they produce similar output
        print("\nTesting models with sample data...")
        
        test_data = np.array([[0.6, 0.8, 0.7, 0.5]])  # Sample test data
        
        tf_prediction = model.predict(test_data)
        fallback_prediction = fallback_model.predict_proba(test_data)[:, 1]
        
        print(f"TensorFlow model prediction: {tf_prediction[0][0]}")
        print(f"Fallback model prediction: {fallback_prediction[0]}")
        print(f"Difference: {abs(tf_prediction[0][0] - fallback_prediction[0]):.6f}")
        
        return True
        
    except ImportError:
        print("Error: TensorFlow is not installed. This script requires TensorFlow to convert the model.")
        return False
    except Exception as e:
        print(f"Error during model conversion: {e}")
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Convert TensorFlow model to pickle format')
    parser.add_argument('--input', default='model.h5', help='Path to input TensorFlow model')
    parser.add_argument('--output', default='model_fallback.pkl', help='Path for output pickle model')
    
    args = parser.parse_args()
    convert_model(args.input, args.output)