import numpy as np
import os
import pickle
import json

class ModelService:
    """
    A lightweight model service that can load a saved model or use a fallback 
    prediction mechanism if TensorFlow is not available
    """
    
    def __init__(self, model_path='model.h5', fallback_path='model_fallback.pkl'):
        self.model = None
        self.fallback_model = None
        self.fallback_path = fallback_path
        self.model_path = model_path
        
        # Try to load the model using the preferred method
        try:
            # Try to import TensorFlow - if available
            from tensorflow.keras.models import load_model
            self.model = load_model(model_path)
            print(f"TensorFlow model loaded successfully from {model_path}")
        except ImportError:
            print("TensorFlow not available. Will use fallback prediction.")
        except Exception as e:
            print(f"Error loading TensorFlow model: {e}")
        
        # If TensorFlow model loading failed, try fallback
        if self.model is None:
            try:
                if os.path.exists(fallback_path):
                    with open(fallback_path, 'rb') as f:
                        self.fallback_model = pickle.load(f)
                    print(f"Fallback model loaded from {fallback_path}")
                else:
                    print(f"Fallback model not found at {fallback_path}. Using basic rules.")
            except Exception as e:
                print(f"Error loading fallback model: {e}")
    
    def predict(self, features):
        """
        Make a prediction with loaded model or fallback logic
        
        Args:
            features: Numpy array of features in the same format expected by model
            
        Returns:
            Prediction array with values between 0-1
        """
        if self.model is not None:
            # Use TensorFlow model if available
            return self.model.predict(features)
        elif self.fallback_model is not None:
            # Use fallback model if available
            return self.fallback_model.predict(features)
        else:
            # Use basic rule-based logic as fallback
            # This is a simplified logic based on your model's features
            experience_score = features[0][0]  # experience_score
            education_score = features[0][1]   # education_score
            skill_score = features[0][2]       # skill_score
            interview_score = features[0][3]   # interview_score
            
            # Simple weighted formula similar to what your model might have learned
            # Adjust these weights based on your model's behavior
            prediction = (
                experience_score * 0.3 + 
                education_score * 0.25 + 
                skill_score * 0.25 + 
                interview_score * 0.2
            )
            
            # Return in the same format as the model would (array with single value)
            return np.array([[prediction]])