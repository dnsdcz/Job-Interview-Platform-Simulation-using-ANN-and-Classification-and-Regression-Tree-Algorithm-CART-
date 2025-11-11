import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

# Load your logged applications CSV
df = pd.read_csv('logged_applications.csv')

# Updated preprocess function to match the logic in routes.py
def preprocess(df):
    # Education mapping - similar to the one in routes.py
    education_map = {
        "high_school": 1,
        "vocational": 2,
        "associate": 3,
        "bachelor": 4,
        "master": 5,
        "phd": 6
    }
    
    # Process features similar to routes.py
    processed_data = []
    for _, row in df.iterrows():
        # Extract education score
        education_score = education_map.get(row['education'], 1)
        
        # Convert experience to numeric (handling format like "2 years" or just "2")
        experience_text = str(row['experience'])
        if 'years' in experience_text or 'year' in experience_text:
            experience_years = int(experience_text.split()[0])
        else:
            experience_years = int(experience_text)
        experience_score = min(15, experience_years) / 15
        
        # Skills assessment
        skills_list = [s.strip() for s in str(row['skills']).split(',') if s.strip()]
        skill_count = len(skills_list)
        skill_score = min(10, skill_count) / 10
        
        # Interview score proxy (as in routes.py)
        interview_score = (skill_score * 0.6) + (experience_score * 0.4)
        
        processed_data.append([experience_score, education_score/6, skill_score, interview_score])
    
    X = np.array(processed_data)
    y = df['eligible'].values
    
    return X, y

# Get preprocessed data
X, y = preprocess(df)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define model (keeping your original neural network architecture)
model = Sequential([
    Dense(16, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(8, activation='relu'),
    Dense(1, activation='sigmoid')  # Binary classification
])

# Compile
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train
model.fit(X_train, y_train, epochs=20, validation_data=(X_test, y_test))

# Save model
model.save('model.h5')