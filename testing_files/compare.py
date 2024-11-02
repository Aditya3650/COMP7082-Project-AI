from sentence_transformers import CrossEncoder
import torch

# Load the CrossEncoder with sigmoid activation to bound the outputs between 0 and 1
model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", default_activation_function=torch.nn.Sigmoid())

# Define sentence pairs
sentence_pairs = [
    ("Hi, my name is Aditya", 
    "Good morning, I am Aditya, friend of Adam"),
]

# Predict similarity scores
scores = model.predict(sentence_pairs)

# Convert the score to a percentage and print it
similarity_percentage = scores[0] * 100
print(f"Similarity Score: {similarity_percentage:.2f}%")
