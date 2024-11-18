import cv2
import numpy as np
import tensorflow as tf
import requests
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input

# Example dictionary for calorie estimation (replace with actual database or API)
calorie_database = {
    'apple': 52,  # Calorie content per 100 grams
    'banana': 96,
    'pizza': 266,
    'burger': 295
}

# Load the ResNet50 model pre-trained on ImageNet
model = ResNet50(weights='imagenet')


def preprocess_image(img_path):
    """ Preprocess the image for model input """
    img = cv2.imread(img_path)
    img = cv2.resize(img, (224, 224))  # Resize to match the input size of the pre-trained model
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert from BGR to RGB
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    img = img / 255.0  # Normalize image pixel values to [0, 1]
    return img


def predict_food(img):
    """ Use ResNet50 to predict food from the image """
    img = preprocess_input(img)
    preds = model.predict(img)
    decoded_preds = tf.keras.applications.resnet50.decode_predictions(preds, top=3)[0]
    return decoded_preds


def estimate_calories(food_name):
    """ Estimate calories based on food name using a mock database """
    food_name = food_name.lower()  # Convert to lowercase for consistency
    calories = calorie_database.get(food_name, "Unknown food item")
    return calories


def fetch_calories_from_api(food_name):
    """ Fetch calorie data from an external API (e.g., Open Food Facts) """
    url = f"https://world.openfoodfacts.org/api/v0/product/{food_name}.json"
    response = requests.get(url)
    data = response.json()

    if 'product' in data:
        product_info = data['product']
        calories = product_info.get('nutriments', {}).get('energy-kcal', "Calorie info not available")
        return calories
    else:
        return "Food item not found"


def estimate_calories_via_api(food_name):
    """ Estimate calories using the Open Food Facts API (or any other API) """
    return fetch_calories_from_api(food_name)


def main(img_path):
    """ Main function to predict food and estimate calories """
    # Step 1: Preprocess the image
    img = preprocess_image(img_path)

    # Step 2: Predict the food item using the trained model
    predictions = predict_food(img)

    # Step 3: Extract the top prediction (most likely food)
    predicted_food = predictions[0][1]  # The food label is in the second element

    # Step 4: Estimate the calories based on the food prediction
    calories = estimate_calories(predicted_food)

    # Display the result
    print(f"Predicted food item: {predicted_food}")
    print(f"Estimated calorie content (per 100g): {calories}")

    # Optionally, fetch calories via API
    # calories_from_api = estimate_calories_via_api(predicted_food)
    # print(f"Calories from API: {calories_from_api}")


# Example usage
img_path = "path_to_your_food_image.jpg"
main(img_path)
