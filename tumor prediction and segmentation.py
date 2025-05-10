import cv2
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

dataset_folder = '/content/drive/MyDrive/dipcancer'
benign_folder = os.path.join(dataset_folder, 'benign')
malignant_folder = os.path.join(dataset_folder, 'malignant')
normal_folder = os.path.join(dataset_folder, 'normal')

combined_output_folder = '/content/combined_images'
os.makedirs(combined_output_folder, exist_ok=True)

def isolate_cancer(image_path):
    gray_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    _, binary_image = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY)
    isolated_region = cv2.bitwise_and(gray_image, gray_image, mask=binary_image)
    return gray_image, isolated_region

def load_data(folder, label):
    images = []
    labels = []
    for file_name in os.listdir(folder):
        file_path = os.path.join(folder, file_name)
        original_image, _ = isolate_cancer(file_path)
        resized_image = cv2.resize(original_image, (128, 128))
        images.append(resized_image.flatten())
        labels.append(label)
    return images, labels

malignant_images, malignant_labels = load_data(malignant_folder, label=1)
benign_images, benign_labels = load_data(benign_folder, label=0)
normal_images, normal_labels = load_data(normal_folder, label=2)

images = np.array(malignant_images + benign_images + normal_images)
labels = np.array(malignant_labels + benign_labels + normal_labels)
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.3, random_state=42)

from sklearn.neighbors import KNeighborsClassifier
model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred) * 100
print(f"Accuracy of the model: {accuracy:.2f}%")

for file_name in os.listdir(malignant_folder):
    file_path = os.path.join(malignant_folder, file_name)
    original_image, isolated_image = isolate_cancer(file_path)
    combined_image = np.hstack((original_image, isolated_image))
    output_path = os.path.join(combined_output_folder, file_name)
    cv2.imwrite(output_path, combined_image)

print(f"Combined images have been saved to: {combined_output_folder}")