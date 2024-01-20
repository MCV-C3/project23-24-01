import numpy as np
from keras.preprocessing import image
from keras.applications.xception import Xception, preprocess_input
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from PIL import Image
import matplotlib.pyplot as plt

img_name = 'inside_city/art626.jpg'
weights_file = '/ghome/group01/group01/project23-24-01/Task3/weights/CPU/100_xception.weights.h5'

# Load the Xception base model
base_model = Xception(weights='imagenet', include_top=False)
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(8, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)
model.load_weights(weights_file)

# Print the summary to see the number of layers
print("Number of layers in saved model: ", len(model.layers))

# Select the first, a middle, and the last layer
first_layer = model.layers[1]  # Assuming the input layer is considered as the first layer
middle_layer = model.layers[len(model.layers)//2]  # Selecting a middle layer
last_layer = model.layers[-1]  # Selecting the last layer

selected_layers = [first_layer, middle_layer, last_layer]

img_path = f'/ghome/mcv/datasets/C3/MIT_split/test/{img_name}'

for selected_layer in selected_layers:
    # Create the feature extraction model
    feature_extraction_model = Model(inputs=model.input, outputs=selected_layer.output)

    # Load and preprocess the example image
    img = Image.open(img_path)
    img = img.resize((299, 299))  # Assuming Xception's default input size is used
    img_array = np.expand_dims(np.array(img), axis=0)
    img_array = preprocess_input(img_array)

    # Get the feature map
    feature_map = feature_extraction_model.predict(img_array)

    # Reshape the feature map to a 2D array for visualization
    feature_map_reshaped = feature_map.reshape((1, -1))

    # Plot and save the feature map as an image
    save_filename = f'results/feature_map/{img_name.replace("/", "_").replace(".", "_")}_feature_map_{selected_layer.name}.jpg'
    plt.imshow(feature_map_reshaped, cmap='viridis', aspect='auto')  # Adjust cmap as needed
    plt.colorbar()
    plt.title(f'Feature Map Visualization - Layer: {selected_layer.name} - Image: {img_name}')
    plt.xlabel('Channel')
    plt.ylabel('Example')
    plt.savefig(save_filename)
