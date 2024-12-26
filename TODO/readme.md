# Week 1: Data Preparation

## Day 1: Understand the Basics
**whattodo**: Familiarize yourself with the project requirements and tools.

1. **Install Python**:  
   - Follow this tutorial: [How to Install Python](https://www.youtube.com/watch?v=nhv82tvFfkM)

2. **Install TensorFlow**:  
   - Watch this guide: [Install TensorFlow](https://youtu.be/AgzBrvWWVNQ?t=120)

3. **Install PyCharm**:  
   - Learn how to set it up: [Install PyCharm](https://www.youtube.com/watch?v=ZVjQFjOI49c)

4. **Install OpenCV**:  
   - Use this video tutorial: [Install OpenCV](https://www.youtube.com/shorts/yHDSbZJlMx0)

5. **Learn the Basics**:
   - Study Convolutional Neural Networks and complete the videos list that I had sent my previous emails.: 
   - Learn about image preprocessing: resizing, scaling, flipping, and rotating.

---

## Day 2: Organize Data
**whattodo**: Create the folder structure and organize images.

1. **Folder Setup**:
   - Create a main project folder, e.g., `MLResearch2024/`.
   - Inside `MLResearch2024/`, create a subfolder named `imgDataSet/`.
   - Within `imgDataSet/`, create two subfolders:
     - `Healthy/`
     - `Placenta_Accreta/`

2. **Organize Images**:
   - Place all images of healthy X-rays in the `Healthy/` folder.
   - Place all images of X-rays showing placenta accreta in the `Placenta_Accreta/` folder.

3. **Resulting Folder Structure**:
   ```
    MLResearch2024/
    ├── imgDataSet/
    │ ├── Healthy/
    │ ├── Placenta_Accreta/
   ```

## Day 3: Data Augmentation Setup
**whattodo**: Understand and prepare for data augmentation techniques to enhance dataset variability.

---

### 1. Understand Data Augmentation
- **What is Data Augmentation?**
  - Data augmentation improves model generalization by creating new training samples through transformations.
  - Common techniques include:
    - **Resizing**: Adjust the size of images to match the input size expected by the model.
    - **Scaling**: Normalize pixel values to a range (e.g., [0, 1]).
    - **Flipping**: Flip images horizontally or vertically to simulate different orientations.
    - **Rotating**: Rotate images by random angles to mimic diverse perspectives.

- **Watch This Tutorial**:  
  [Data Augmentation in Deep Learning](https://www.youtube.com/shorts/S-7LpWzUaOg)
  [Example - Data Augmentation](https://colab.research.google.com/github/tensorflow/docs/blob/master/site/en/tutorials/images/data_augmentation.ipynb#scrollTo=0BkRvvsXb6SI)

- **Example Code**
```
import tensorflow as tf                 
import matplotlib.pyplot as plt          # this helps us create graphs and show images


dataset = tf.keras.preprocessing.image_dataset_from_directory(
    "imgDataSet/Healthy",                      # get images from folder named "imgDataSet"
    image_size=(256, 256),              # make all images 256x256 pixels in size we will change this later depending on the accuracy of the models
    batch_size=32,                      # process 32 images at a time this is done to save computer memory
)

data_augmentation = tf.keras.Sequential([                       # Now we will create a pipeline to modify our images
    tf.keras.layers.RandomFlip("horizontal_and_vertical"),      # we will randomly flip images up-down and left-right
    tf.keras.layers.RandomRotation(0.2),                        # here we are randomly rotate images by up to 20% we can change this as much as possible
   # we can use one or combination of many data augmentation techniques to modify our images and increase our dataset
   # tf.keras.layers.Resizing(256, 256),                          # we are resizing all images to be the same size (256x256 pixels)
   # tf.keras.layers.Rescaling(1./255)                             # this divides every pixel value by 255 we do this to make the computation easier 0 means completely black 1 means white
   # other augmentations:
   # tf.keras.layers.RandomZoom(0.2),      # zoom in/out by up to 20%
   # tf.keras.layers.RandomBrightness(0.2), # adjust brightness
   # tf.keras.layers.RandomContrast(0.2),   # adjust contrast
])

# If you want to view the augmentated data use the code below
# everything below this is optional code for debugging and understanding
# This is not modifying anything just setting up the container
for image_batch, _ in dataset.take(1):                          # Now we will take one batch ( of 32 images) from our dataset
    augmented_images = data_augmentation(image_batch)           # and apply our modifications to these images from our previous function created above
    plt.figure(figsize=(10, 10))                               # this will create a 10x10 inch window/container to display our images
    for i in range(9):                                         # this will loop through our dataset's first 9 images 
        ax = plt.subplot(3, 3, i + 1)                          # This will create a 3x3 grid of images to display our images 
        plt.imshow(augmented_images[i].numpy().astype("uint8"))# Displ image
        plt.axis("off")                                        # Hide the x and y axis lines
    plt.show()                                                 # Show all the images on screen
```

---



   
