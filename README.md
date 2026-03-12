# Image Forgery Detection Pipeline

A deep learning pipeline designed to identify manipulated images, classify the type of forgery, and localize the tampered regions using EfficientNet and a DualStream U-Net Model.
## Project Structure
### Notebooks

- **ImageManipulationDetection_07_EfficientNetRetraining.ipynb**: Handles training for binary (Authentic/Forged) and type (Spliced/Copy-Move) classification.

- **ImageManipulationDetection_08_Localization.ipynb**: Handles training for the localization model that is responsible for identifying forged areas in images.

- **ImageManipulationPredictionPipeline.ipynb**: Contains the final prediction pipeline. It contains all the preprocessing functions and Classes needed to load the localization model.

### Directories

- **Data**: Contains the CASIA2 Dataset used for training and validation. IMPORTANT: NOT UPLOADED TO GITHUB.

- **ForgeryDetectionModelEfficientNet**: Storage for all serialized model weights and architecture configurations.

- **ImageForgeryDetectionPipeline_Predictions**: Stores pipeline outputs, including model performance metrics.

- **TestImages**: A set of 30 images (10 Authentic, 10 Spliced, 10 Copy-Move) for verification.

## How to Run the Pipeline
1. Default Testing

    To verify the pipeline with pre-existing data, open and run ImageManipulationPredictionPipeline.ipynb. By default, it is configured to process the samples within the /TestImages folder and automatically load the weights from /ForgeryDetectionModelEfficientNet.
2. Testing Custom Images

    To test your own images, place your image and its corresponding ground-truth mask (if available) into the /TestImages folder. Use the following function call within the notebook:
Python

## Configuration for custom prediction
IMAGE_FILENAME = 'your_image.jpg'

MASK_FILENAME = 'your_mask.png'

````python
predict_image_forgery(
    image_path=os.path.join(DATA_TEST_PATH, IMAGE_FILENAME), 
    bin_model=saved_bin_pretrained_model, 
    type_model=saved_type_pretrained_model, 
    loc_model=loc_model, 
    original_class=1,      # 0: Authentic, 1: Forged
    original_type=0,       # 0: Spliced, 1: Copy-Move
    device=device, 
    mask_path=os.path.join(DATA_TEST_PATH, MASK_FILENAME)
)
````

## Model Architecture Overview

The pipeline uses a multi-stage approach to ensure high precision:

- **Binary Classification**: Determines if the image has been tampered with.

- **Type Classification**: Categorizes the forgery (Splicing vs. Copy-Move).

- **Localization**: Generates a prediction mask highlighting the specific manipulated pixels.