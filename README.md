# Indian Postage Stamp Dataset and Classification

This project involves creating a comprehensive dataset for Indian postage stamps by scraping data from the official [India Post Stamps website](https://postagestamps.gov.in/). The pipeline includes data preprocessing, augmentation, model training, and API development for stamp-related features.

---

## Features

### 1. Dataset Creation
- Scraped data from [India Post Stamps website](https://postagestamps.gov.in/), extracting:
  - Stamp images
  - Release dates
  - Stamp names
  - Prices
- Preprocessed the dataset to separate columns like date, month, and year.

### 2. Data Validation
- **Script:** `dataset.py`
  - Removed invalid image links.
  - Checked link accessibility and removed rows with inaccessible links.

### 3. Data Augmentation
- Allocated class labels in the dataset.
- Downloaded images from valid links and stored them locally.
- Generated 10 augmented images per original image and saved them in the `augmented` folder.
- Updated the dataset to replace remote links with local file paths.

### 4. Dataset Splitting
- Split the data into:
  - **Training set**: Original image + 2 augmented images
  - **Validation set**: 3 augmented images
  - **Test set**: 5 augmented images
- Saved the splits in `train.csv`, `val.csv`, and `test.csv`.

### 5. Model Training
- Trained a ResNet-18 model on the dataset.
- Saved the best weights in `stamp_model.pth`.

### 6. Testing
- **Script:** `test.py`
  - Evaluated model accuracy on test data.

### 7. API Development
- **Script:** `app.py`
  - Created an API for querying stamp-related information and features.

---

## Directory Structure

