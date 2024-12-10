import os
import requests
import pandas as pd
import validators
from PIL import Image, ImageEnhance, ImageOps
import random
import shutil
from sklearn.model_selection import train_test_split
import re
# Step 1: Load Dataset
def load_dataset(file_path):
    return pd.read_csv(file_path)

# Step 2: Validate URLs and remove invalid ones
import pandas as pd
import validators

def remove_invalid_urls(dataframe, column_name):
    """
    Traverses a specified column in a DataFrame, checks if each entry is a valid URL,
    and removes the rows with invalid URLs.

    Parameters:
    - dataframe (pd.DataFrame): The DataFrame to traverse.
    - column_name (str): The name of the column containing URLs.

    Returns:
    - pd.DataFrame: A DataFrame with rows containing invalid URLs removed.
    """
    invalid_indices = []  # List to store the indices of invalid URL rows

    # Loop through each URL in the column
    for index, url in dataframe[column_name].items():
        # Check if the value is NaN or not a string
        if pd.isna(url) or not isinstance(url, str):
            print(f"Invalid URL found at row {index}: {url} (Not a valid string or NaN)")
            invalid_indices.append(index)
        # Validate the URL using the validators library
        elif not validators.url(url):
            print(f"Invalid URL found at row {index}: {url}")
            invalid_indices.append(index)

    # Remove rows with invalid URLs using the collected indices
    dataframe_cleaned = dataframe.drop(index=invalid_indices)
    
    return dataframe_cleaned


def check_image_accessibility(dataframe, url_column):
    """
    Checks the accessibility of the images by verifying if the URL returns a successful status code (200).
    Removes rows with inaccessible image URLs.

    Parameters:
    - dataframe (pd.DataFrame): The DataFrame to check for accessible image URLs.
    - url_column (str): The name of the column containing image URLs.

    Returns:
    - pd.DataFrame: The DataFrame with rows containing inaccessible image URLs removed.
    """
    accessible_indices = []  # List to store indices of rows with accessible image URLs

    for index, row in dataframe.iterrows():
        url = row[url_column]
        try:
            response = requests.get(url)
            
            # Check if the URL is accessible (status code 200)
            if response.status_code == 200:
                accessible_indices.append(index)
                print(f"Accessible row: {index}, URL: {url}")
            else:
                print(f"Image at row {index} is not accessible: {url}. Status code: {response.status_code}")
        except requests.exceptions.RequestException as e:
            print(f"Error accessing image at row {index}: {url} - {e}")
    
    # Filter the dataframe to include only rows with accessible URLs
    dataframe_accessible = dataframe.loc[accessible_indices]
    return dataframe_accessible

# Step 4: Download Images
def download_image(url, image_folder, class_label):
    image_path = os.path.join(image_folder, f"{class_label}.jpg")
    
    # Check if the image already exists
    if not os.path.exists(image_path):
        try:
            os.makedirs(image_folder, exist_ok=True)
            response = requests.get(url)
            
            if response.status_code == 200:
                with open(image_path, 'wb') as file:
                    file.write(response.content)
                print(f"Downloaded image: {class_label}.jpg")
            else:
                print(f"Failed to download image from {url}")
        except Exception as e:
            print(f"Error downloading image from {url}: {e}")
    else:
        print(f"Image for class {class_label} already exists. Skipping download.")

# Step 5: Add Class Labels
def add_class_labels(df):
    df['class'] = range(1, len(df) + 1)  # Assign labels starting from 1 to len(df)
    return df

from PIL import Image, ImageOps, ImageEnhance, ImageChops


def add_shadow(image):
    """
    Add a shadow effect to the image.
    This simulates shadow by creating an alpha mask and overlaying it.
    """
    shadow = image.convert('RGBA')
    shadow = Image.new('RGBA', shadow.size, (0, 0, 0, 100))  # Dark transparent shadow
    image_with_shadow = Image.alpha_composite(shadow, image.convert('RGBA'))
    return image_with_shadow.convert('RGB')


def random_crop(image):
    """
    Perform a random crop on the image.
    This will randomly select a region from the image.
    """
    width, height = image.size
    crop_size = random.randint(int(width * 0.7), width)  # Randomly choose a crop size
    if width > crop_size and height > crop_size:
        x = random.randint(0, width - crop_size)
        y = random.randint(0, height - crop_size)
        return image.crop((x, y, x + crop_size, y + crop_size))
    return image


def augment_image(image_path, output_folder, original_row, augmentation_count=10):
    """
    Augments an image and saves different versions with specified transformations applied.
    Ensures that 10 augmented images are created using various transformations.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    augmented_rows = []  # List to store new rows with augmented images
    image = Image.open(image_path)
    if image.mode != 'RGB':
        image = image.convert('RGB')

    for i in range(augmentation_count):
        augmented_image = image.copy()
        augmented_image_path = os.path.join(output_folder, f"{original_row['class']}_augmented_{i+1}.jpg")

        # Check if augmented image already exists
        if os.path.exists(augmented_image_path):
            print(f"Augmented image {original_row['class']}_augmented_{i+1}.jpg already exists. Skipping.")
            continue

        # Randomly apply transformations
        transform_type = random.choice(['rotate', 'flip', 'resize', 'color', 'shadow', 'crop'])

        if transform_type == 'rotate':
            augmented_image = augmented_image.rotate(random.randint(-30, 30), expand=True)  # Random rotation
        elif transform_type == 'flip':
            augmented_image = ImageOps.mirror(augmented_image)  # Horizontal flip
        elif transform_type == 'resize':
            new_size = random.randint(150, 300)
            augmented_image = augmented_image.resize((new_size, new_size))  # Random resize
        elif transform_type == 'color':
            enhancer = ImageEnhance.Color(augmented_image)
            augmented_image = enhancer.enhance(random.uniform(0.5, 1.5))  # Random color enhancement
        elif transform_type == 'shadow':
            augmented_image = add_shadow(augmented_image)  # Add shadow
        elif transform_type == 'crop':
            augmented_image = random_crop(augmented_image)  # Random cropping

        # Save the augmented image and track its path
        augmented_image.save(augmented_image_path)
        print(f"Saved augmented image {augmented_image_path}")

        # Create a new row for the augmented image with metadata from the original image
        augmented_row = {
            'image': augmented_image_path,
            'name of stamp': original_row['name of stamp'],
            'price': original_row['price'],
            'day': original_row['day'],
            'month': original_row['month'],
            'year': original_row['year'],
            'class': original_row['class']
        }

        augmented_rows.append(augmented_row)  # Add the augmented row to the list

    return augmented_rows

    
    return augmented_rows
def replace_url_with_local_path(dataframe, url_column, class_column, destination_folder):
    """
    Traverses the specified column in the DataFrame, checks if the value is a URL,
    and replaces the URL with a local file path in the format 'dataset/images/classname.jpg'.

    Parameters:
    - dataframe (pd.DataFrame): The DataFrame to traverse.
    - url_column (str): The name of the column containing the URLs.
    - class_column (str): The name of the column containing the class (used for the image file name).
    - destination_folder (str): The local folder to save images.
    
    Returns:
    - pd.DataFrame: The DataFrame with URLs replaced by local file paths.
    """
    for index, row in dataframe.iterrows():
        url = row[url_column]
        class_name = row[class_column]
        
        if isinstance(url, str) and url.startswith("http"):  # Check if it's a URL
            # Construct the local path using the class column (e.g., 'class_name.jpg')
            local_path = os.path.join(destination_folder, f"{class_name}.jpg")

            # Replace the URL with the local path
            dataframe.at[index, url_column] = local_path
            print(f"Replaced URL at row {index} with local path: {local_path}")
    
    return dataframe
# Step 7: Main Function to Process Dataset and Update CSV
def process_dataset(input_csv_path, image_folder, augmented_folder):
    df = load_dataset(input_csv_path)
    
    # Step 2: Validate URLs and remove invalid ones
    df = remove_invalid_urls(df, "image")

    # Step 3: Remove inaccessible images before downloading
    df = check_image_accessibility(df,'image')
     # Step 5: Add class labels
    df = add_class_labels(df)
    
    # Step 4: Download images
    for index, row in df.iterrows():
        download_image(row["image"], image_folder, row["class"])

    # Step 6: Apply data augmentation and create new rows for augmented images
    augmented_rows = []
    for index, row in df.iterrows():
        image_path = os.path.join(image_folder, f"{row['class']}.jpg")
        augmented_rows.extend(augment_image(image_path, augmented_folder, row))
    
    # Add the augmented rows to the original dataframe
    augmented_df = pd.DataFrame(augmented_rows)
    # Replace URLs with local paths in the 'image_url' column
    destination_folder = 'dataset/images'
    df = replace_url_with_local_path(df, 'image', 'class', destination_folder)
    updated_df = pd.concat([df, augmented_df], ignore_index=True)
    
    # Step 7: Save the updated dataset
    updated_df.to_csv('dataset/updated_dataset.csv', index=False)
    print("Dataset processing complete. Updated dataset saved.")
    
    return updated_df
def collect_images_with_classes(folder_path):
    """
    Collect all .jpg images from a folder (and its subfolders), 
    extracting the class as an integer from the filename.

    Args:
    - folder_path (str): The path to the folder containing images.

    Returns:
    - List[dict]: A list of dictionaries containing image paths and their associated classes as integers.
    """
    images = []
    pattern = re.compile(r"^(\d+)(_augmented_\d+)?\.jpg$")  # Matches class and optional augmented suffix

    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.endswith(".jpg"):
                match = pattern.match(file)
                if match:
                    class_name = int(match.group(1))  # Convert class to integer
                    image_path = os.path.join(root, file).replace("\\", "/")  # Standardize paths
                    images.append({
                        "path": image_path,
                        "class": class_name
                    })
    return images

def prepare_split(images,class_metadata):
    """
    Process image data and extract metadata fields.
    Directly uses `img['path']` for the full image path to avoid redundant path manipulation.
    """
    data = []
    for img in images:
        class_name = img['class']  # Extract class directly
        
        # Debugging: Log processing information
        print(f"Processing image path: {img['path']} with class: {class_name}")
        
        if class_name in class_metadata:
            metadata = class_metadata[class_name]
            # Debugging: Log matched metadata
            print(f"Matched metadata for class '{class_name}': {metadata}")
            data.append({
                "image": img['path'],  # Directly use the full path
                "class": class_name,
                "name of stamp": metadata.get("name of stamp", ""),
                "price": metadata.get("price", ""),
                "day": metadata.get("day", ""),
                "month": metadata.get("month", ""),
                "year": metadata.get("year", ""),
            })
        else:
            # Debugging: Log if class isn't found
            print(f"Class '{class_name}' NOT found in class_metadata.")
    
    # Log the data prepared
    print("Split data prepared for DataFrame:")
    print(data)
    
    # Convert to DataFrame and return
    return pd.DataFrame(data)


def extract_and_split_datasets(og_folder,augmented_folder, image_folder, df):
    """
    Extract augmented images and split them into train, validation, and test sets,
    group class information and metadata into their splits, and save into train.csv, val.csv, and test.csv.
    """
    # Create directories for splits
    train_image_folder = os.path.join(image_folder, 'train')
    val_image_folder = os.path.join(image_folder, 'validation')
    test_image_folder = os.path.join(image_folder, 'test')

    os.makedirs(train_image_folder, exist_ok=True)
    os.makedirs(val_image_folder, exist_ok=True)
    os.makedirs(test_image_folder, exist_ok=True)
    source_folder = og_folder
    destination_folder =train_image_folder
# Copy all images from source to destination
    for file_name in os.listdir(source_folder):
        src_image_path = os.path.join(source_folder, file_name)
        dest_image_path = os.path.join(destination_folder, file_name)
        if os.path.isfile(src_image_path):  # Ensure it's a file
            print(f"Copying file: {src_image_path} -> {dest_image_path}")
            shutil.move(src_image_path, dest_image_path)
    print("All images have been copied to the train directory.")
    # Parse augmented images
    augmented_images = []
    for file_name in os.listdir(augmented_folder):
        if "augmented" in file_name and file_name.endswith(".jpg"):
            try:
                class_name, index = file_name.split('_augmented_')
                index = int(index.replace(".jpg", ""))
                augmented_images.append({
                    "file_path": os.path.join(augmented_folder, file_name),
                    "class": class_name,
                    "aug_index": index
                })
            except ValueError:
                continue

    # Split augmented images
    train_images = [img for img in augmented_images if img['aug_index'] in [1, 2]]
    val_images = [img for img in augmented_images if img['aug_index'] in [3, 4, 5]]
    test_images = [img for img in augmented_images if img['aug_index'] in [6, 7, 8, 9, 10]]

    # Move augmented images
    for img in train_images:
        shutil.move(img['file_path'], os.path.join(train_image_folder, os.path.basename(img['file_path'])))

    for img in val_images:
        shutil.move(img['file_path'], os.path.join(val_image_folder, os.path.basename(img['file_path'])))

    for img in test_images:
        shutil.move(img['file_path'], os.path.join(test_image_folder, os.path.basename(img['file_path'])))

    # Debugging to check df data
   

    # 
    # Extract unique class metadata
    unique_classes_df = df.drop_duplicates(subset='class').copy()
    if 'image' in unique_classes_df.columns:  # Ensure 'image' column exists before dropping
        unique_classes_df = unique_classes_df.drop(columns=['image'])

    # Create class metadata dictionary
    class_metadata = unique_classes_df.set_index('class').to_dict(orient='index')
    print("Class Metadata:")
    print(class_metadata)

    
    train_images=collect_images_with_classes('dataset/train')
    train_df = prepare_split(train_images,class_metadata)
    val_images=collect_images_with_classes('dataset/validation')
    val_df = prepare_split(val_images,class_metadata)
    test_images=collect_images_with_classes('dataset/test')
    test_df = prepare_split(test_images,class_metadata)

    # Debugging logs for splits
    print(f"Train DF: {train_df}")
    print(f"Validation DF: {val_df}")
    print(f"Test DF: {test_df}")

    # Save the datasets into CSVs
    train_df.to_csv('dataset/train.csv', index=False)
    val_df.to_csv('dataset/val.csv', index=False)
    test_df.to_csv('dataset/test.csv', index=False)

    print("Datasets created and saved successfully.")
    return train_df, val_df, test_df

 





# Call main function to process dataset
if __name__ == "__main__":
    input_csv_path = 'dataset/dataset.csv'  # Your input CSV file with URLs and labels
    image_folder = 'dataset'         # Folder to store the downloaded images
    augmented_folder = 'dataset/augmented' 
    og_folder = 'dataset/image' 
    
    # Process dataset (download images, validate URLs, apply augmentation)
    updated_df = process_dataset(input_csv_path, og_folder, augmented_folder)
  #  updated_df=pd.read_csv("dataset/updated_dataset.csv")
    # Split the updated dataset into training and testing sets
    train_df, val_df, test_df = extract_and_split_datasets(og_folder,augmented_folder, image_folder, updated_df)
