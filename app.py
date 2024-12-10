from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import torch
from torchvision import transforms
from PIL import Image
import torchvision
import pandas as pd
from io import BytesIO
import openai

app = FastAPI()

# OpenAI API key setup
#openai.api_key = "YOUR_API_KEY"

# Load pre-trained model
model = torchvision.models.resnet18(pretrained=False)
model.fc = torch.nn.Linear(model.fc.in_features, 1020)  # Assume there are 10 classes
model.load_state_dict(torch.load('models/stamp_model.pth'))
model.eval()

# Load the CSV file containing class information
stamp_data = pd.read_csv('dataset/train.csv')

# Convert the DataFrame to a dictionary for easy lookup
stamp_data =stamp_data.drop_duplicates(subset='class').copy()
if 'image' in stamp_data.columns:  # Ensure 'image' column exists before dropping
    stamp_data = stamp_data.drop(columns=['image'])
stamp_info_dict = stamp_data.set_index('class').to_dict(orient='index')

# Transformation for incoming image
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Check if the file is present
    if file is None:
        return JSONResponse(status_code=400, content={"error": "No file provided"})

    try:
        # Read the image file
        image_bytes = await file.read()

        # Open the image using PIL
        image = Image.open(BytesIO(image_bytes))

        # Ensure the image is in a valid format (e.g., RGB)
        image = image.convert('RGB')
        
        # Apply transformation
        image = transform(image).unsqueeze(0)  # Add batch dimension
        
        # Perform inference
        with torch.no_grad():
            output = model(image)
            _, predicted_class = torch.max(output, 1)
        
        class_label = predicted_class.item()

        # Lookup class information from the CSV data
        stamp_info = stamp_info_dict.get(class_label, {
            "name of stamp": "Unknown Stamp",
            "price": 0.0,
            "day": 1,
            "month": 1,
            "year": 2000
        })
        
        # Prepare the response
        return {
            'name of stamp': stamp_info["name of stamp"],
            'release date': f"{stamp_info['day']}-{stamp_info['month']}-{stamp_info['year']}",
            'price': stamp_info["price"]
        }

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
