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
#openai.api_key = "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"
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

# Function to generate description using OpenAI
def generate_stamp_description(stamp_info):
    prompt = f"Provide a detailed description for the following stamp:\n\n" \
             f"Name: {stamp_info['name of stamp']}\n" \
             f"Release Date: {stamp_info['day']}-{stamp_info['month']}-{stamp_info['year']}\n" \
             f"Price: ${stamp_info['price']}\n\n" \
             "Description:"
    
    # OpenAI API call using the newer model (gpt-3.5-turbo)
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",  # New model
        messages=[{"role": "user", "content": prompt}],
        max_tokens=150,
        temperature=0.7
    )
    
    return response['choices'][0]['message']['content'].strip()

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Check if the file is present
    if file is None:
        return JSONResponse(status_code=400, content={"error": "No file provided"})

    try:
        # Read the image file
        image_bytes = await file.read()
        image = Image.open(BytesIO(image_bytes))
        
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
        
        # Generate description using OpenAI
       # description = generate_stamp_description(stamp_info)
        
        # Prepare the response
        return {
            'name of stamp': stamp_info["name of stamp"],
            'release date': f"{stamp_info['day']}-{stamp_info['month']}-{stamp_info['year']}",
            'price': stamp_info["price"]
            #'description': description
        }

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


