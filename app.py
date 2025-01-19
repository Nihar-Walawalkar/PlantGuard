import os
from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
import torch
from torchvision import transforms
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F

# Define the Flask app
app = Flask(__name__)

# Set the upload folder and allowed extensions
UPLOAD_FOLDER = 'static/uploads/'
ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Define the model (same ResNet18 model as used during training)
class PlantDiseaseModel(nn.Module):
    def __init__(self):
        super(PlantDiseaseModel, self).__init__()
        self.model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=False)
        self.model.fc = nn.Linear(self.model.fc.in_features, 4)

    def forward(self, x):
        return self.model(x)

# Load the model and set it to evaluation mode
model = PlantDiseaseModel()
model.load_state_dict(torch.load('best_plant_disease_model.pth', map_location=torch.device('cpu')))
model.eval()

# Define the image transformations
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Allowed file extension checker
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Home route
@app.route('/')
def index():
    return render_template('index.html')

# Route to handle image upload and prediction
@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(request.url)
    
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        # Perform prediction
        image = Image.open(file_path).convert('RGB')
        image = transform(image).unsqueeze(0)  # Add batch dimension
        with torch.no_grad():
            output = model(image)
            predictions = torch.sigmoid(output).numpy()
        
        # Mapping predictions to class labels
        classes = ['Healthy', 'Multiple Diseases', 'Rust', 'Scab']
        results = {classes[i]: predictions[0][i] for i in range(len(classes))}
        
        return render_template('result.html', filename=filename, results=results)

    return redirect(request.url)

# Serve the uploaded file
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return redirect(url_for('static', filename='uploads/' + filename), code=301)

# Run the app
if __name__ == "__main__":
    app.run(debug=True)
