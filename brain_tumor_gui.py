import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image, ImageTk
import customtkinter as ctk
from tkinter import filedialog, messagebox

# Set appearance mode and theme
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("dark-blue")

# Transformation to resize and normalize images (250x250)
transform = transforms.Compose([
    transforms.Resize((250, 250)),  # Resize to 250x250 as per your model
    transforms.ToTensor(),
    # Standard ImageNet normalization
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Define the model structure (custom CNN for Brain Tumor Detection)


class BrainTumorDetectionModel(nn.Module):
    def __init__(self):
        # Corrected __init__ method
        super(BrainTumorDetectionModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=7, stride=1, padding=2)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool = nn.MaxPool2d(4, 4)
        self.fc_input_size = 32 * 15 * 15  # Adjusted after pooling layers
        self.fc = nn.Linear(self.fc_input_size, 1)

    def forward(self, x):
        x = torch.relu(self.bn1(self.conv1(x)))  # Apply first convolution
        x = self.pool(x)  # Max-pooling
        x = self.pool(x)  # Apply second pooling
        x = x.view(x.size(0), -1)  # Flatten
        x = torch.sigmoid(self.fc(x))  # Sigmoid for binary classification
        return x

# Load the trained model


def load_model(weights_path, device):
    model = BrainTumorDetectionModel()
    model.load_state_dict(torch.load(
        weights_path, map_location=device, weights_only=True))  # Use weights_only=True
    model.eval()  # Set the model to evaluation mode
    return model

# Function to classify the image


def classify_image(model, image_path, device):
    try:
        image = Image.open(image_path).convert('RGB')
        image = transform(image).unsqueeze(0)  # Add batch dimension
        image = image.to(device)

        with torch.no_grad():
            output = model(image)  # Get model prediction
            prediction = "Tumor Detected" if output.item(
            ) > 0.5 else "Healthy"  # Use 0.5 threshold

        return prediction
    except Exception as e:
        raise Exception(f"Error during classification: {e}")

# Function to open file dialog, upload, and classify image


def open_file_and_classify():
    file_path = filedialog.askopenfilename(
        title="Select an Image",
        filetypes=[("Image Files", "*.png *.jpg *.jpeg *.bmp")]
    )

    if file_path:
        try:
            display_image(file_path)  # Display the selected image
            predicted_class = classify_image(
                model, file_path, device)  # Classify the image
            prediction_label.configure(
                text=f"Prediction: {predicted_class}")  # Show the result
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {e}")
    else:
        messagebox.showwarning("File Error", "No file selected!")

# Function to display image on the GUI


def display_image(image_path):
    try:
        img = Image.open(image_path)
        img = img.resize((300, 300))  # Resize image for display

        # Convert PIL image to CTkImage
        img_tk = ctk.CTkImage(light_image=img, size=(300, 300))

        # Update the image label
        image_label.configure(image=img_tk)
        image_label.image = img_tk  # Keep a reference to avoid garbage collection
    except Exception as e:
        raise Exception(f"Error loading image: {e}")


# Main GUI setup
app = ctk.CTk()
app.geometry("800x600")  # Set the size of the window
app.title("Brain Tumor Detection")

# Set the background color to black
app.configure(bg="black")

# Load the model
weights_path = 'best_model.pth'  # Path to your trained model weights
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = load_model(weights_path, device)
model.to(device)

# Title label
title_label = ctk.CTkLabel(app, text="Brain Tumor Detection", font=(
    "Helvetica", 24, "bold"), text_color="white")
title_label.pack(pady=20)

# Label to display image (initially blank)
image_label = ctk.CTkLabel(app, text="", width=300,
                           height=300, text_color="white")
image_label.pack(pady=20)

# Button to upload and classify image (red background)
upload_button = ctk.CTkButton(
    app,
    text="Upload and Classify Image",
    font=("Arial", 16),
    width=200,
    height=50,
    command=open_file_and_classify,
    text_color="white",
    fg_color="red"
)
upload_button.pack(pady=20)

# Label to show the prediction result
prediction_label = ctk.CTkLabel(
    app, text="Prediction: ", font=("Arial", 16), text_color="white")
prediction_label.pack(pady=20)

# Run the GUI
app.mainloop()
