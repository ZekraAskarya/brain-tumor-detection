import os
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.metrics import accuracy_score
from torchvision import transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


IMG_WIDTH, IMG_HEIGHT = 250, 250
IMAGE_FOLDER = 'Brain_Tumor_Data_Set/Brain_Tumor_Data_Set'  # Replace with your actual path

transform = transforms.Compose([
    transforms.ToPILImage(),               # Convert NumPy array to PIL Image
    transforms.Resize((IMG_WIDTH, IMG_HEIGHT)),  # Resize the image to 250x250
    transforms.ToTensor(),                 # Convert image to tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize the image
])

def load_data_from_directories(image_folder):
    X, y = [], []

    for label, folder in enumerate(['Healthy', 'Brain_Tumor']):
        folder_path = os.path.join(image_folder, folder)
        for img_name in os.listdir(folder_path):
            img_path = os.path.join(folder_path, img_name)

            image = cv2.imread(img_path)
            if image is None:
                print(f"Warning: Unable to load image {img_path}. Skipping.")
                continue

            # Convert to RGB (in case it's not)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            image = transform(image)

            X.append(image)
            y.append(label)

    X = torch.stack(X)  
    y = torch.tensor(y, dtype=torch.long)

    return X, y

class BrainDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class BrainTumorDetectionModel(nn.Module):
    def __init__(self):
        super(BrainTumorDetectionModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=7, stride=1, padding=2)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool = nn.MaxPool2d(4, 4)

        self.fc_input_size = 32 * 15 * 15  

        self.fc = nn.Linear(self.fc_input_size, 1)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        x = self.pool(x)  
        x = x.view(x.size(0), -1)  
        x = torch.sigmoid(self.fc(x))
        return x

X, y = load_data_from_directories(IMAGE_FOLDER)

data = BrainDataset(X, y)
data_len = len(data)
train_len = int(0.8 * data_len)
val_len = int(0.2 * data_len)

train_data, val_data = random_split(data, [train_len, val_len])

train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
val_loader = DataLoader(val_data, batch_size=32)

model = BrainTumorDetectionModel().to(device)

criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

patience = 5
best_val_loss = float('inf')
epochs_without_improvement = 0
best_model_path = "best_model.pth"

def train_model(model, train_loader, val_loader, epochs):
    global best_val_loss, epochs_without_improvement

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            y_batch = y_batch.float().unsqueeze(1)  # Match output shape

            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        val_loss = 0
        val_acc = 0
        model.eval()
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                y_batch = y_batch.float().unsqueeze(1)

                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                val_loss += loss.item()

                preds = (outputs > 0.5).float()
                val_acc += accuracy_score(y_batch.cpu(), preds.cpu())

        val_loss /= len(val_loader)
        val_acc /= len(val_loader)
        print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss / len(train_loader):.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_acc:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            torch.save(model.state_dict(), best_model_path)
            print("Best model saved!")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print("Early stopping triggered!")
                break

train_model(model, train_loader, val_loader, epochs=50)
