import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, random_split
from torchvision import models

# === CONFIG ===
# Ensure these paths are correct for your environment
DATA_DIR = '/content/Final_Project/data/handwriting_dataset_training'
TEST_DATA_DIR = '/content/Final_Project/data/handwriting_dataset_testing'
MODEL_PATH = '/content/Final_Project/models/handwriting_model.pt'

NUM_EPOCHS = 40
BATCH_SIZE = 32
LEARNING_RATE = 1e-3

# === PREPROCESSING (Using Robust Normalization) ===
transform_train = transforms.Compose([
    transforms.Grayscale(1),
    transforms.RandomRotation(10),
    transforms.RandomAffine(0, shear=10, scale=(0.8, 1.2), translate=(0.1, 0.1)),
    transforms.ColorJitter(brightness=0.3, contrast=0.3),
    transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    # Reverting to standard normalization - safer than fixed threshold binarization
    transforms.Normalize([0.5], [0.5])
])

transform_eval = transforms.Compose([
    transforms.Grayscale(1),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

# === ADVANCED DATA SPLIT (40/10 + 40/10) ===
if not os.path.exists(DATA_DIR):
    print(f"❌ ERROR: Data directory not found at {DATA_DIR}")
    exit()

full_dataset = ImageFolder(DATA_DIR, transform=transform_train)
class_names = full_dataset.classes
num_classes = len(class_names)

total_len = len(full_dataset)
len_train_a = int(0.4 * total_len)
len_val_a = int(0.1 * total_len)
len_train_b = int(0.4 * total_len)
len_val_b = total_len - (len_train_a + len_val_a + len_train_b)

print(f"✂️ Splitting data: Set A ({len_train_a}/{len_val_a}) | Set B ({len_train_b}/{len_val_b})")

train_a, val_a, train_b, val_b = random_split(full_dataset, [len_train_a, len_val_a, len_train_b, len_val_b])

val_a.dataset.transform = transform_eval
val_b.dataset.transform = transform_eval

train_loader_a = DataLoader(train_a, batch_size=BATCH_SIZE, shuffle=True)
val_loader_a = DataLoader(val_a, batch_size=BATCH_SIZE, shuffle=False)
train_loader_b = DataLoader(train_b, batch_size=BATCH_SIZE, shuffle=True)
val_loader_b = DataLoader(val_b, batch_size=BATCH_SIZE, shuffle=False)

# === MODEL SETUP (MobileNetV3-Small) ===
model = models.mobilenet_v3_small(weights='IMAGENET1K_V1')

# 1. Modify first layer for 1-channel grayscale input
original_first_layer = model.features[0][0]
model.features[0][0] = nn.Conv2d(
    in_channels=1,
    out_channels=original_first_layer.out_channels,
    kernel_size=original_first_layer.kernel_size,
    stride=original_first_layer.stride,
    padding=original_first_layer.padding,
    bias=False
)

# 2. Modify final classifier layer
num_features = model.classifier[3].in_features
model.classifier[3] = nn.Sequential(
    nn.Dropout(0.5),
    nn.Linear(num_features, num_classes)
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)

# === TRAINING LOOP ===
best_val_acc = 0.0

for epoch in range(NUM_EPOCHS):
    if epoch < 25:
        current_train_loader = train_loader_a
        current_val_loader = val_loader_a
        phase_name = "A"
    else:
        current_train_loader = train_loader_b
        current_val_loader = val_loader_b
        phase_name = "B"

    model.train()
    train_loss = 0
    for images, labels in current_train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in current_val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    acc = 100 * correct / total
    scheduler.step(acc)
    
    print(f"📘 [{phase_name}] Epoch {epoch+1:02}/{NUM_EPOCHS} | Loss: {train_loss/len(current_train_loader):.4f} | Val Acc: {acc:.2f}%")

    if acc > best_val_acc:
        best_val_acc = acc
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        torch.save({'model_state_dict': model.state_dict(), 'class_names': class_names}, MODEL_PATH)

print(f"\n✅ Best model saved with {best_val_acc:.2f}% accuracy.")

# === TESTING ===
if os.path.exists(TEST_DATA_DIR):
    print("\n🧪 Starting testing on NEW ready_for_testing data...")
    test_dataset = ImageFolder(TEST_DATA_DIR, transform=transform_eval)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    checkpoint = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f"🎉 Final Test Accuracy: {100 * correct / total:.2f}%")
else:
    print(f"⚠️ Test folder not found at {TEST_DATA_DIR}")