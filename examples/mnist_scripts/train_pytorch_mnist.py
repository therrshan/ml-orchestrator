#!/usr/bin/env python3
"""
Train PyTorch CNN model on MNIST data.
"""

import os
import sys
import json
import argparse
import numpy as np
from pathlib import Path

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, TensorDataset
    print("✅ PyTorch imported successfully")
except ImportError:
    print("❌ PyTorch not found. Install with: pip install torch")
    sys.exit(1)


class SimpleCNN(nn.Module):
    """Simple CNN for MNIST classification."""
    
    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        
        # Pooling
        self.pool = nn.MaxPool2d(2, 2)
        
        # Fully connected layers
        self.fc1 = nn.Linear(64 * 3 * 3, 128)
        self.fc2 = nn.Linear(128, num_classes)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x):
        # Conv block 1
        x = self.pool(F.relu(self.conv1(x)))  # 28x28 -> 14x14
        
        # Conv block 2
        x = self.pool(F.relu(self.conv2(x)))  # 14x14 -> 7x7
        
        # Conv block 3
        x = F.relu(self.conv3(x))  # 7x7 -> 7x7
        x = self.pool(x)  # 7x7 -> 3x3
        
        # Flatten and fully connected
        x = x.view(-1, 64 * 3 * 3)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x


def load_data(data_dir):
    """Load processed MNIST data."""
    data_dir = Path(data_dir)
    
    # Load training data
    train_data = np.load(data_dir / 'train.npz')
    X_train, y_train = train_data['X'], train_data['y']
    
    # Load validation data
    val_data = np.load(data_dir / 'validation.npz')
    X_val, y_val = val_data['X'], val_data['y']
    
    print(f"📂 Loaded training data: {X_train.shape}")
    print(f"📂 Loaded validation data: {X_val.shape}")
    
    return X_train, y_train, X_val, y_val


def create_data_loaders(X_train, y_train, X_val, y_val, batch_size=64):
    """Create PyTorch data loaders."""
    
    # Convert to PyTorch tensors
    X_train_tensor = torch.FloatTensor(X_train).unsqueeze(1)  # Add channel dimension
    y_train_tensor = torch.LongTensor(y_train)
    X_val_tensor = torch.FloatTensor(X_val).unsqueeze(1)
    y_val_tensor = torch.LongTensor(y_val)
    
    # Create datasets
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    print(f"🔄 Created data loaders:")
    print(f"   Train batches: {len(train_loader)}")
    print(f"   Validation batches: {len(val_loader)}")
    
    return train_loader, val_loader


def train_model(model, train_loader, val_loader, epochs=5, learning_rate=0.001):
    """Train the CNN model."""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️  Training on device: {device}")
    
    model.to(device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    training_history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    print(f"🏋️  Starting training for {epochs} epochs...")
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            train_correct += pred.eq(target.view_as(pred)).sum().item()
            train_total += target.size(0)
            
            # Print progress every 100 batches
            if batch_idx % 100 == 0:
                print(f'Epoch {epoch+1}/{epochs}, Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}')
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                val_loss += criterion(output, target).item()
                pred = output.argmax(dim=1, keepdim=True)
                val_correct += pred.eq(target.view_as(pred)).sum().item()
                val_total += target.size(0)
        
        # Calculate metrics
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        train_acc = 100. * train_correct / train_total
        val_acc = 100. * val_correct / val_total
        
        # Store history
        training_history['train_loss'].append(train_loss)
        training_history['train_acc'].append(train_acc)
        training_history['val_loss'].append(val_loss)
        training_history['val_acc'].append(val_acc)
        
        print(f'Epoch {epoch+1}/{epochs}:')
        print(f'  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        print(f'  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
        print()
    
    return training_history


def save_model(model, model_path, training_history, model_info):
    """Save the trained model."""
    
    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save model state dict
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_class': 'SimpleCNN',
        'training_history': training_history,
        'model_info': model_info
    }, model_path)
    
    # Save model metadata as JSON
    metadata_path = model_path.with_suffix('.json')
    with open(metadata_path, 'w') as f:
        json.dump({
            'model_class': 'SimpleCNN',
            'training_history': training_history,
            'model_info': model_info
        }, f, indent=2)
    
    print(f"💾 Model saved:")
    print(f"   Model: {model_path}")
    print(f"   Metadata: {metadata_path}")


def main():
    parser = argparse.ArgumentParser(description='Train PyTorch CNN on MNIST')
    parser.add_argument('--data', required=True, help='Directory with processed data')
    parser.add_argument('--model', required=True, help='Path to save trained model')
    parser.add_argument('--epochs', type=int, default=5, help='Number of training epochs')
    
    args = parser.parse_args()
    
    # Get environment variables
    batch_size = int(os.getenv('BATCH_SIZE', 64))
    learning_rate = float(os.getenv('LEARNING_RATE', 0.001))
    
    print("🤖 Starting PyTorch MNIST training...")
    print(f"📊 Parameters:")
    print(f"   Epochs: {args.epochs}")
    print(f"   Batch size: {batch_size}")
    print(f"   Learning rate: {learning_rate}")
    
    try:
        # Load data
        X_train, y_train, X_val, y_val = load_data(args.data)
        
        # Create data loaders
        train_loader, val_loader = create_data_loaders(
            X_train, y_train, X_val, y_val, batch_size
        )
        
        # Create model
        model = SimpleCNN(num_classes=10)
        print(f"🏗️  Created model with {sum(p.numel() for p in model.parameters())} parameters")
        
        # Train model
        training_history = train_model(
            model, train_loader, val_loader, args.epochs, learning_rate
        )
        
        # Model info
        model_info = {
            'architecture': 'SimpleCNN',
            'num_parameters': sum(p.numel() for p in model.parameters()),
            'epochs_trained': args.epochs,
            'batch_size': batch_size,
            'learning_rate': learning_rate,
            'final_train_acc': training_history['train_acc'][-1],
            'final_val_acc': training_history['val_acc'][-1],
            'device': str(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        }
        
        # Save model
        save_model(model, args.model, training_history, model_info)
        
        print("🎉 Training completed successfully!")
        print(f"📈 Final validation accuracy: {model_info['final_val_acc']:.2f}%")
        
        return 0
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())