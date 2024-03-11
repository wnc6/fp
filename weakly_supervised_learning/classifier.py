import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models import densenet121

class Classifier(nn.Module):
    """
    A classifier model that uses DenseNet121 for feature extraction and a custom classifier for prediction.
    
    Attributes:
    - densenet (nn.Module): The DenseNet121 model used for feature extraction.
    - classifier (nn.Sequential): The custom classifier used for prediction.
    """
    def __init__(self, num_classes):
        """
        Initializes the Classifier with the specified number of classes.
        
        Parameters:
        - num_classes (int): The number of output classes for the classifier.
        """
        super(Classifier, self).__init__()
        # Load the pre-trained DenseNet121 model
        self.densenet = densenet121(pretrained=True)
        
        # Freeze the parameters of the DenseNet121 model
        for param in self.densenet.parameters():
            param.requires_grad = False
        
        # Replace the classifier part of the DenseNet121 model with our own
        num_features = self.densenet.classifier.in_features
        self.densenet.classifier = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes),
            nn.Softmax(dim=1)
        )
    
    def forward(self, x):
        """
        Forward pass of the input through the model.
        
        Parameters:
        - x (torch.Tensor): The input tensor.
        
        Returns:
        - torch.Tensor: The output tensor after passing through the model.
        """
        x = self.densenet(x)
        return x

# Example usage
num_classes = 55 # Number of weak labels
model = Classifier(num_classes)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Example of training loop
for epoch in range(10): # Number of epochs
    running_loss = 0.0
    for i, data in enumerate(dataloader, 0):
        inputs, labels = data
        optimizer.zero_grad()
        
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    print(f'Epoch {epoch + 1}, Loss: {running_loss / len(dataloader)}')
