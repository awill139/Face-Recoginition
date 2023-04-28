import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import torch.optim as optim

from torch_data import get_data

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Using device {device}')

class MultiLinear(nn.Module):
    def __init__(self, input_size, num_layers, num_classes):
        super(MultiLinear, self).__init__()
        self.layers = nn.ModuleList([
            nn.Linear(input_size, num_classes) for i in range(num_layers)
        ])
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        out = []
        for layer in self.layers:
            out.append(self.softmax(layer(x)))
        joint = out[0]
        for i in range(1, len(out)):
            joint = joint * out[i]
        joint /= joint.sum()
        return joint

class Face_Model(nn.Module):
    def __init__(self, num_classes=40, bayesian=False, lr=0.001):
        super().__init__()
        self.bayesian = bayesian
        self.model = models.resnet18(pretrained=True)
        self.model.fc = nn.Linear(512, 512)

        if bayesian:
            self.out_layer = MultiLinear(512, 5, num_classes)
        else:
            self.out_layer = nn.Linear(512, num_classes)

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.parameters(), lr=lr)

        self.to(device)

    def get_embedding(self, x):
        return self.model(x)
    
    def forward(self, x):
        x = self.get_embedding(x)
        return self.out_layer(x)
    
    def save_model(self):
        if self.bayesian:
            torch.save(self.state_dict, 'models/bayesian_model.pth')
        else:
            torch.save(self.state_dict, 'models/model.pth')
    
    def load_model(self):
        if self.bayesian:
            state_dict = torch.load('models/bayesian_model.pth')
            self.state_dict = state_dict
        else:
            state_dict = torch.load('models/model.pth', map_location=device)
            self.state_dict = state_dict

def train(model, train_loader, test_loader, num_epochs=10):
    min_loss = 9999999
    for epoch in range(num_epochs):
        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            model.optimizer.zero_grad()
            outputs = model(inputs)

            # outputs = get_output(inputs, base_model, fc_layers)
            # import pdb; pdb.set_trace()
            loss = model.criterion(outputs, labels)
            loss.backward()
            model.optimizer.step()
        for inputs, labels in test_loader:
            with torch.no_grad():
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = model(inputs)
                # outputs = get_output(inputs, base_model, fc_layers)
                test_loss = model.criterion(outputs, labels).item()
        print('Epoch [{}/{}], Loss: {:.4f}, Test Loss: {:.4f}'.format(epoch+1, num_epochs, loss.item(), test_loss))
        if test_loss < min_loss:
            print('new best loss... saving')
            model.save_model()
    
if __name__ == '__main__':
    train_loader = get_data('train')
    test_loader = get_data('test')
    model = Face_Model(bayesian=False)
    train(model, train_loader, test_loader)