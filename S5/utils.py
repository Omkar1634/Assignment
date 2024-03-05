import torch  # PyTorch library for deep learning
from torchvision import datasets, transforms  # Datasets and transformations utilities from torchvision
import torch.nn.functional as F  # Functional interface containing typical operations used for building neural networks like activation functions
import torch.optim as optim  # Optimizers for training neural networks

from tqdm import tqdm

class ModelTrainer:
    def __init__(self):
        self.train_losses = []
        self.test_losses = []
        self.train_acc = []
        self.test_acc = []
        self.test_incorrect_pred = {'images': [], 'ground_truths': [], 'predicted_vals': []}
    
    def get_correct_pred_count(self, prediction, labels):
        return prediction.argmax(dim=1).eq(labels).sum().item()
    
    def train_model(self, model, device, train_loader, optimizer, criterion):
        model.train()
        pbar = tqdm(train_loader)
        train_loss = 0
        correct = 0
        processed = 0
        for batch_idx, (data, target) in enumerate(pbar):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            pred = model(data)
            loss = criterion(pred, target)
            train_loss += loss.item()
            loss.backward()
            optimizer.step()
            correct += self.get_correct_pred_count(pred, target)
            processed += len(data)
            pbar.set_description(desc=f'Train: Loss={loss.item():0.4f} Batch_id={batch_idx} Accuracy={100*correct/processed:0.2f}')
        self.train_acc.append(100*correct/processed)
        self.train_losses.append(train_loss / len(train_loader))
    
    def test_model(self, model, device, test_loader, criterion):
        model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(test_loader):
                data, target = data.to(device), target.to(device)
                output = model(data)
                test_loss += criterion(output, target).item()
                correct += self.get_correct_pred_count(output, target)
        test_loss /= len(test_loader.dataset)
        self.test_acc.append(100. * correct / len(test_loader.dataset))
        self.test_losses.append(test_loss)
        print(f'Test set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({100. * correct / len(test_loader.dataset):.2f}%)')


    def run_training_cycle(self, model, device, train_loader, test_loader, optimizer, criterion, scheduler, num_epochs):
        for epoch in range(1, num_epochs+1):
            print(f'Epoch {epoch}')
            self.train_model(model, device, train_loader, optimizer, criterion)
            self.test_model(model, device, test_loader, criterion)
            scheduler.step()


    





