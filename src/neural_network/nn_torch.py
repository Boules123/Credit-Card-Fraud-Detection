import numpy as np
import torch 
import torch.nn as nn 
import random
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score
from imblearn.over_sampling import SMOTE
    

class MLP(nn.Module):
    def __init__(self, nin, nouts):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(nin, nouts[0]),
            nn.LayerNorm(nouts[0]),
            nn.ReLU(),
            nn.Dropout(0.5),
            
            nn.Linear(nouts[0], nouts[1]),
            nn.LayerNorm(nouts[1]),
            nn.ReLU(),
            nn.Dropout(0.5),
            
            nn.Linear(nouts[1], nouts[2])
        )
    def forward(self, x):
        return self.layers(x)

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, inputs, targets):
        bce_loss = nn.BCEWithLogitsLoss(reduction='none')(inputs, targets)
        prob = torch.sigmoid(inputs) # get probabilities from logits
        pt = prob * targets + (1 - prob) * (1 - targets)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        loss = alpha_t * (1 - pt) ** self.gamma * bce_loss
        return loss.mean()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train():
    # load dataset from csv
    set_seed(42)
    
    df = pd.read_csv('/kaggle/input/datasets/organizations/mlg-ulb/creditcardfraud/creditcard.csv')
    df = df.drop(columns=['Time']) # drop id column if exists
    X = df.drop(columns=['Class']).values
    y = df['Class'].values
    
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
    
    # normalize features only (y is already 0/1 for classification)
    mean_train = X_train.mean(axis=0)
    std_train = X_train.std(axis=0)
    
    X_train = (X_train - mean_train) / (std_train + 1e-8)
    X_val = (X_val - mean_train) / (std_train + 1e-8)
    X_test = (X_test - mean_train) / (std_train + 1e-8)
    
    # handle class imbalance with SMOTE
    # smote = SMOTE(random_state=42)
    # X_train, y_train = smote.fit_resample(X_train, y_train)
    
    train_dataset = torch.utils.data.TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32))
    val_dataset = torch.utils.data.TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.float32))
    test_dataset = torch.utils.data.TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.float32))
    
    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
    valloader = torch.utils.data.DataLoader(val_dataset, batch_size=32)
    testloader = torch.utils.data.DataLoader(test_dataset, batch_size=32)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MLP(X_train.shape[1], [8, 8, 1]).to(device) # 10 inputs, 2 hidden layers of 4 neurons each, 3 outputs
    
    num_pos = y_train.sum()
    num_neg = len(y_train) - num_pos
    pos_weight = torch.tensor([num_neg / num_pos]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    print(f"Using Device: {device}")
    print(f"Num of Model Param: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Len of Trainloader: {len(trainloader):,}")
    print("Starting training...")
    best_f1 = -1.0
    
    for k in range(100):
        epoch_loss = 0.0
        all_preds = []
        all_targets = []
        model.train()
        for i, (x, y) in enumerate(trainloader):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            pred = model(x) # shape(32, 1)
            pred = pred.view(-1) #(32,) flatten to 1D
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            if i % 300 == 0:
                print(f"Epoch {k}, batch {i}, Loss: {epoch_loss / len(trainloader):.6f}")
        
        epoch_loss = 0.0
        model.eval()
        with torch.no_grad():
            for x, y in valloader:
                x, y = x.to(device), y.to(device)
                pred = model(x)
                pred = pred.view(-1) 
                loss = criterion(pred, y)
                epoch_loss += loss.item()
                probs = torch.sigmoid(pred)
                all_preds.extend((probs > 0.5).cpu().numpy())
                all_targets.extend(y.cpu().numpy())
            f1 = f1_score(all_targets, all_preds)
            print(f"Val Loss: {epoch_loss / len(valloader):.6f}, F1 Score: {f1:.4f}")
            if f1 > best_f1:
                best_f1 = f1
                torch.save(model.state_dict(), "best_model.pth")
                print(f"New best model saved with F1 Score: {best_f1:.4f}")
    ckpt = torch.load("best_model.pth", map_location=device)
    model.load_state_dict(ckpt)
    return model, testloader, criterion

def test(model, testloader, criterion):
    # load best model and evaluate on test set
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    ckpt = torch.load("best_model.pth", map_location=device)
    model.load_state_dict(ckpt)
    
    test_loss = 0.0
    all_preds = []
    all_targets = []
    model.eval()
    with torch.no_grad():
        for x, y in testloader:
            x, y = x.to(device), y.to(device)
            pred = model(x)
            pred = pred.view(-1)
            loss = criterion(pred, y)
            test_loss += loss.item()
            probs = torch.sigmoid(pred)
            all_preds.extend((probs > 0.5).cpu().numpy())
            all_targets.extend(y.cpu().numpy())
    f1 = f1_score(all_targets, all_preds)
    print(f"Test Loss: {test_loss / len(testloader):.6f}, F1 Score: {f1:.4f}")

if __name__ == "__main__":
    model, testloader, criterion = train()
    test(model, testloader, criterion)