import torch
import torch.nn as nn
import torch.optim as optim
import torchvision

class MultiClassSVM(nn.Module):
    def __init__(self, input_size, num_classes):
        super(MultiClassSVM, self).__init__()
        self.linear = nn.Linear(input_size, num_classes)

    def forward(self, x):
        return self.linear(x)
    
def multiclass_hinge_loss(outputs, labels):
    # 获取类别数
    num_classes = outputs.size(1)

    # 正确标签类别的得分
    correct_scores = outputs[torch.arange(len(labels)), labels].unsqueeze(1)

    # 计算合页损失
    margins = torch.clamp(1 - (correct_scores - outputs), min=0)
    margins[torch.arange(len(labels)), labels] = 0  # 正确标签的损失设置为0
    loss = margins.sum() / len(labels)
    
    return loss

def train(
    model,
    train_loader,    
    ):
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(data)
        loss = multiclass_hinge_loss(output, target)
        loss.backward()
        optimizer.step()
        _, argmax = torch.max(output, 1)
        accuracy = (target == argmax.squeeze()).float().mean()
        print("for epoch {} accuracy is",batch_idx,accuracy)

def main():
    svm = MultiClassSVM(input_size=28,num_classes=10)
    train_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST(
            root=".",
            train=True,
            download=True,
            transform=torchvision.transforms.Compose(
                [
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Resize((28,28))                    
                ]
            ),
            
        ),
        batch_size=64,
        shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST(
            root=".",
            train=False,
            transform=torchvision.transforms.Compose(
                [
                    torchvision.transforms.PILToTensor(),
                    
                ]
            ),
        ),
        batch_size=64,
        
    )
    train(svm,train_loader)

if __name__ == "__main__":
    main()
