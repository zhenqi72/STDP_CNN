import torch
import torch.nn as nn
import torch.optim as optim
import torchvision

class MultiClassSVM(nn.Module):
    def __init__(self, input_size, num_classes):
        super(MultiClassSVM, self).__init__()
        self.linear = nn.Linear(input_size, num_classes)

    def forward(self, x):
        outputs = self.linear(x)

        return outputs
    
def multiclass_hinge_loss(outputs, labels):
    # 获取类别数
    num_classes = outputs.size(1)
    # 正确标签类别的得分
    correct_scores = outputs[torch.arange(1), labels]
    correct_scores = correct_scores.view(1,1)
    correct_scores = correct_scores.repeat(1,2)
    margins = torch.clamp(1 - (correct_scores - outputs), min=0)
    print("margins size",margins.shape)
    margins[torch.arange(1), labels] = 0  # 正确标签的损失设置为0
    loss = margins.sum() / 1
    
    return loss
import torch

def binary_hinge_loss(outputs, labels):
    t = 2 * labels - 1  
    t = t.float()  
    y = outputs[torch.arange(outputs.size(0)), labels]
    loss = torch.clamp(1 - t * y, min=0)
    
    # 返回平均损失
    return loss.mean()


def train(
    model,
    train_loader,    
    ):
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    for batch_idx, (data, label) in enumerate(train_loader):
        optimizer.zero_grad()
        #print("size of data is",data.shape)
        data = data.view(data.shape[0],-1)
        output = model(data)
        loss = multiclass_hinge_loss(output, label)
        loss.backward()
        optimizer.step()
        _, argmax = torch.max(output, 1)
        accuracy = (label == argmax.squeeze()).float().mean()
        print("for epoch {} accuracy is {}".format(batch_idx,accuracy))

def main():
    svm = MultiClassSVM(input_size=28*28,num_classes=10)
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


