import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.optim as optim
import visdom
import os
# 可视化
viz = visdom.Visdom(server="http://127.0.0.1", port=8097)
# viz = visdom.Visdom()

# 设置超参数
train_parameters = {
    "epochs": 20,
    "batch": 500,
    "learning_rate": 0.001
}

# 导入数据
train_db = datasets.MNIST('data', download=True, train=True, transform=transforms.Compose([transforms.ToTensor()]))
train_db, val_db = torch.utils.data.random_split(dataset=train_db, lengths=[50000, 10000])
test_db = datasets.MNIST('data', download=True, train=False, transform=transforms.Compose([transforms.ToTensor()]))

train_db_loader = DataLoader(dataset=train_db, shuffle=True, batch_size=train_parameters["batch"])
val_db_loader = DataLoader(dataset=val_db, shuffle=True, batch_size=train_parameters["batch"] * 2)
test_db_loader = DataLoader(dataset=test_db, shuffle=True, batch_size=train_parameters["batch"])
# 构建神经网络模型


class Mnist(nn.Module):
    def __init__(self):
        super(Mnist, self).__init__()
        self.Flatten = nn.Flatten()
        self.d1 = nn.Linear(784, 200)
        self.d2 = nn.Linear(200, 80)
        self.d3 = nn.Linear(80, 10)

    def forward(self, x):
        x = self.Flatten(x)
        x = self.d1(x)
        x = F.leaky_relu_(x)
        x = self.d2(x)
        x = F.leaky_relu_(x)
        x = self.d3(x)
        output = F.softmax(x)
        return output

# 获得模型
model = Mnist()
# torch.cuda.is_available()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#  图像初始化
viz.line([0.], [0.], win='train_loss', opts=dict(title="train_loss"))
viz.line([0.], [0.], win='val_loss', opts=dict(title='val_loss'))
viz.line([0.], [0.], win='val_acc', opts=dict(title='val_acc'))

# 保存断点
RESUME = False
check_point_path = "./checkpoint/_epoch_15.pth"
if os.path.exists(check_point_path):
    RESUME = True
    check_point = torch.load(check_point_path, map_location=device)  # 使用map_location将模型参数置于gpu上

    start_epoch = check_point["start_epoch"]
    optimizer = check_point["optimizer"]
    model.load_state_dict(check_point["weight"])
else:
    if not os.path.exists("./checkpoint/"):
        os.mkdir("./checkpoint/")
model.to(device)  # 将模型置于device上
# 配置
if not RESUME:
    start_epoch = 0
    optimizer = optim.Adam(model.parameters(), lr=train_parameters["learning_rate"])
loss_function = nn.CrossEntropyLoss()

# 训练
val_global_step = 0
global_step = 0
for epoch in range(start_epoch, train_parameters["epochs"]):
    model.train()
    for step, (x, y) in enumerate(train_db_loader):
        global_step += 1
        if torch.cuda.is_available():  # 判断gpu是否可以用
            x = x.cuda()
            y = y.cuda()
        pred = model(x)
        train_loss = loss_function(pred, y)
        optimizer.zero_grad()  # 优化器梯度清空
        train_loss.backward()  # 误差反向传播
        optimizer.step()  # 更新权重
        # 输出
        print("epoch:{0} step:{1}` train_loss:{2}".format(epoch, step, train_loss))
        viz.line([train_loss.item()], [global_step], win='train_loss', update='append')
    if epoch % 5 == 0:  # 每5个epoch保存一次模型
        check_point = {
            "start_epoch": epoch,
            "optimizer": optimizer,
            "weight": model.state_dict()
        }
        torch.save(check_point, "./checkpoint/" + "_epoch_" + str(epoch) + ".pth")
    # 验证模型
    model.eval()
    with torch.no_grad():
        val_correct = 0
        val_total = 0
        val_loss_total = 0
        for step, (val_x, val_y) in enumerate(val_db_loader):
            if torch.cuda.is_available():  # 判断gpu是否可以使用
                val_x = val_x.cuda()
                val_y = val_y.cuda()
            val_correct = 0
            val_global_step += 1
            pred = model(val_x)
            val_loss = loss_function(pred, val_y)
            val_loss_total += val_loss

            pred = torch.argmax(pred, axis=-1)
            val_correct += torch.eq(pred, val_y).float().sum()
            acc = 100 * (val_correct / (val_x.size()[0]))
            print("epoch:{0} step:{1} val_loss:{2}".format(epoch, step, val_loss))
            viz.line([val_loss.item()], [val_global_step], win='val_loss', update='append')
            viz.line([acc.item()], [val_global_step], win='val_acc', update='append')

# 测试模型
nums = 0
correct = 0
for (x, y) in test_db_loader:
    if torch.cuda.is_available():
        x = x.cuda()
        y = y.cuda()
    pred = model(x)
    pred = torch.argmax(pred, axis=-1)
    nums += x.size(0)
    correct += torch.eq(pred, y).float().sum()
    # viz.text(str(pred), win="prediction")
    # # viz.text(str(y), win="answers")
    # viz.images(x.view(-1, 1, 28, 28)[0:36])
print("accuracy:{0}".format(correct / nums))






