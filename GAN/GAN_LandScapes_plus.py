import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets,transforms,utils
from torchvision.utils import save_image 


image_size = 64
channels_img = 3
z_dim = 100
features_g = 64
features_d = 64
batch_size = 128
lr = 2e-4
num_epochs = 200
device = torch.device("cuda")

transform = transforms.Compose([
    transforms.Resize(image_size),
    transforms.CenterCrop(image_size),
    transforms.ToTensor(),
    transforms.Normalize([0.5 for _ in range(channels_img)],[0.5 for _ in range(channels_img)])
])

dataset = datasets.ImageFolder(root="/home/slorn/work/dataset/landscapes",transform=transform)
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# DCGAN
# Generator
class Generator(nn.Module):
    def __init__(self, z_dim, channels_img, features_g):
        super().__init__()
        self.net = nn.Sequential(
            self._block(z_dim, features_g * 8, 4, 1, 0),
            self._block(features_g * 8, features_g * 4, 4, 2, 1),
            self._block(features_g * 4, features_g * 2, 4, 2, 1),
            self._block(features_g * 2, features_g * 1, 4, 2, 1),
            nn.ConvTranspose2d(
                features_g,
                channels_img, 
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False
            ),
            nn.Tanh()
        )

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.ConvTranspose2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True)
        )
    
    def forward(self, x):
        return self.net(x)
    
# Discriminator
class Discriminator(nn.Module):
    def __init__(self, channels_img, features_d):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels_img, features_d, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            self._block(features_d, features_d * 2, 4, 2, 1),
            self._block(features_d * 2, features_d * 4, 4, 2, 1),
            self._block(features_d * 4, features_d * 8, 4, 2, 1),
            nn.Conv2d(features_d * 8, 1, 4, 1, 0),
            nn.Sigmoid()
        )

    def _block(self, in_channels, out_channels,kernel_size,stride,padding):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels,kernel_size,stride,padding,bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True)
        )
    
    def forward(self,x):
        return self.net(x)

#model initialization
gen = Generator(z_dim, channels_img, features_g).to(device)
disc = Discriminator(channels_img, features_d).to(device)


#optimizer
criterion = nn.BCELoss()
opt_gen = optim.Adam(gen.parameters(),lr=lr,betas=(0.5,0.999))
opt_disc = optim.Adam(disc.parameters(),lr=lr,betas=(0.5,0.999))
# 初始化固定噪声：64个噪声向量，维度是z_dim，形状(64, z_dim, 1, 1)（适配生成器输入）
fix_noise = torch.randn(64,z_dim,1,1).to(device)

# training loop
print("Traning started...")


# 定义绝对路径（关键！避免相对路径权限问题）
BASE_DIR = os.path.abspath(".")  # 当前代码所在目录的绝对路径
MAIN_DIR = os.path.join(BASE_DIR, "generated")
IMAGE_DIR = os.path.join(MAIN_DIR, "small")
MODEL_DIR = os.path.join(MAIN_DIR, "small_models")

# 一次性创建所有目录，并显式赋予当前用户可写权限
for dir_path in [MAIN_DIR, IMAGE_DIR, MODEL_DIR]:
    os.makedirs(dir_path, exist_ok=True)
    # 赋予目录可写权限（Linux下关键）
    # os.chmod(dir_path, 0o755)  # rwxr-xr-x，当前用户可写



for epoch in range(num_epochs):
    # 初始化损失，避免空值
    epoch_loss_disc = 0.0
    epoch_loss_gen = 0.0
    
    for batch_idx, (real, _) in enumerate(loader):
        real = real.to(device)
        noise = torch.randn(real.size(0), z_dim, 1, 1).to(device)
        fake = gen(noise)

        # Discriminator loss
        disc_real = disc(real).view(-1)
        disc_fake = disc(fake.detach()).view(-1)
        loss_disc = criterion(disc_real,torch.ones_like(disc_real))+criterion(disc_fake, torch.zeros_like(disc_fake))

        disc.zero_grad()   # 等价opt_disc.zero_grad()
        loss_disc.backward()
        # 梯度裁剪：解决损失波动/NaN问题（关键！）
        torch.nn.utils.clip_grad_norm_(disc.parameters(), max_norm=1.0)
        opt_disc.step()
        
        # Generator loss
        output = disc(fake).view(-1)
        loss_gen = criterion(output, torch.ones_like(output))

        gen.zero_grad()   # 等价opt_gen.zero_grad()
        torch.nn.utils.clip_grad_norm_(gen.parameters(), max_norm=1.0)
        loss_gen.backward()
        opt_gen.step()


        # 累加批次损失
        epoch_loss_disc += loss_disc.item()
        epoch_loss_gen += loss_gen.item()
    
    # 计算epoch平均损失（更稳定）
    avg_loss_disc = epoch_loss_disc / len(loader)
    avg_loss_gen = epoch_loss_gen / len(loader)
    print(f"[{epoch+1}/{num_epochs}]  Loss D: {avg_loss_disc:.4f}, loss G: {avg_loss_gen:.4f}")
   

    with torch.no_grad():
        fake_samples = gen(fix_noise)
        fake_samples = (fake_samples + 1) / 2  # 反归一化到0-1
        # 1. 生成绝对保存路径
        save_path = os.path.join(IMAGE_DIR, f"epoch_{epoch+1}.png")
        # 2. 检查并删除已存在的文件（避免锁定）
        if os.path.exists(save_path):
            os.remove(save_path)
        # 3. 确保张量在CPU且无NaN/Inf
        fake_samples = fake_samples.detach().cpu()
        fake_samples = torch.clamp(fake_samples, 0.0, 1.0)  # 限制像素值范围
        # 4. 保存图片
        save_image(fake_samples, save_path, nrow=4, normalize=False)  # 已反归一化，无需再normalize


    # 每个epoch结束后保存模型（包含断点续训所需的所有信息）
    checkpoint = {
        "epoch": epoch + 1,
        "gen_state_dict": gen.state_dict(),
        "disc_state_dict": disc.state_dict(),
        "opt_gen_state_dict": opt_gen.state_dict(),
        "opt_disc_state_dict": opt_disc.state_dict(),
        "loss_gen": loss_gen.item(),
        "loss_disc": loss_disc.item()
    }
    model_save_path = os.path.join(MODEL_DIR, f"checkpoint_epoch_{epoch+1}.pth")
    # 避免模型文件锁定
    if os.path.exists(model_save_path):
        os.remove(model_save_path)
    torch.save(checkpoint, model_save_path)


# 训练结束后保存最终模型（标记为final，方便后续加载）
final_model = {
    "gen_state_dict": gen.state_dict(),
    "disc_state_dict": disc.state_dict(),
    "hyperparams": {  # 保存超参数，方便后续复现
        "image_size": image_size,
        "z_dim": z_dim,
        "features_g": features_g,
        "features_d": features_d,
        "lr": lr,
        "num_epochs": num_epochs
    }
}
torch.save(final_model, os.path.join(MODEL_DIR, "final_model.pth"))

print("Training finished!")
print(f"模型已保存到 {MODEL_DIR} 目录下")
print(f"生成的图片已保存到 {IMAGE_DIR} 目录下")
