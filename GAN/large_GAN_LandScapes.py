import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets,transforms,utils
from torchvision.utils import save_image 


image_size = 128
channels_img = 3
z_dim = 100
features_g = 64
features_d = 64
batch_size = 128
lr = 2e-4
num_epochs = 50
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
            self._block(z_dim, features_g * 16, 4, 1, 0),     # Added an extra block for larger image size
            self._block(features_g * 16, features_g * 8, 4, 2, 1),
            self._block(features_g * 8, features_g * 4, 4, 2, 1),
            self._block(features_g * 4, features_g * 2, 4, 2, 1),
            self._block(features_g * 2, features_g * 1, 4, 2, 1),
            nn.ConvTranspose2d(
                features_g,
                channels_img, 
                kernel_size=2,   # Changed kernel size to 2 to achieve 128x128 output
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
            self._block(features_d * 8, features_d * 16, 4, 2, 1),
            nn.Conv2d(features_d * 16, 1, 2, 1, 0),
            nn.Sigmoid(),
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

# 1. 新增：创建主目录和模型子目录（和生成图像目录一致）
main_dir = "generated"
model_dir = os.path.join(main_dir, "large_models")
os.makedirs(main_dir, exist_ok=True)
os.makedirs(model_dir, exist_ok=True)
os.makedirs(os.path.join(main_dir, "large"), exist_ok=True)  # 确保图像保存子目录存在

for epoch in range(num_epochs):
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
        opt_disc.step()
        
        # Generator loss
        output = disc(fake).view(-1)
        loss_gen = criterion(output, torch.ones_like(output))

        gen.zero_grad()   # 等价opt_gen.zero_grad()
        loss_gen.backward()
        opt_gen.step()
    
    print(f"[{epoch+1}/{num_epochs}]  Loss D: {loss_disc:.4f}, loss G: {loss_gen:.4f}")

    with torch.no_grad():
         fake_samples = gen(fix_noise)
         fake_samples = (fake_samples +1) /2  #反归一化到0-1之间
         save_image(fake_samples, f"generated/large/epoch_{epoch+1}.png",nrow=4)


    # 2. 新增：每个epoch结束后保存模型（包含断点续训所需的所有信息）
    checkpoint = {
        "epoch": epoch + 1,
        "gen_state_dict": gen.state_dict(),
        "disc_state_dict": disc.state_dict(),
        "opt_gen_state_dict": opt_gen.state_dict(),
        "opt_disc_state_dict": opt_disc.state_dict(),
        "loss_gen": loss_gen.item(),
        "loss_disc": loss_disc.item()
    }
    torch.save(checkpoint, os.path.join(model_dir, f"checkpoint_epoch_{epoch+1}.pth"))



# 3. 新增：训练结束后保存最终模型（标记为final，方便后续加载）
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
torch.save(final_model, os.path.join(model_dir, "final_model.pth"))

print("Training finished!")
print(f"模型已保存到 {model_dir} 目录下")

