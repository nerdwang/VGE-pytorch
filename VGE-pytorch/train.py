import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from vge import VGE, ConvL2, FClayerL2
from MPIIGaze import MPIIGazeDataset
from util.nn import latent_kl
from util.gaze import pytorch_angular_error_from_pitchyaw

def criterion(model, gaze_direction, y2, gmap, y1, m_v_post, m_v_prior):
        # 计算基本损失
        kl_loss = torch.tensor(0.0, device='cuda')
        for q, p in zip(m_v_post, m_v_prior):
             loss_var = latent_kl(q, p)
             kl_loss += loss_var
        gazemaps_loss = -torch.mean(torch.sum(y1 * torch.log(torch.clamp(gmap, min=1e-10, max=1.0)), dim=[1, 2, 3]))
        gazemaps_ce = 10 * kl_loss + gazemaps_loss
        gaze_mse = torch.mean(torch.square(gaze_direction - y2))
        combined_loss = 1e-4 * gazemaps_ce + gaze_mse
        # 添加ConvL2层和FClayerL2层的L2正则化
        l2_alpha = 1e-4
        for module in model.modules():
            if isinstance(module, ConvL2) or isinstance(module, FClayerL2):
               combined_loss += l2_alpha * module.weight.pow(2.0).sum()



        return combined_loss


# 实例化数据集
i = 0
person_id = 'p%02d' % i
other_person_ids = ['p%02d' % j for j in range(15) if i != j]
train_data = MPIIGazeDataset(hdf_path="/home/ubuntu/VGE-Net/datasets/MPIIGaze.h5", keys_to_use=['train/' + s for s in other_person_ids])
test_data = MPIIGazeDataset(hdf_path="/home/ubuntu/VGE-Net/datasets/MPIIGaze.h5", keys_to_use=['test/' + person_id])

# 创建数据加载器
train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
test_loader = DataLoader(test_data, batch_size=64, shuffle=False)

# 实例化模型
model = VGE()

# 检查是否有多个GPU，如果有，使用所有GPU来构建模型
if torch.cuda.device_count() > 1:
    print(f'Using {torch.cuda.device_count()} GPUs!')
    model = torch.nn.DataParallel(model)

# 确保模型在GPU上
model.to('cuda')

for name, param in model.named_parameters():
    if param.requires_grad:
        print(name, param.shape)
        
# 设置优化器
optimizer = Adam(model.parameters(), lr=6e-4)

# 设置训练轮数
num_epochs = 20

# 训练模型
for epoch in range(num_epochs):
    model.train()
    batch_count = 0  # 用于跟踪批次数量
    for batch in train_loader:
        # 确保数据也在GPU上
        batch = [item.to('cuda') for item in batch if isinstance(item, torch.Tensor)]
        optimizer.zero_grad()
        _ , y2 , y1 = batch
        gaze_direction, gmap, m_v_post, m_v_prior = model(batch)
        loss = criterion(model.module, gaze_direction, y2, gmap, y1, m_v_post, m_v_prior)  # 计算损失
        loss.backward()
        optimizer.step()

        batch_count += 1
        # 每训练完 50 个 batch 进行一次评估并打印角度误差
        if batch_count % 50 == 0:
            model.eval()
            total_angular_error = 0.0
            with torch.no_grad():
                for test_batch in test_loader:
                    # 确保测试数据也在 GPU 上
                    test_batch = [item.to('cuda') for item in test_batch if isinstance(item, torch.Tensor)]
                    _ , test_y2 , _ = test_batch
                    test_gaze_direction, _, _, _ = model(test_batch)
                    # 确保计算角度误差时，所有张量都在 GPU 上
                    angular_error = pytorch_angular_error_from_pitchyaw(test_y2.to('cuda'), test_gaze_direction.to('cuda'))
                    total_angular_error += angular_error.item()
                avg_angular_error = total_angular_error / len(test_loader)
                print(f'Epoch {epoch + 1}, Batch {batch_count}, Average Angular Error: {avg_angular_error} degrees')
            model.train()