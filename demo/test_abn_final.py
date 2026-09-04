# test_abn_final.py —— 包青天终审版
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei']  # 微軟正黑體
plt.rcParams['axes.unicode_minus'] = False  # 負號正常顯示

# ---------- 模型 ----------
class TinyLanguageModel(nn.Module):
    def __init__(self, vocab_size=128, d_model=64, nhead=4, num_layers=3, seq_len=32):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos_enc = nn.Parameter(torch.randn(1, seq_len, d_model) * 0.1)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model*4,
            batch_first=True, dropout=0.0
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, vocab_size)
    def forward(self, x):
        x = self.embed(x) + self.pos_enc
        x = self.transformer(x)
        return self.fc(x)

# ---------- 数据 ----------
torch.manual_seed(42)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"使用设备: {device}")

def create_dataloader(batch_size=8, seq_len=32, vocab_size=128, num_batches=40):
    data = torch.randint(0, vocab_size, (num_batches * batch_size, seq_len))
    return [(data[i*batch_size:(i+1)*batch_size], data[i*batch_size:(i+1)*batch_size])
            for i in range(num_batches)]

# ---------- 训练函数（通过名称获取目标参数） ----------
def run_epoch(model, dataloader, optimizer, device, target_layer_name='fc.weight',
              rank_ratio=0.15, buffer_size=10):
    model.train()
    criterion = nn.CrossEntropyLoss()
    losses = []
    grad_norms = []
    target_grad_norms = []

    # ---- 根据名称获取目标参数 ----
    target_param = None
    for name, param in model.named_parameters():
        if name == target_layer_name:
            target_param = param
            break
    if target_param is None:
        raise ValueError(f"未找到层 {target_layer_name}")

    dim = target_param.numel()
    print(f"🎯 目标层 '{target_layer_name}' 参数量: {dim}")

    grad_history = []

    for step, (inputs, targets) in enumerate(dataloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        logits = model(inputs)
        loss = criterion(logits.view(-1, logits.size(-1)), targets.view(-1))
        loss.backward()

        # ---- 检查目标梯度 ----
        if target_param.grad is None:
            print(f"⚠️ 第 {step} 步：目标层梯度为 None，跳过投影")
            target_grad_norms.append(0.0)
        else:
            grad_flat = target_param.grad.data.view(-1)
            # 关键断言：确保维度匹配
            assert grad_flat.numel() == dim, \
                f"❌ 维度错误！梯度 {grad_flat.numel()} vs 参数 {dim}"

            # 存储历史
            grad_history.append(grad_flat.detach().clone())
            if len(grad_history) > buffer_size:
                grad_history.pop(0)

            # 当历史足够时构建投影
            if len(grad_history) == buffer_size:
                G = torch.stack(grad_history, dim=1)  # [dim, buffer_size]
                try:
                    U, S, Vt = torch.linalg.svd(G, full_matrices=False)
                    rank = max(1, int(dim * rank_ratio))
                    if rank < dim:
                        null_basis = Vt[rank:, :].T
                        P = null_basis @ null_basis.T
                        proj_grad = P @ grad_flat
                        target_param.grad.data = proj_grad.view(target_param.grad.shape)
                        target_grad_norms.append(torch.norm(proj_grad).item())
                    else:
                        target_grad_norms.append(torch.norm(grad_flat).item())
                except Exception as e:
                    print(f"⚠️ SVD 失败: {e}")
                    target_grad_norms.append(torch.norm(grad_flat).item())
            else:
                # 历史不足，不投影，记录原始范数
                target_grad_norms.append(torch.norm(grad_flat).item())

        # ---- 计算全局梯度范数 ----
        total_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                total_norm += p.grad.norm().item() ** 2
        grad_norms.append(total_norm ** 0.5)

        optimizer.step()
        losses.append(loss.item())

        if len(losses) >= 40:
            break

    return losses, grad_norms, target_grad_norms

# ---------- 主程序 ----------
if __name__ == "__main__":
    vocab_size = 128
    seq_len = 32
    d_model = 64
    batch_size = 8
    num_batches = 40
    lr = 1e-3

    dataloader = create_dataloader(batch_size, seq_len, vocab_size, num_batches)

    # ---- 原始 AdamW ----
    print("\n⚔️ 运行原始 AdamW...")
    model_base = TinyLanguageModel(vocab_size, d_model).to(device)
    opt_base = optim.AdamW(model_base.parameters(), lr=lr)
    losses_base, norms_base, _ = run_epoch(model_base, dataloader, opt_base, device,
                                           target_layer_name='fc.weight')  # 仍传入名称，但不影响

    # ---- 手动投影版 ----
    print("\n⚔️ 运行手动投影版（展昭）...")
    model_abn = TinyLanguageModel(vocab_size, d_model).to(device)
    model_abn.load_state_dict(model_base.state_dict())
    opt_abn = optim.AdamW(model_abn.parameters(), lr=lr)
    losses_abn, norms_abn, target_norms = run_epoch(
        model_abn, dataloader, opt_abn, device,
        target_layer_name='fc.weight',  # 明确指定
        rank_ratio=0.15,
        buffer_size=10
    )

    # ---- 绘图 ----
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4))

    ax1.plot(losses_base, label='AdamW (原始)', color='red', alpha=0.7, marker='o')
    ax1.plot(losses_abn, label='ABN-投影 (展昭)', color='blue', linewidth=2, marker='s')
    ax1.set_title('判決書：損失收斂速度')
    ax1.set_xlabel('迭代步數')
    ax1.set_ylabel('交叉熵損失')
    ax1.legend(); ax1.grid(True)

    ax2.plot(norms_base, label='原始全局梯度範數', color='red', alpha=0.5)
    ax2.plot(norms_abn, label='投影後全局梯度範數', color='blue', linewidth=2)
    ax2.set_title('全局梯度範數對比')
    ax2.set_xlabel('迭代步數')
    ax2.set_ylabel('梯度 L2 範數')
    ax2.legend(); ax2.grid(True)

    if target_norms and len(target_norms) > 0:
        ax3.plot(target_norms, label='投影後 fc.weight 梯度範數', color='green', linewidth=2)
        ax3.set_title('投影層（fc.weight）梯度範數')
        ax3.set_xlabel('迭代步數')
        ax3.set_ylabel('梯度 L2 範數')
        ax3.legend(); ax3.grid(True)

    if len(norms_abn) > 5 and len(norms_base) > 5:
        final_reduction = (1 - np.mean(norms_abn[-5:]) / (np.mean(norms_base[-5:]) + 1e-8)) * 100
        ax2.text(0.5, 0.9, f'最終全局降熵：{final_reduction:.1f}%',
                 transform=ax2.transAxes, ha='center', fontsize=12, color='blue',
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.suptitle("包青天 · 明斷『熵增』案 | 展昭終審定版", fontsize=14)
    plt.tight_layout()
    plt.savefig("verdict_final.png", dpi=150)
    print("\n✅ 判決書已存檔：verdict_final.png")
    if len(norms_abn) > 5 and len(norms_base) > 5:
        print(f"📊 原始 AdamW 最後 5 步平均梯度範數：{np.mean(norms_base[-5:]):.4f}")
        print(f"📊 ABN-投影 最後 5 步平均梯度範數：{np.mean(norms_abn[-5:]):.4f}")
        print(f"⚔️ 全局降熵幅度：{final_reduction:.1f}%")
