# ABN_Optimizer.py —— 展昭最终版
import torch
from torch.optim import Optimizer

class LayerProjector:
    def __init__(self, dim, buffer_size=10, target_rank_ratio=0.15):
        self.dim = dim
        self.buffer = []
        self.buffer_size = min(buffer_size, dim)
        self.target_rank = max(1, int(dim * target_rank_ratio))
        # 直接存储单位阵（初始不干预）
        self.P = torch.eye(dim, dtype=torch.float32)
        self.step_counter = 0
        self.is_ready = False

    def update(self, grad_flat):
        self.buffer.append(grad_flat.detach().clone())
        if len(self.buffer) > self.buffer_size:
            self.buffer.pop(0)
        self.step_counter += 1
        if self.step_counter % 5 != 0 or len(self.buffer) < self.buffer_size:
            return
        G = torch.stack(self.buffer, dim=1)
        try:
            U, S, Vt = torch.linalg.svd(G, full_matrices=False)
        except RuntimeError:
            return
        rank = min(self.target_rank, G.shape[1] - 1)
        if rank >= G.shape[1]:
            return
        null_basis = Vt[rank:, :].T
        self.P = null_basis @ null_basis.T
        self.is_ready = True

    def project(self, grad_flat):
        if not self.is_ready:
            return grad_flat, 0.0
        proj_g = self.P @ grad_flat
        orig_norm = torch.norm(grad_flat)
        proj_norm = torch.norm(proj_g)
        reduction = 1.0 - (proj_norm / (orig_norm + 1e-12))
        return proj_g, reduction


class ABNAdamW(Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0.01, target_rank_ratio=0.15,
                 max_proj_dim=20000):  # 允许最大20k维参数投影
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super(ABNAdamW, self).__init__(params, defaults)
        self.target_rank_ratio = target_rank_ratio
        self.projectors = {}
        self.max_proj_dim = max_proj_dim

        for group in self.param_groups:
            for p in group['params']:
                if p.requires_grad:
                    dim = p.numel()
                    # 只对维度在合理范围内的参数创建投影器（避免内存爆炸）
                    if 100 < dim <= self.max_proj_dim:
                        self.projectors[id(p)] = LayerProjector(
                            dim=dim,
                            buffer_size=10,
                            target_rank_ratio=target_rank_ratio
                        )
                        print(f"✅ 为参数 {id(p)} 创建投影器，维度 {dim}")
                    else:
                        print(f"⏭️ 跳过参数 {id(p)}，维度 {dim}（不投影）")
        self.reduction_log = []

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            betas = group['betas']
            eps = group['eps']
            weight_decay = group['weight_decay']

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('ABNAdamW does not support sparse gradients')

                # 获取投影器，若不存在则使用原始梯度
                proj = self.projectors.get(id(p))
                if proj is not None:
                    grad_flat = grad.view(-1)
                    proj_grad_flat, reduction = proj.project(grad_flat)
                    self.reduction_log.append(reduction)
                    proj.update(grad_flat)  # 更新历史
                    proj_grad = proj_grad_flat.view(grad.shape)
                else:
                    proj_grad = grad  # 原始梯度

                # ----- 标准 AdamW 更新（使用 proj_grad）-----
                state = self.state[p]
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p.data)
                    state['exp_avg_sq'] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']
                state['step'] += 1
                t = state['step']

                if weight_decay != 0:
                    p.data.mul_(1 - lr * weight_decay)

                exp_avg.mul_(beta1).add_(proj_grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(proj_grad, proj_grad, value=1 - beta2)

                bias_correction1 = 1 - beta1 ** t
                bias_correction2 = 1 - beta2 ** t
                denom = (exp_avg_sq.sqrt() / (bias_correction2 ** 0.5)).add_(eps)
                step_size = lr / bias_correction1
                p.data.addcdiv_(exp_avg, denom, value=-step_size)

        return loss
