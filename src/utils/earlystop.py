import torch

class EarlyStopping:
    def __init__(self, patience=5, delta=0.0, verbose=False, path="checkpoint.pth"):
        """
        初始化 EarlyStopping 的参数
        :param patience: 容忍的epoch数，验证损失没有改进就停止训练
        :param delta: 损失改善的最小阈值，若小于该值则认为没有改善
        :param verbose: 是否打印早停信息
        :param path: 最佳模型保存的路径
        """
        self.patience = patience
        self.delta = delta
        self.verbose = verbose
        self.path = path
        self.counter = 0
        self.best_loss = float('inf')  # 初始时损失为无穷大
        self.early_stop = False
        self.best_model = None

    def __call__(self, val_loss, model):
        """
        在每个epoch结束时调用，判断是否提前停止训练
        :param val_loss: 当前epoch的验证损失
        :param model: 当前模型，若验证损失改进，保存模型
        """
        if val_loss < self.best_loss - self.delta:  # 如果验证损失显著降低
            self.best_loss = val_loss
            self.counter = 0
            if self.verbose:
                print(f"验证损失降低到 {val_loss:.4f}，保存模型")
            # 保存当前最优模型
            torch.save(model.state_dict(), self.path)
        else:
            self.counter += 1
            if self.verbose:
                print(f"验证损失没有改善 ({self.counter}/{self.patience})")
            if self.counter >= self.patience:  # 如果超过patience个epoch没有改善
                self.early_stop = True
            torch.save(model.state_dict(), "intermediate_model.pth")