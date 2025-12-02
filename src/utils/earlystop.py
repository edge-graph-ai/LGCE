import torch


class EarlyStopping:
    """Implements patience-based early stopping with checkpointing."""

    def __init__(self, patience: int = 5, delta: float = 0.0, verbose: bool = False, path: str = "checkpoint.pth"):
        self.patience = patience
        self.delta = delta
        self.verbose = verbose
        self.path = path
        self.counter = 0
        # Initialize the best loss with infinity so any real loss will improve it.
        self.best_loss = float("inf")
        self.early_stop = False
        self.best_model = None

    def __call__(self, val_loss: float, model: torch.nn.Module):
        # Significant validation loss improvement triggers a checkpoint save.
        if val_loss < self.best_loss - self.delta:
            self.best_loss = val_loss
            self.counter = 0
            if self.verbose:
                print(f"Validation loss decreased to {val_loss:.4f}; saving model")
            # Persist the current best model.
            torch.save(model.state_dict(), self.path)
        else:
            self.counter += 1
            if self.verbose:
                print(f"No validation improvement ({self.counter}/{self.patience})")
            # Stop early when patience is exceeded without improvement.
            if self.counter >= self.patience:
                self.early_stop = True
            torch.save(model.state_dict(), "intermediate_model.pth")
