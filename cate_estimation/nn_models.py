"""
TARNet and DragonNet in PyTorch
================================

TARNet (Shalit et al., 2017):
    Shared representation φ(x) → two separate outcome heads μ₀(φ), μ₁(φ)
    Loss = Σ_i [ T_i·BCE(Y_i, μ₁(φ(Xᵢ))) + (1-Tᵢ)·BCE(Yᵢ, μ₀(φ(Xᵢ))) ]

DragonNet (Shi et al., 2019):
    Same as TARNet + propensity head π(φ) from the shared representation.
    Loss = outcome_loss + α·propensity_loss
    
    The propensity head regularizes the shared representation toward features 
    that predict treatment assignment — exactly the features where confounding 
    lives. This reduces extrapolation in regions of poor overlap.

Usage:
    from nn_models_torch import TARNet, DragonNet

    model = DragonNet(input_dim=10)
    model.fit(X, T, Y)
    tau = model.predict_cate(X)
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from sklearn.preprocessing import StandardScaler


# =============================================================================
# TARNet
# =============================================================================

class TARNetModule(nn.Module):
    """
    PyTorch module for TARNet.
    
    Architecture:
        Input → [Shared layers] → φ(x) → [Head 0] → μ₀(x)
                                        → [Head 1] → μ₁(x)
    """
    
    def __init__(self, input_dim, shared_dims=(200, 200), head_dims=(100,),
                 dropout=0.1):
        super().__init__()
        
        # --- Shared representation ---
        shared_layers = []
        prev_dim = input_dim
        for dim in shared_dims:
            shared_layers.extend([
                nn.Linear(prev_dim, dim),
                nn.ELU(),
                nn.Dropout(dropout),
            ])
            prev_dim = dim
        self.shared = nn.Sequential(*shared_layers)
        
        # --- Outcome head 0 (control: ranibizumab) ---
        head0_layers = []
        prev_dim = shared_dims[-1]
        for dim in head_dims:
            head0_layers.extend([
                nn.Linear(prev_dim, dim),
                nn.ELU(),
                nn.Dropout(dropout),
            ])
            prev_dim = dim
        head0_layers.append(nn.Linear(prev_dim, 1))
        self.head0 = nn.Sequential(*head0_layers)
        
        # --- Outcome head 1 (treated: aflibercept) ---
        head1_layers = []
        prev_dim = shared_dims[-1]
        for dim in head_dims:
            head1_layers.extend([
                nn.Linear(prev_dim, dim),
                nn.ELU(),
                nn.Dropout(dropout),
            ])
            prev_dim = dim
        head1_layers.append(nn.Linear(prev_dim, 1))
        self.head1 = nn.Sequential(*head1_layers)
    
    def forward(self, x, t):
        """
        Parameters
        ----------
        x : (batch, input_dim)
        t : (batch,) binary treatment indicator
        
        Returns
        -------
        y0 : (batch,) predicted outcome under control
        y1 : (batch,) predicted outcome under treatment
        """
        phi = self.shared(x)
        y0 = torch.sigmoid(self.head0(phi)).squeeze(-1)
        y1 = torch.sigmoid(self.head1(phi)).squeeze(-1)
        return y0, y1


class TARNet:
    """
    Wrapper class with fit/predict interface for TARNet.
    
    Parameters
    ----------
    input_dim : int
    shared_dims : tuple — hidden sizes for shared representation
    head_dims : tuple — hidden sizes for each outcome head
    lr : float — learning rate
    weight_decay : float — L2 regularization
    dropout : float
    epochs : int — maximum training epochs
    batch_size : int
    patience : int — early stopping patience
    device : str — 'cuda' or 'cpu'
    """
    
    def __init__(self, input_dim, shared_dims=(200, 200), head_dims=(100,),
                 lr=1e-3, weight_decay=1e-3, dropout=0.1,
                 epochs=300, batch_size=128, patience=20, device=None):
        self.input_dim = input_dim
        self.shared_dims = shared_dims
        self.head_dims = head_dims
        self.lr = lr
        self.weight_decay = weight_decay
        self.dropout = dropout
        self.epochs = epochs
        self.batch_size = batch_size
        self.patience = patience
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.scaler = StandardScaler()
        
        self.model = TARNetModule(
            input_dim, shared_dims, head_dims, dropout
        ).to(self.device)
    
    def fit(self, X, T, Y, val_fraction=0.15, verbose=False):
        """
        Train TARNet with early stopping on validation loss.
        
        Parameters
        ----------
        X : np.ndarray (n, d) — covariates
        T : np.ndarray (n,) — binary treatment
        Y : np.ndarray (n,) — binary outcome
        """
        X = self.scaler.fit_transform(X).astype(np.float32)
        T = T.astype(np.float32)
        Y = Y.astype(np.float32)
        
        # Train/val split
        n = len(Y)
        idx = np.random.permutation(n)
        n_val = int(n * val_fraction)
        val_idx, train_idx = idx[:n_val], idx[n_val:]
        
        X_train = torch.tensor(X[train_idx], device=self.device)
        T_train = torch.tensor(T[train_idx], device=self.device)
        Y_train = torch.tensor(Y[train_idx], device=self.device)
        X_val = torch.tensor(X[val_idx], device=self.device)
        T_val = torch.tensor(T[val_idx], device=self.device)
        Y_val = torch.tensor(Y[val_idx], device=self.device)
        
        train_ds = TensorDataset(X_train, T_train, Y_train)
        train_loader = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True)
        
        optimizer = optim.Adam(
            self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        
        best_val_loss = float('inf')
        best_state = None
        wait = 0
        
        for epoch in range(self.epochs):
            # --- Train ---
            self.model.train()
            epoch_loss = 0.0
            for xb, tb, yb in train_loader:
                optimizer.zero_grad()
                loss = self._compute_loss(xb, tb, yb)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * len(xb)
            
            # --- Validate ---
            self.model.eval()
            with torch.no_grad():
                val_loss = self._compute_loss(X_val, T_val, Y_val).item()
            
            if verbose and epoch % 50 == 0:
                print(f"  Epoch {epoch}: train={epoch_loss/len(X_train):.4f}, "
                      f"val={val_loss:.4f}")
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.clone() for k, v in self.model.state_dict().items()}
                wait = 0
            else:
                wait += 1
            
            if wait >= self.patience:
                if verbose:
                    print(f"  Early stopping at epoch {epoch}")
                break
        
        if best_state is not None:
            self.model.load_state_dict(best_state)
        
        return self
    
    def _compute_loss(self, x, t, y):
        """Factual outcome loss: only penalize the head corresponding to observed treatment."""
        y0, y1 = self.model(x, t)
        
        # Select the factual prediction for each unit
        y_pred = t * y1 + (1 - t) * y0
        
        loss = nn.functional.binary_cross_entropy(y_pred, y, reduction='mean')
        return loss
    
    @torch.no_grad()
    def predict_cate(self, X):
        """Predict τ(x) = μ₁(x) - μ₀(x)."""
        self.model.eval()
        X = self.scaler.transform(X).astype(np.float32)
        x = torch.tensor(X, device=self.device)
        y0, y1 = self.model(x, torch.zeros(len(x), device=self.device))
        return (y1 - y0).cpu().numpy()
    
    @torch.no_grad()
    def predict_mu(self, X):
        """Predict both potential outcomes μ₀(x), μ₁(x)."""
        self.model.eval()
        X = self.scaler.transform(X).astype(np.float32)
        x = torch.tensor(X, device=self.device)
        y0, y1 = self.model(x, torch.zeros(len(x), device=self.device))
        return y0.cpu().numpy(), y1.cpu().numpy()


# =============================================================================
# DragonNet
# =============================================================================

class DragonNetModule(nn.Module):
    """
    PyTorch module for DragonNet.
    
    Architecture:
        Input → [Shared layers] → φ(x) → [Head 0] → μ₀(x)
                                        → [Head 1] → μ₁(x)
                                        → [Propensity head] → π(x) = P(T=1|X)
    """
    
    def __init__(self, input_dim, shared_dims=(200, 200, 200),
                 head_dims=(100,), dropout=0.1):
        super().__init__()
        
        # --- Shared representation ---
        shared_layers = []
        prev_dim = input_dim
        for dim in shared_dims:
            shared_layers.extend([
                nn.Linear(prev_dim, dim),
                nn.ELU(),
                nn.Dropout(dropout),
            ])
            prev_dim = dim
        self.shared = nn.Sequential(*shared_layers)
        
        rep_dim = shared_dims[-1]
        
        # --- Outcome head 0 ---
        head0_layers = []
        prev_dim = rep_dim
        for dim in head_dims:
            head0_layers.extend([
                nn.Linear(prev_dim, dim),
                nn.ELU(),
                nn.Dropout(dropout),
            ])
            prev_dim = dim
        head0_layers.append(nn.Linear(prev_dim, 1))
        self.head0 = nn.Sequential(*head0_layers)
        
        # --- Outcome head 1 ---
        head1_layers = []
        prev_dim = rep_dim
        for dim in head_dims:
            head1_layers.extend([
                nn.Linear(prev_dim, dim),
                nn.ELU(),
                nn.Dropout(dropout),
            ])
            prev_dim = dim
        head1_layers.append(nn.Linear(prev_dim, 1))
        self.head1 = nn.Sequential(*head1_layers)
        
        # --- Propensity head ---
        # Simple: single linear layer from representation to logit
        self.propensity_head = nn.Linear(rep_dim, 1)
    
    def forward(self, x):
        """
        Returns
        -------
        y0 : (batch,) predicted P(Y=1|T=0,X)
        y1 : (batch,) predicted P(Y=1|T=1,X)
        pi : (batch,) predicted P(T=1|X)
        """
        phi = self.shared(x)
        y0 = torch.sigmoid(self.head0(phi)).squeeze(-1)
        y1 = torch.sigmoid(self.head1(phi)).squeeze(-1)
        pi = torch.sigmoid(self.propensity_head(phi)).squeeze(-1)
        return y0, y1, pi


class DragonNet:
    """
    Wrapper class with fit/predict interface for DragonNet.
    
    The key difference from TARNet is the propensity loss term:
        L = L_outcome + alpha_prop * L_propensity
    
    This biases the shared representation toward confounding features,
    reducing extrapolation error in CATE estimation.
    
    Parameters
    ----------
    input_dim : int
    shared_dims : tuple
    head_dims : tuple
    alpha_prop : float — weight on propensity loss (default 1.0 per Shi et al.)
    lr, weight_decay, dropout, epochs, batch_size, patience : training params
    targeted_reg : bool — if True, add targeted regularization term (Shi et al. eq. 3)
    """
    
    def __init__(self, input_dim, shared_dims=(200, 200, 200), head_dims=(100,),
                 alpha_prop=1.0, targeted_reg=False, beta_targeted=1.0,
                 lr=1e-3, weight_decay=1e-3, dropout=0.1,
                 epochs=300, batch_size=128, patience=20, device=None):
        self.input_dim = input_dim
        self.alpha_prop = alpha_prop
        self.targeted_reg = targeted_reg
        self.beta_targeted = beta_targeted
        self.epochs = epochs
        self.batch_size = batch_size
        self.patience = patience
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.scaler = StandardScaler()
        
        self.model = DragonNetModule(
            input_dim, shared_dims, head_dims, dropout
        ).to(self.device)
    
    def fit(self, X, T, Y, val_fraction=0.15, verbose=False):
        X = self.scaler.fit_transform(X).astype(np.float32)
        T = T.astype(np.float32)
        Y = Y.astype(np.float32)
        
        n = len(Y)
        idx = np.random.permutation(n)
        n_val = int(n * val_fraction)
        val_idx, train_idx = idx[:n_val], idx[n_val:]
        
        X_train = torch.tensor(X[train_idx], device=self.device)
        T_train = torch.tensor(T[train_idx], device=self.device)
        Y_train = torch.tensor(Y[train_idx], device=self.device)
        X_val = torch.tensor(X[val_idx], device=self.device)
        T_val = torch.tensor(T[val_idx], device=self.device)
        Y_val = torch.tensor(Y[val_idx], device=self.device)
        
        train_ds = TensorDataset(X_train, T_train, Y_train)
        train_loader = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True)
        
        # Shi et al. suggest using SGD with momentum for the targeted reg phase,
        # but Adam works well in practice for the standard DragonNet loss
        optimizer = optim.Adam(
            self.model.parameters(), lr=1e-3, weight_decay=1e-3
        )
        
        best_val_loss = float('inf')
        best_state = None
        wait = 0
        
        for epoch in range(self.epochs):
            self.model.train()
            epoch_loss = 0.0
            for xb, tb, yb in train_loader:
                optimizer.zero_grad()
                loss = self._compute_loss(xb, tb, yb)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * len(xb)
            
            self.model.eval()
            with torch.no_grad():
                val_loss = self._compute_loss(X_val, T_val, Y_val).item()
            
            if verbose and epoch % 50 == 0:
                # Report propensity accuracy too
                with torch.no_grad():
                    _, _, pi_val = self.model(X_val)
                    prop_acc = ((pi_val > 0.5).float() == T_val).float().mean().item()
                print(f"  Epoch {epoch}: train={epoch_loss/len(X_train):.4f}, "
                      f"val={val_loss:.4f}, prop_acc={prop_acc:.3f}")
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.clone() for k, v in self.model.state_dict().items()}
                wait = 0
            else:
                wait += 1
            
            if wait >= self.patience:
                if verbose:
                    print(f"  Early stopping at epoch {epoch}")
                break
        
        if best_state is not None:
            self.model.load_state_dict(best_state)
        
        return self
    
    def _compute_loss(self, x, t, y):
        """
        DragonNet loss = outcome_loss + alpha * propensity_loss [+ beta * targeted_reg]
        """
        y0, y1, pi = self.model(x)
        
        # Factual outcome loss (same as TARNet)
        y_pred = t * y1 + (1 - t) * y0
        outcome_loss = nn.functional.binary_cross_entropy(y_pred, y, reduction='mean')
        
        # Propensity loss
        propensity_loss = nn.functional.binary_cross_entropy(pi, t, reduction='mean')
        
        loss = outcome_loss + self.alpha_prop * propensity_loss
        
        # Optional: targeted regularization (Shi et al., 2019, Section 3.2)
        # This encourages the outcome predictions to be consistent with the
        # propensity scores via an AIPW-like influence function penalty.
        if self.targeted_reg:
            eps = 1e-6
            pi_clamped = torch.clamp(pi, eps, 1 - eps)
            
            # AIPW-style pseudo-outcome
            y1_residual = t * (y - y1) / pi_clamped
            y0_residual = (1 - t) * (y - y0) / (1 - pi_clamped)
            
            targeted_loss = torch.mean((y1_residual - y0_residual) ** 2)
            loss = loss + self.beta_targeted * targeted_loss
        
        return loss
    
    @torch.no_grad()
    def predict_cate(self, X):
        """Predict τ(x) = μ₁(x) - μ₀(x)."""
        self.model.eval()
        X = self.scaler.transform(X).astype(np.float32)
        x = torch.tensor(X, device=self.device)
        y0, y1, _ = self.model(x)
        return (y1 - y0).cpu().numpy()
    
    @torch.no_grad()
    def predict_mu(self, X):
        """Predict both potential outcomes."""
        self.model.eval()
        X = self.scaler.transform(X).astype(np.float32)
        x = torch.tensor(X, device=self.device)
        y0, y1, _ = self.model(x)
        return y0.cpu().numpy(), y1.cpu().numpy()
    
    @torch.no_grad()
    def predict_propensity(self, X):
        """Predict P(T=1|X) from the propensity head."""
        self.model.eval()
        X = self.scaler.transform(X).astype(np.float32)
        x = torch.tensor(X, device=self.device)
        _, _, pi = self.model(x)
        return pi.cpu().numpy()

if __name__ == '__main__':
    # Quick smoke test with synthetic data
    np.random.seed(42)
    torch.manual_seed(42)
    
    n = 2000
    d = 5
    X = np.random.randn(n, d)
    
    # True propensity and outcome functions
    true_e = 1 / (1 + np.exp(-(X[:, 0] + 0.5 * X[:, 1])))
    T = (np.random.rand(n) < true_e).astype(float)
    
    # True CATE: τ(x) = 0.2 * x₀ (heterogeneous)
    true_tau = 0.2 * X[:, 0]
    mu0 = 1 / (1 + np.exp(-(X[:, 0] + X[:, 1])))
    mu1 = 1 / (1 + np.exp(-(X[:, 0] + X[:, 1] + true_tau)))
    Y = (np.random.rand(n) < (T * mu1 + (1 - T) * mu0)).astype(float)
    
    print("=" * 60)
    print("SMOKE TEST: Synthetic data with known CATE")
    print("=" * 60)
    print(f"True mean CATE: {true_tau.mean():.4f}")
    print(f"True CATE std:  {true_tau.std():.4f}")
    
    for name, ModelClass in [("TARNet", TARNet), ("DragonNet", DragonNet)]:
        print(f"\n--- {name} ---")
        model = ModelClass(input_dim=d)
        model.fit(X, T, Y, verbose=True)
        tau_hat = model.predict_cate(X)
        
        print(f"  Predicted mean CATE: {tau_hat.mean():.4f}")
        print(f"  Predicted CATE std:  {tau_hat.std():.4f}")
        
        # Correlation with true CATE
        corr = np.corrcoef(true_tau, tau_hat)[0, 1]
        print(f"  Correlation with true CATE: {corr:.4f}")
        
        # MSE of CATE
        mse = np.mean((tau_hat - true_tau) ** 2)
        print(f"  CATE MSE: {mse:.4f}")
    
    print("\nSmoke test passed.")