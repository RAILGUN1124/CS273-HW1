import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import time

warnings.filterwarnings('ignore')

sns.set_style('whitegrid')
plt.rcParams['font.size'] = 9

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running on device: {DEVICE}")


# ============================================================================
# MODEL DEFINITION
# ============================================================================

class MultiTaskDNN(nn.Module):
    def __init__(self, input_dim, target_names, hidden_dim=128, dropout=0.3):
        super(MultiTaskDNN, self).__init__()
        self.target_names = target_names

        self.shared_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim // 2)
        )

        self.heads = nn.ModuleDict()
        for target in target_names:
            self.heads[target] = nn.Linear(hidden_dim // 2, 1)

    def forward(self, x):
        shared_out = self.shared_layer(x)
        return {target: head(shared_out) for target, head in self.heads.items()}


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def load_and_prepare_data(filepath, dataset_name):
    """Load preprocessed data and separate features from available targets"""
    try:
        df = pd.read_csv(filepath)
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return None, None, None

    possible_targets = ['valence', 'arousal', 'dominance']
    existing_targets = [t for t in possible_targets if t in df.columns]

    if not existing_targets:
        print(f"Error: No valid targets found in {dataset_name}")
        return None, None, None

    df = df.dropna(subset=existing_targets)

    df = df.fillna(df.mean(numeric_only=True))

    exclude_cols = possible_targets + ['dataset', 'genre', 'category', 'fnames', 'splits',
                                       'vocals', 'emotional_intensity', 'emotional_quadrant',
                                       'source', 'description', 'fname', 'BE_Classification']

    feature_cols = [col for col in df.columns if col not in exclude_cols and np.issubdtype(df[col].dtype, np.number)]

    X = df[feature_cols].values
    y = df[existing_targets].values

    return X, y, existing_targets


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0

    for batch_data in loader:
        X_batch = batch_data[0].to(device)
        target_batches = {name: batch_data[i + 1].to(device) for i, name in enumerate(model.target_names)}

        optimizer.zero_grad()
        preds = model(X_batch)

        loss = 0
        for name in model.target_names:
            loss += criterion(preds[name], target_batches[name])

        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    return running_loss / len(loader)


def plot_training_history(history, dataset_name):
    plt.figure(figsize=(10, 5))
    plt.plot(history['train_loss'], label='Training Loss', linewidth=2)
    plt.plot(history['val_loss'], label='Validation Loss', linewidth=2, linestyle='--')
    plt.title(f'DNN Implementation - {dataset_name}: Model Convergence')
    plt.xlabel('Epochs')
    plt.ylabel('MSE Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    filename = f'DNN_{dataset_name}_loss_curve.png'
    plt.savefig(filename, dpi=300)
    plt.close()
    print(f"Saved graph: {filename}")


def plot_predictions(y_true, y_pred, target_names, dataset_name):
    n_targets = len(target_names)
    fig, axes = plt.subplots(1, n_targets, figsize=(5 * n_targets, 5))

    if n_targets == 1: axes = [axes]

    for i, target in enumerate(target_names):
        ax = axes[i]
        sns.scatterplot(x=y_true[:, i], y=y_pred[:, i], ax=ax, alpha=0.5, color='teal')

        min_val = min(y_true[:, i].min(), y_pred[:, i].min())
        max_val = max(y_true[:, i].max(), y_pred[:, i].max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')

        ax.set_title(f'{target.capitalize()}')
        ax.set_xlabel('Actual Value')
        ax.set_ylabel('Predicted Value')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.suptitle(f'DNN Implementation - {dataset_name}: Actual vs Predicted', fontsize=14)
    plt.tight_layout()
    filename = f'DNN_{dataset_name}_actual_vs_pred.png'
    plt.savefig(filename, dpi=300)
    plt.close()
    print(f"Saved graph: {filename}")


# ============================================================================
# PROCESSING LOOP
# ============================================================================

def process_dataset(filepath, dataset_name, k_features=34):
    print(f"\nProcessing {dataset_name}...")

    X, y, target_names = load_and_prepare_data(filepath, dataset_name)
    if X is None: return []

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    fold_results = {'MSE': [], 'R2': []}

    # Store history for the last fold
    last_fold_history = {'train_loss': [], 'val_loss': []}
    last_fold_preds = None
    last_fold_y_val = None

    fold = 1
    for train_idx, val_idx in kf.split(X):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # Feature Selection
        selector = SelectKBest(score_func=f_regression, k=k_features)
        X_train_sel = selector.fit_transform(X_train, y_train[:, 0])
        X_val_sel = selector.transform(X_val)

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_sel)
        X_val_scaled = scaler.transform(X_val_sel)

        # Prepare Tensors
        tensors = [torch.tensor(X_train_scaled, dtype=torch.float32)]
        for i in range(len(target_names)):
            tensors.append(torch.tensor(y_train[:, i:i + 1], dtype=torch.float32))

        train_dataset = TensorDataset(*tensors)
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

        # Init Model
        model = MultiTaskDNN(input_dim=k_features, target_names=target_names).to(DEVICE)
        optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
        criterion = nn.MSELoss()

        fold_train_loss = []
        fold_val_loss = []

        for epoch in range(100):
            t_loss = train_one_epoch(model, train_loader, optimizer, criterion, DEVICE)
            fold_train_loss.append(t_loss)

            # Validation pass
            model.eval()
            with torch.no_grad():
                X_val_tensor = torch.tensor(X_val_scaled, dtype=torch.float32).to(DEVICE)
                val_preds = model(X_val_tensor)
                v_loss = 0
                for i, name in enumerate(target_names):
                    target_tensor = torch.tensor(y_val[:, i], dtype=torch.float32).view(-1, 1).to(DEVICE)
                    v_loss += criterion(val_preds[name], target_tensor).item()
                fold_val_loss.append(v_loss)

        if fold == 5:
            last_fold_history['train_loss'] = fold_train_loss
            last_fold_history['val_loss'] = fold_val_loss

            model.eval()
            with torch.no_grad():
                preds_dict = model(torch.tensor(X_val_scaled, dtype=torch.float32).to(DEVICE))

            last_fold_preds = np.zeros_like(y_val)
            for i, name in enumerate(target_names):
                last_fold_preds[:, i] = preds_dict[name].cpu().numpy().flatten()
            last_fold_y_val = y_val

        # Evaluation metrics
        model.eval()
        with torch.no_grad():
            preds_dict = model(torch.tensor(X_val_scaled, dtype=torch.float32).to(DEVICE))

        fold_mses = []
        fold_r2s = []
        for i, name in enumerate(target_names):
            y_true = y_val[:, i]
            y_pred = preds_dict[name].cpu().numpy().flatten()
            fold_mses.append(mean_squared_error(y_true, y_pred))
            fold_r2s.append(r2_score(y_true, y_pred))

        fold_results['MSE'].append(np.mean(fold_mses))
        fold_results['R2'].append(np.mean(fold_r2s))

        print(f"  Fold {fold}: Avg MSE={np.mean(fold_mses):.4f}, Avg R2={np.mean(fold_r2s):.4f}")
        fold += 1

    plot_training_history(last_fold_history, dataset_name)
    plot_predictions(last_fold_y_val, last_fold_preds, target_names, dataset_name)

    avg_mse = np.mean(fold_results['MSE'])
    avg_r2 = np.mean(fold_results['R2'])

    return [{
        'Dataset': dataset_name,
        'Model': 'Multi-Task DNN',
        'Features': k_features,
        'MSE': avg_mse,
        'R2': avg_r2
    }]


def create_combined_visualizations(all_results):
    if not all_results: return

    df_res = pd.DataFrame(all_results)

    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    sns.barplot(data=df_res, x='Dataset', y='MSE', hue='Model', palette='viridis')
    plt.title('DNN Implementation Average MSE (Lower is Better)')
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    sns.barplot(data=df_res, x='Dataset', y='R2', hue='Model', palette='viridis')
    plt.title('DNN Implementation Average R2 Score (Higher is Better)')
    plt.ylim(0, 1)
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('DNN_implementation_comparison.png', dpi=300)
    print("\nSaved plot: DNN_implementation_comparison.png")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    print("\n" + "=" * 80)
    print(" " * 25 + "MULTI-TASK DNN ANALYSIS")
    print(" " * 20 + "with 5-Fold Cross-Validation")
    print("=" * 80)

    datasets = [
        ('IADSED-2.csv', 'IADSED-2'),
        ('EmoSounds-3.csv', 'EmoSounds-3')
    ]

    all_results = []

    for filepath, dataset_name in datasets:
        results = process_dataset(filepath, dataset_name)
        all_results.extend(results)

    if all_results:
        print("\n" + "=" * 80)
        print("FINAL RESULTS SUMMARY")
        print("=" * 80)
        df = pd.DataFrame(all_results)
        print(df.to_string(index=False))

        create_combined_visualizations(all_results)

    print("\n" + "=" * 80)
    print("Analysis Complete.")
    print("=" * 80)


if __name__ == "__main__":
    main()
