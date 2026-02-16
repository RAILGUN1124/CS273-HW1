#XGBoost Regression Analysis with 5-Fold CV and Feature Selection

import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.feature_selection import SelectKBest, f_regression
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from ridge_regression_analysis import (
    split_data,
    evaluate_on_test,
)

sns.set_style('whitegrid')
plt.rcParams['font.size'] = 9

import xgboost as xgb

# Store results for final report
all_experiment_results = []

# Early-stopping patience
EARLY_STOPPING_ROUNDS = 20


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def load_and_prepare_data(filepath, dataset_name):
    """Load preprocessed data and separate features from targets"""
    df = pd.read_csv(filepath)

    possible_targets = ['arousal', 'valence', 'dominance']
    emotion_labels = [t for t in possible_targets if t in df.columns]

    exclude_cols = list(emotion_labels)
    for col in ['dataset', 'source', 'genre', 'category', 'fnames', 'fname', 'splits',
                'vocals', 'BE_Classification', 'description', 'emotional_intensity', 'emotional_quadrant']:
        if col in df.columns:
            exclude_cols.append(col)

    feature_cols = [col for col in df.columns
                   if col not in exclude_cols and df[col].dtype in ['float64', 'int64']]

    print(f"\n{'='*80}")
    print(f"Dataset: {dataset_name}")
    print(f"{'='*80}")
    print(f"Total samples: {len(df)}")
    print(f"Number of features: {len(feature_cols)}")
    print(f"Target variables: {emotion_labels}")

    return df, feature_cols, emotion_labels


def get_hyperparameter_grid():
    """XGBoost hyperparameter search space for 5-Fold Cross Validation"""
    from itertools import product

    keys = [
        'max_depth', 'learning_rate', 'n_estimators', 'min_child_weight',
        'subsample', 'colsample_bytree', 'reg_alpha', 'reg_lambda',
    ]
    values = [
        [3, 4, 5],
        [0.01, 0.05, 0.1],
        [300],
        [3, 5],
        [0.7, 0.8],
        [0.7, 0.8],
        [0.1, 1.0, 10.0],
        [1.0, 10.0, 50.0],
    ]
    return [dict(zip(keys, p)) for p in product(*values)]


def hyperparameter_tuning_with_cv(X_train, y_train, X_val, y_val):
    """5-Fold Cross Validation for Hyperparameter Tuning"""
    print(f"\n 5-Fold Cross Validation for Hyperparameter Tuning")

    X_train_val = np.vstack([X_train, X_val])
    y_train_val = np.concatenate([y_train, y_val])
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    param_combos = get_hyperparameter_grid()
    print(f"Searching {len(param_combos)} hyperparameter combinations")

    best_avg_rmse = np.inf
    best_params = None
    cv_results = {}

    for params in param_combos:
        fold_scores = []
        fold_best_iters = []
        for train_idx, val_idx in kf.split(X_train_val):
            X_fold_train = X_train_val[train_idx]
            X_fold_val = X_train_val[val_idx]
            y_fold_train = y_train_val[train_idx]
            y_fold_val = y_train_val[val_idx]

            model = xgb.XGBRegressor(
                max_depth=params['max_depth'],
                learning_rate=params['learning_rate'],
                n_estimators=params['n_estimators'],
                min_child_weight=params['min_child_weight'],
                subsample=params['subsample'],
                colsample_bytree=params['colsample_bytree'],
                reg_alpha=params['reg_alpha'],
                reg_lambda=params['reg_lambda'],
                early_stopping_rounds=EARLY_STOPPING_ROUNDS,
                random_state=42,
                verbosity=0,
            )
            model.fit(
                X_fold_train, y_fold_train,
                eval_set=[(X_fold_val, y_fold_val)],
                verbose=False,
            )
            y_pred = model.predict(X_fold_val)
            rmse = np.sqrt(mean_squared_error(y_fold_val, y_pred))
            fold_scores.append(rmse)
            fold_best_iters.append(
                getattr(model, 'best_iteration', params['n_estimators'])
            )

        avg_rmse = np.mean(fold_scores)
        std_rmse = np.std(fold_scores)
        avg_best_iter = int(np.mean(fold_best_iters))
        cv_results[str(params)] = {
            'avg_rmse': avg_rmse, 'std_rmse': std_rmse,
            'fold_scores': fold_scores, 'avg_best_iter': avg_best_iter,
        }
        if avg_rmse < best_avg_rmse:
            best_avg_rmse = avg_rmse
            best_params = params.copy()
            best_params['_avg_best_iter'] = avg_best_iter

    print(f"  Best CV RMSE: {best_avg_rmse:.4f}")
    print(f"  Best params: { {pk: pv for pk, pv in best_params.items() if pk != '_avg_best_iter'} }")
    print(f"  Avg best iteration (early stop): {best_params.get('_avg_best_iter', 'N/A')}")
    return best_params, cv_results


def train_final_model(X_train, y_train, X_val, y_val, best_params):
    """Step 3: Retrain final model with best hyperparameters"""
    print(f"\n  Step 3: Retraining final model with best hyperparameters")

    X_train_val = np.vstack([X_train, X_val])
    y_train_val = np.concatenate([y_train, y_val])
    effective_n = best_params.get('_avg_best_iter', best_params['n_estimators'])
    effective_n = max(int(effective_n) + 10, 50)

    model = xgb.XGBRegressor(
        max_depth=best_params['max_depth'],
        learning_rate=best_params['learning_rate'],
        n_estimators=effective_n,
        min_child_weight=best_params['min_child_weight'],
        subsample=best_params['subsample'],
        colsample_bytree=best_params['colsample_bytree'],
        reg_alpha=best_params['reg_alpha'],
        reg_lambda=best_params['reg_lambda'],
        random_state=42,
        verbosity=0,
    )
    model.fit(X_train_val, y_train_val)

    y_train_pred = model.predict(X_train_val)
    train_rmse = np.sqrt(mean_squared_error(y_train_val, y_train_pred))
    train_mse = mean_squared_error(y_train_val, y_train_pred)
    train_r2 = r2_score(y_train_val, y_train_pred)

    print(f"  Effective n_estimators: {effective_n}")
    print(f"  Training RMSE: {train_rmse:.4f}")
    print(f"  Training MSE:  {train_mse:.4f}")
    print(f"  Training R²:   {train_r2:.4f}")
    return model, train_rmse, train_mse, train_r2, X_train_val, y_train_val


# ============================================================================
# FEATURE SELECTION
# ============================================================================

def apply_feature_selection_importance(X_train, y_train, X_val, X_test, k=34):
    """Feature selection by XGBoost importance top k features"""
    k = min(k, X_train.shape[1])
    print(f"\n  Feature Selection: XGBoost Importance-based (top {k})")

    preliminary = xgb.XGBRegressor(
        max_depth=3, learning_rate=0.1, n_estimators=100,
        min_child_weight=5, subsample=0.8, colsample_bytree=0.8,
        reg_alpha=1.0, reg_lambda=10.0, random_state=42, verbosity=0,
    )
    preliminary.fit(X_train, y_train)
    importances = preliminary.feature_importances_
    top_indices = np.sort(np.argsort(importances)[::-1][:k])

    X_train_fs = X_train[:, top_indices]
    X_val_fs = X_val[:, top_indices]
    X_test_fs = X_test[:, top_indices]
    print(f"  Selected {len(top_indices)} features")
    return X_train_fs, X_val_fs, X_test_fs, top_indices, preliminary


def apply_feature_selection_selectkbest(X_train, y_train, X_val, X_test, k=34):
    """Feature selection with SelectKBest"""
    k = min(k, X_train.shape[1])
    print(f"\n  Feature Selection: SelectKBest f_regression (k={k})")
    selector = SelectKBest(f_regression, k=k)
    selector.fit(X_train, y_train)
    X_train_fs = selector.transform(X_train)
    X_val_fs = selector.transform(X_val)
    X_test_fs = selector.transform(X_test)
    selected_indices = selector.get_support(indices=True)
    print(f"  Selected {len(selected_indices)} features")
    return X_train_fs, X_val_fs, X_test_fs, selected_indices, selector


def train_and_evaluate_xgb(X_train, y_train, X_val, y_val, X_test, y_test,
                            target_name, use_feature_selection=False,
                            fs_method='importance', k=34,
                            feature_names=None, dataset_name='Dataset'):
    print(f"\n{'-'*80}")
    if use_feature_selection:
        print(f"XGBOOST WITH FEATURE SELECTION ({fs_method}) - {target_name.upper()}")
    else:
        print(f"XGBOOST WITHOUT FEATURE SELECTION - {target_name.upper()}")
    print(f"{'-'*80}")

    if use_feature_selection:
        if fs_method == 'importance':
            X_train_p, X_val_p, X_test_p, selected_idx, selector = \
                apply_feature_selection_importance(X_train, y_train, X_val, X_test, k=k)
        else:
            X_train_p, X_val_p, X_test_p, selected_idx, selector = \
                apply_feature_selection_selectkbest(X_train, y_train, X_val, X_test, k=k)
        n_features = X_train_p.shape[1]
    else:
        X_train_p, X_val_p, X_test_p = X_train, X_val, X_test
        selector = None
        selected_idx = None
        n_features = X_train_p.shape[1]

    best_params, cv_results = hyperparameter_tuning_with_cv(
        X_train_p, y_train, X_val_p, y_val
    )
    final_model, train_rmse, train_mse, train_r2, X_train_val, y_train_val = train_final_model(
        X_train_p, y_train, X_val_p, y_val, best_params
    )
    test_rmse, test_mse, test_r2, y_test_pred = evaluate_on_test(final_model, X_test_p, y_test)
    y_train_pred = final_model.predict(X_train_val)

    overfit_ratio = train_rmse / test_rmse if test_rmse > 0 else float('inf')
    print(f"\n  Overfit check – Train/Test RMSE ratio: {overfit_ratio:.3f}  "
          f"({'OK' if overfit_ratio > 0.5 else 'SEVERE OVERFIT'})")

    results = {
        'dataset_name': dataset_name,
        'target': target_name,
        'feature_selection': use_feature_selection,
        'fs_method': fs_method if use_feature_selection else 'none',
        'n_features': n_features,
        'best_params': {pk: pv for pk, pv in best_params.items() if pk != '_avg_best_iter'},
        'cv_results': cv_results,
        'model': final_model,
        'train_rmse': train_rmse,
        'train_mse': train_mse,
        'train_r2': train_r2,
        'test_rmse': test_rmse,
        'test_mse': test_mse,
        'test_r2': test_r2,
        'y_train_val': y_train_val,
        'y_train_pred': y_train_pred,
        'y_test': y_test,
        'y_test_pred': y_test_pred,
        'selector': selector,
        'selected_indices': selected_idx,
        'feature_names': feature_names,
    }
    all_experiment_results.append(results)
    return results


def compare_results(results_no_fs, results_with_fs, dataset_name='Dataset'):
    """Compare performance before and after feature selection"""
    print(f"\n{'='*80}")
    print(f"PERFORMANCE COMPARISON: BEFORE vs AFTER FEATURE SELECTION (XGBoost)")
    print(f"{'='*80}")
    print(f"\nTarget: {results_no_fs['target'].upper()}")
    print(f"\n{'Metric':<20} {'Without FS':<20} {'With FS':<20} {'Change':<15}")
    print(f"{'-'*75}")

    n_before = results_no_fs['n_features']
    n_after = results_with_fs['n_features']
    feat_pct = ((n_before - n_after) / n_before) * 100 if n_before else 0
    print(f"{'# Features':<20} {n_before:<20} {n_after:<20} {f'-{feat_pct:.1f}%':<15}")

    for metric, key in [
        ('Train RMSE', 'train_rmse'),
        ('Train MSE', 'train_mse'),
        ('Train R²', 'train_r2'),
        ('Test RMSE', 'test_rmse'),
        ('Test MSE', 'test_mse'),
        ('Test R²', 'test_r2'),
    ]:
        v1 = results_no_fs[key]
        v2 = results_with_fs[key]
        if 'R²' in metric or 'r2' in key:
            ch = f"{v2 - v1:+.4f}"
        else:
            pct = ((v2 - v1) / v1) * 100 if v1 != 0 else 0
            ch = f"{pct:+.2f}%"
        print(f"{metric:<20} {v1:<20.4f} {v2:<20.4f} {ch:<15}")

    print(f"\nSummary: Feature selection reduced features from {n_before} to {n_after}.")


def process_dataset(filepath, dataset_name, k_features=34, fs_method='importance'):
    """Process entire dataset: load, split, train with/without FS, compare"""
    df, feature_cols, emotion_labels = load_and_prepare_data(filepath, dataset_name)
    all_results = []

    for target in emotion_labels:
        print(f"\n\n{'#'*80}")
        print(f"# TARGET: {target.upper()}")
        print(f"{'#'*80}")

        X = df[feature_cols].values
        y = df[target].values
        X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)

        results_no_fs = train_and_evaluate_xgb(
            X_train, y_train, X_val, y_val, X_test, y_test,
            target_name=target,
            use_feature_selection=False,
            feature_names=feature_cols,
            dataset_name=dataset_name,
        )
        all_results.append(results_no_fs)

        results_with_fs = train_and_evaluate_xgb(
            X_train, y_train, X_val, y_val, X_test, y_test,
            target_name=target,
            use_feature_selection=True,
            fs_method=fs_method,
            k=k_features,
            feature_names=feature_cols,
            dataset_name=dataset_name,
        )
        all_results.append(results_with_fs)
        compare_results(results_no_fs, results_with_fs, dataset_name)

    return all_results


def print_final_summary(all_results):
    """Print final summary table for all experiments"""
    print(f"\n\n{'='*80}")
    print(f"FINAL SUMMARY - ALL RESULTS")
    print(f"{'='*80}")
    print(f"\n{'Target':<15} {'Features':<12} {'Best Alpha':<12} {'Train RMSE':<12} {'Test RMSE':<12} {'Test R²':<12}")
    print(f"{'-'*85}")

    for r in all_results:
        fs_label = r.get('fs_method', 'none')
        if fs_label == 'none':
            fs_label = 'No'
        print(f"{r['dataset_name']:<14} {r['target']:<10} {fs_label:<12} {r['n_features']:<6} "
              f"{r['train_rmse']:<12.4f} {r['train_mse']:<12.4f} "
              f"{r['test_rmse']:<12.4f} {r['test_mse']:<12.4f} {r['test_r2']:<10.4f}")
    print()


# ============================================================================
# COMBINED VISUALIZATION FUNCTIONS
# ============================================================================

def create_combined_visualizations():
    """Create comprehensive combined visualizations"""
    print(f"\n\n{'='*80}")
    print("GENERATING COMBINED VISUALIZATIONS")
    print(f"{'='*80}\n")

    datasets = {}
    for result in all_experiment_results:
        dataset_name = result['dataset_name']
        if dataset_name not in datasets:
            datasets[dataset_name] = {}
        target = result['target']
        if target not in datasets[dataset_name]:
            datasets[dataset_name][target] = {'no_fs': None, 'fs': None}
        key = 'fs' if result['feature_selection'] else 'no_fs'
        datasets[dataset_name][target][key] = result

    for dataset_name, targets_data in datasets.items():
        create_dataset_comprehensive_figure(dataset_name, targets_data)
    create_overall_summary_figure(datasets)

    print(f"\nAll visualizations generated successfully!")
    print(f"  Total files created: {len(datasets) * 1 + 1}")


def create_dataset_comprehensive_figure(dataset_name, targets_data):
    """Create ONE comprehensive figure per dataset with all analyses"""
    targets = list(targets_data.keys())
    n_targets = len(targets)

    fig = plt.figure(figsize=(24, 5 * n_targets))
    gs = fig.add_gridspec(n_targets, 6, hspace=0.3, wspace=0.3)

    for i, target in enumerate(targets):
        data_no_fs = targets_data[target]['no_fs']
        data_fs = targets_data[target]['fs']
        if data_no_fs is None or data_fs is None:
            continue


        # Column 0: CV Results Comparison
        ax_cv = fig.add_subplot(gs[i, 0])
        for data, label, color in [(data_no_fs, f'No FS (68)', 'steelblue'),
                                   (data_fs, f'FS (34)', 'coral')]:
            cv_res = data['cv_results']
            avg_rmses = sorted([cv_res[key]['avg_rmse'] for key in cv_res])
            ax_cv.plot(range(len(avg_rmses)), avg_rmses, label=label, color=color, linewidth=1.5, alpha=0.8)
        ax_cv.set_xlabel('Config rank (sorted)', fontsize=9)
        ax_cv.set_ylabel('CV RMSE', fontsize=9)
        ax_cv.set_title(f'{target.upper()}\nCV RMSE Distribution', fontsize=10, fontweight='bold')
        ax_cv.legend(fontsize=7)
        ax_cv.grid(True, alpha=0.3)

        # Column 1: Test Predictions Without FS
        ax_pred1 = fig.add_subplot(gs[i, 1])
        y_test, y_pred = data_no_fs['y_test'], data_no_fs['y_test_pred']
        ax_pred1.scatter(y_test, y_pred, alpha=0.5, s=25, edgecolors='black', linewidth=0.2)
        min_val = min(y_test.min(), y_pred.min())
        max_val = max(y_test.max(), y_pred.max())
        ax_pred1.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=1.5)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        ax_pred1.set_xlabel('Actual', fontsize=8)
        ax_pred1.set_ylabel('Predicted', fontsize=8)
        ax_pred1.set_title(f'Test (No FS)\nRMSE={rmse:.3f} R²={r2:.3f}', fontsize=9, fontweight='bold')
        ax_pred1.grid(True, alpha=0.3)

        # Column 2: Test Predictions With FS
        ax_pred2 = fig.add_subplot(gs[i, 2])
        y_test, y_pred = data_fs['y_test'], data_fs['y_test_pred']
        ax_pred2.scatter(y_test, y_pred, alpha=0.5, s=25, edgecolors='black', linewidth=0.2, color='coral')
        min_val = min(y_test.min(), y_pred.min())
        max_val = max(y_test.max(), y_pred.max())
        ax_pred2.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=1.5)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        ax_pred2.set_xlabel('Actual', fontsize=8)
        ax_pred2.set_ylabel('Predicted', fontsize=8)
        ax_pred2.set_title(f'Test (FS)\nRMSE={rmse:.3f} R²={r2:.3f}', fontsize=9, fontweight='bold')
        ax_pred2.grid(True, alpha=0.3)

        # Column 3: Residuals Without FS
        ax_res1 = fig.add_subplot(gs[i, 3])
        y_test, y_pred = data_no_fs['y_test'], data_no_fs['y_test_pred']
        residuals = y_test - y_pred
        ax_res1.scatter(y_pred, residuals, alpha=0.5, s=25, edgecolors='black', linewidth=0.2)
        ax_res1.axhline(y=0, color='r', linestyle='--', linewidth=1.5)
        ax_res1.set_xlabel('Predicted', fontsize=8)
        ax_res1.set_ylabel('Residuals', fontsize=8)
        ax_res1.set_title(f'Residuals (No FS)\nMean={residuals.mean():.3f}', fontsize=9, fontweight='bold')
        ax_res1.grid(True, alpha=0.3)

        # Column 4: Residuals With FS
        ax_res2 = fig.add_subplot(gs[i, 4])
        y_test, y_pred = data_fs['y_test'], data_fs['y_test_pred']
        residuals = y_test - y_pred
        ax_res2.scatter(y_pred, residuals, alpha=0.5, s=25, edgecolors='black', linewidth=0.2, color='coral')
        ax_res2.axhline(y=0, color='r', linestyle='--', linewidth=1.5)
        ax_res2.set_xlabel('Predicted', fontsize=8)
        ax_res2.set_ylabel('Residuals', fontsize=8)
        ax_res2.set_title(f'Residuals (FS)\nMean={residuals.mean():.3f}', fontsize=9, fontweight='bold')
        ax_res2.grid(True, alpha=0.3)

        # Column 5: Feature Importance
        ax_feat = fig.add_subplot(gs[i, 5])
        if data_no_fs.get('model') is not None and hasattr(data_no_fs['model'], 'feature_importances_'):
            imp = data_no_fs['model'].feature_importances_
            fnames = data_no_fs['feature_names']
            if fnames is not None and len(fnames) == len(imp):
                top_idx = np.argsort(imp)[-10:][::-1]
                top_features = [fnames[j] for j in top_idx]
                top_imp = imp[top_idx]
                y_pos = np.arange(len(top_features))
                colors = sns.color_palette('viridis', len(top_features))
                ax_feat.barh(y_pos, top_imp, color=colors, edgecolor='black', linewidth=0.5)
                ax_feat.set_yticks(y_pos)
                ax_feat.set_yticklabels(top_features, fontsize=7)
                ax_feat.set_xlabel('Importance', fontsize=8)
                ax_feat.set_title(f'Top 10 Features (No FS)', fontsize=9, fontweight='bold')
                ax_feat.invert_yaxis()
                ax_feat.grid(True, alpha=0.3, axis='x')
            else:
                ax_feat.text(0.5, 0.5, 'N/A', ha='center', va='center', transform=ax_feat.transAxes)
                ax_feat.set_title('Top 10 Features (No FS)', fontsize=9, fontweight='bold')
        else:
            ax_feat.text(0.5, 0.5, 'N/A', ha='center', va='center', transform=ax_feat.transAxes)
            ax_feat.set_title('Top 10 Features (No FS)', fontsize=9, fontweight='bold')

    fig.suptitle(f'Comprehensive XGBoost Analysis: {dataset_name}', fontsize=16, fontweight='bold', y=0.995)
    filename = f'{dataset_name.lower().replace("-", "")}_full_analysis.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {filename}")


def create_overall_summary_figure(datasets):
    """Create ONE overall summary figure comparing all datasets and targets"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    all_labels = []
    test_rmse_no_fs = []
    test_rmse_fs = []
    test_r2_no_fs = []
    test_r2_fs = []
    train_rmse_no_fs = []
    train_rmse_fs = []

    for dataset_name, targets_data in datasets.items():
        for target, data in targets_data.items():
            if data['no_fs'] and data['fs']:
                all_labels.append(f"{dataset_name[:8]}\n{target}")
                test_rmse_no_fs.append(data['no_fs']['test_rmse'])
                test_rmse_fs.append(data['fs']['test_rmse'])
                test_r2_no_fs.append(data['no_fs']['test_r2'])
                test_r2_fs.append(data['fs']['test_r2'])
                train_rmse_no_fs.append(data['no_fs']['train_rmse'])
                train_rmse_fs.append(data['fs']['train_rmse'])

    x = np.arange(len(all_labels))
    width = 0.35

    axes[0, 0].bar(x - width/2, test_rmse_no_fs, width, label='No FS', color='steelblue', edgecolor='black')
    axes[0, 0].bar(x + width/2, test_rmse_fs, width, label='FS', color='coral', edgecolor='black')
    axes[0, 0].set_ylabel('Test RMSE', fontsize=11)
    axes[0, 0].set_title('Test RMSE Comparison', fontsize=12, fontweight='bold')
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(all_labels, fontsize=8)
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3, axis='y')

    axes[0, 1].bar(x - width/2, test_r2_no_fs, width, label='No FS', color='steelblue', edgecolor='black')
    axes[0, 1].bar(x + width/2, test_r2_fs, width, label='FS', color='coral', edgecolor='black')
    axes[0, 1].set_ylabel('Test R²', fontsize=11)
    axes[0, 1].set_title('Test R² Comparison', fontsize=12, fontweight='bold')
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels(all_labels, fontsize=8)
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3, axis='y')

    axes[1, 0].bar(x - width/2, train_rmse_no_fs, width, label='No FS', color='steelblue', edgecolor='black')
    axes[1, 0].bar(x + width/2, train_rmse_fs, width, label='FS', color='coral', edgecolor='black')
    axes[1, 0].set_ylabel('Train RMSE', fontsize=11)
    axes[1, 0].set_title('Train RMSE Comparison', fontsize=12, fontweight='bold')
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(all_labels, fontsize=8)
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3, axis='y')

    improvements = [(fs - no_fs) / no_fs * 100 for fs, no_fs in zip(test_rmse_fs, test_rmse_no_fs)]
    colors_imp = ['green' if imp < 0 else 'red' for imp in improvements]
    axes[1, 1].bar(x, improvements, color=colors_imp, edgecolor='black')
    axes[1, 1].axhline(y=0, color='black', linestyle='-', linewidth=1)
    axes[1, 1].set_ylabel('RMSE Change (%)', fontsize=11)
    axes[1, 1].set_title('Feature Selection Impact on Test RMSE\n(Negative = Improvement)', fontsize=12, fontweight='bold')
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(all_labels, fontsize=8)
    axes[1, 1].grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    filename = 'xgboost_v2_overall_model_comparison.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {filename}")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    print("\n" + "="*80)
    print(" "*15 + "XGBoost Regression Analysis")
    print(" "*10 + "5-Fold CV, Early Stopping, Feature Selection")
    print("="*80)

    datasets = [
        ('emosounds-3_preprocessed.csv', 'EmoSounds-3'),
        ('iadsed-2_preprocessed.csv', 'IADSED-2'),
    ]
    k_features = 34
    fs_method = 'importance'
    all_results = []

    for filepath, dataset_name in datasets:
        try:
            results = process_dataset(filepath, dataset_name, k_features, fs_method)
            all_results.extend(results)
        except FileNotFoundError:
            print(f"\nError: File '{filepath}' not found. Skipping {dataset_name}.")
        except Exception as e:
            print(f"\nError processing {dataset_name}: {str(e)}")
            raise

    if all_results:
        print_final_summary(all_results)
        create_combined_visualizations()

    print("\n" + "="*80)
    print(" "*25 + "ANALYSIS COMPLETE")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
