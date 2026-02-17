"""
Ridge Regression Analysis with Combined Visualizations
Implements 5-fold CV for hyperparameter tuning and creates minimal comprehensive plots
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold
from sklearn.linear_model import Ridge
from sklearn.feature_selection import SelectKBest, f_regression, RFE
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Set plotting style
sns.set_style('whitegrid')
plt.rcParams['font.size'] = 9

# Store all results for batch plotting
all_experiment_results = []

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def load_and_prepare_data(filepath, dataset_name):
    """Load preprocessed data and separate features from targets"""
    df = pd.read_csv(filepath)
    
    # Define target variables
    emotion_labels = ['arousal', 'valence', 'dominance']
    
    # Define columns to exclude (non-numeric or non-feature columns)
    exclude_cols = emotion_labels + ['dataset', 'genre', 'category', 'fnames', 'splits', 
                                      'vocals', 'emotional_intensity', 'emotional_quadrant']
    
    # Get feature columns (only numeric features, exclude targets and categorical columns)
    feature_cols = [col for col in df.columns 
                   if col not in exclude_cols and df[col].dtype in ['float64', 'int64']]
    
    print(f"\n{'='*80}")
    print(f"Dataset: {dataset_name}")
    print(f"{'='*80}")
    print(f"Total samples: {len(df)}")
    print(f"Number of features: {len(feature_cols)}")
    print(f"Target variables: {emotion_labels}")
    
    return df, feature_cols, emotion_labels


def split_data(X, y, test_size=0.2, val_size=0.2, random_state=42):
    """Split data into Train, Validation, and Test sets (60:20:20)"""
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    val_ratio = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_ratio, random_state=random_state
    )
    
    print(f"\nData Split (60:20:20):")
    print(f"  Training set: {len(X_train)} samples ({len(X_train)/len(X)*100:.1f}%)")
    print(f"  Validation set: {len(X_val)} samples ({len(X_val)/len(X)*100:.1f}%)")
    print(f"  Test set: {len(X_test)} samples ({len(X_test)/len(X)*100:.1f}%)")
    
    return X_train, X_val, X_test, y_train, y_val, y_test


def hyperparameter_tuning_with_cv(X_train, y_train, X_val, y_val, alpha_values):
    """5-Fold Cross-Validation for Hyperparameter Tuning"""
    print(f"\n  Step 1-2: 5-Fold Cross-Validation for Hyperparameter Tuning")
    print(f"  Testing alpha values: {alpha_values}")
    
    X_train_val = np.vstack([X_train, X_val])
    y_train_val = np.concatenate([y_train, y_val])
    
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_results = {}
    
    for alpha in alpha_values:
        fold_scores = []
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(X_train_val), 1):
            X_fold_train = X_train_val[train_idx]
            X_fold_val = X_train_val[val_idx]
            y_fold_train = y_train_val[train_idx]
            y_fold_val = y_train_val[val_idx]
            
            model = Ridge(alpha=alpha)
            model.fit(X_fold_train, y_fold_train)
            
            y_pred = model.predict(X_fold_val)
            rmse = np.sqrt(mean_squared_error(y_fold_val, y_pred))
            fold_scores.append(rmse)
        
        avg_rmse = np.mean(fold_scores)
        std_rmse = np.std(fold_scores)
        cv_results[alpha] = {
            'avg_rmse': avg_rmse,
            'std_rmse': std_rmse,
            'fold_scores': fold_scores
        }
    
    best_alpha = min(cv_results, key=lambda x: cv_results[x]['avg_rmse'])
    
    print(f"\n  Cross-Validation Results:")
    for alpha in alpha_values:
        result = cv_results[alpha]
        print(f"    Alpha={alpha:7.4f}: Avg RMSE={result['avg_rmse']:.4f} (±{result['std_rmse']:.4f})")
    
    print(f"\n  Best Alpha: {best_alpha} (Avg RMSE: {cv_results[best_alpha]['avg_rmse']:.4f})")
    
    return best_alpha, cv_results


def train_final_model(X_train, y_train, X_val, y_val, best_alpha):
    """Retrain the best model on all train+validation data"""
    print(f"\n  Step 3: Retraining final model with best hyperparameters")
    
    X_train_val = np.vstack([X_train, X_val])
    y_train_val = np.concatenate([y_train, y_val])
    
    print(f"  Training on combined train+val: {len(X_train_val)} samples")
    
    final_model = Ridge(alpha=best_alpha)
    final_model.fit(X_train_val, y_train_val)
    
    y_train_pred = final_model.predict(X_train_val)
    train_rmse = np.sqrt(mean_squared_error(y_train_val, y_train_pred))
    train_r2 = r2_score(y_train_val, y_train_pred)
    
    print(f"  Training RMSE: {train_rmse:.4f}")
    print(f"  Training R²: {train_r2:.4f}")
    
    return final_model, train_rmse, train_r2, X_train_val, y_train_val


def evaluate_on_test(model, X_test, y_test):
    """Evaluate once on the test set"""
    print(f"\n  Step 4: Final Evaluation on Test Set")
    
    y_test_pred = model.predict(X_test)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    test_mse = mean_squared_error(y_test, y_test_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    
    print(f"  Test RMSE: {test_rmse:.4f}")
    print(f"  Test MSE: {test_mse:.4f}")
    print(f"  Test R²: {test_r2:.4f}")
    
    return test_rmse, test_mse, test_r2, y_test_pred


def apply_feature_selection(X_train, y_train, X_val, X_test, method='selectkbest', k=34):
    """Apply feature selection technique"""
    if method == 'selectkbest':
        print(f"\n  Feature Selection: SelectKBest (k={k} features)")
        selector = SelectKBest(f_regression, k=k)
        selector.fit(X_train, y_train)
        
        X_train_fs = selector.transform(X_train)
        X_val_fs = selector.transform(X_val)
        X_test_fs = selector.transform(X_test)
        
        selected_features = selector.get_support(indices=True)
        print(f"  Selected {len(selected_features)} features")
        
    elif method == 'rfe':
        print(f"\n  Feature Selection: RFE (k={k} features)")
        estimator = Ridge(alpha=1.0)
        selector = RFE(estimator, n_features_to_select=k, step=5)
        selector.fit(X_train, y_train)
        
        X_train_fs = selector.transform(X_train)
        X_val_fs = selector.transform(X_val)
        X_test_fs = selector.transform(X_test)
        
        selected_features = selector.get_support(indices=True)
        print(f"  Selected {len(selected_features)} features")
    
    return X_train_fs, X_val_fs, X_test_fs, selected_features, selector


def train_and_evaluate_ridge(X_train, y_train, X_val, y_val, X_test, y_test, 
                             target_name, use_feature_selection=False, fs_method='selectkbest', k=34,
                             feature_names=None, dataset_name='Dataset'):
    """Complete Ridge regression pipeline with CV"""
    print(f"\n{'-'*80}")
    if use_feature_selection:
        print(f"RIDGE REGRESSION WITH FEATURE SELECTION - {target_name.upper()}")
    else:
        print(f"RIDGE REGRESSION WITHOUT FEATURE SELECTION - {target_name.upper()}")
    print(f"{'-'*80}")
    
    # Apply feature selection if requested
    selector = None
    if use_feature_selection:
        X_train_proc, X_val_proc, X_test_proc, selected_features, selector = apply_feature_selection(
            X_train, y_train, X_val, X_test, method=fs_method, k=k
        )
        if feature_names is not None:
            selected_feature_names = [feature_names[i] for i in selected_features]
        else:
            selected_feature_names = None
    else:
        X_train_proc = X_train
        X_val_proc = X_val
        X_test_proc = X_test
        selected_features = None
        selected_feature_names = feature_names
    
    # Define hyperparameter search space
    alpha_values = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
    
    # Step 1-2: Hyperparameter tuning with CV
    best_alpha, cv_results = hyperparameter_tuning_with_cv(
        X_train_proc, y_train, X_val_proc, y_val, alpha_values
    )
    
    # Step 3: Train final model
    final_model, train_rmse, train_r2, X_train_val, y_train_val = train_final_model(
        X_train_proc, y_train, X_val_proc, y_val, best_alpha
    )
    
    # Step 4: Evaluate on test set
    test_rmse, test_mse, test_r2, y_test_pred = evaluate_on_test(final_model, X_test_proc, y_test)
    
    # Get training predictions
    y_train_pred = final_model.predict(X_train_val)
    
    results = {
        'dataset_name': dataset_name,
        'target': target_name,
        'feature_selection': use_feature_selection,
        'n_features': X_train_proc.shape[1],
        'selected_features': selected_features,
        'best_alpha': best_alpha,
        'train_rmse': train_rmse,
        'train_r2': train_r2,
        'test_rmse': test_rmse,
        'test_mse': test_mse,
        'test_r2': test_r2,
        'cv_results': cv_results,
        'model': final_model,
        'y_train_val': y_train_val,
        'y_train_pred': y_train_pred,
        'y_test': y_test,
        'y_test_pred': y_test_pred,
        'feature_names': selected_feature_names,
        'selector': selector
    }
    
    all_experiment_results.append(results)
    
    return results


def compare_results(results_no_fs, results_with_fs, dataset_name='Dataset'):
    """Compare performance before and after feature selection"""
    print(f"\n{'='*80}")
    print(f"PERFORMANCE COMPARISON: BEFORE vs AFTER FEATURE SELECTION")
    print(f"{'='*80}")
    
    print(f"\nTarget: {results_no_fs['target'].upper()}")
    print(f"\n{'Metric':<20} {'Without FS':<20} {'With FS':<20} {'Improvement':<15}")
    print(f"{'-'*75}")
    
    # Number of features
    n_feat_before = results_no_fs['n_features']
    n_feat_after = results_with_fs['n_features']
    feat_reduction = ((n_feat_before - n_feat_after) / n_feat_before) * 100
    print(f"{'Features':<20} {n_feat_before:<20} {n_feat_after:<20} {f'-{feat_reduction:.1f}%':<15}")
    
    # Best alpha
    print(f"{'Best Alpha':<20} {results_no_fs['best_alpha']:<20} {results_with_fs['best_alpha']:<20} {'-':<15}")
    
    # Training RMSE
    train_rmse_diff = results_with_fs['train_rmse'] - results_no_fs['train_rmse']
    train_rmse_pct = (train_rmse_diff / results_no_fs['train_rmse']) * 100
    train_improvement = f"{train_rmse_pct:+.2f}%" if train_rmse_diff != 0 else "Same"
    print(f"{'Training RMSE':<20} {results_no_fs['train_rmse']:<20.4f} {results_with_fs['train_rmse']:<20.4f} {train_improvement:<15}")
    
    # Training R²
    train_r2_diff = results_with_fs['train_r2'] - results_no_fs['train_r2']
    train_r2_improvement = f"{train_r2_diff:+.4f}"
    print(f"{'Training R²':<20} {results_no_fs['train_r2']:<20.4f} {results_with_fs['train_r2']:<20.4f} {train_r2_improvement:<15}")
    
    # Test RMSE
    test_rmse_diff = results_with_fs['test_rmse'] - results_no_fs['test_rmse']
    test_rmse_pct = (test_rmse_diff / results_no_fs['test_rmse']) * 100
    test_improvement = f"{test_rmse_pct:+.2f}%" if test_rmse_diff != 0 else "Same"
    print(f"{'Test RMSE':<20} {results_no_fs['test_rmse']:<20.4f} {results_with_fs['test_rmse']:<20.4f} {test_improvement:<15}")
    
    # Test MSE
    test_mse_diff = results_with_fs['test_mse'] - results_no_fs['test_mse']
    test_mse_pct = (test_mse_diff / results_no_fs['test_mse']) * 100
    mse_improvement = f"{test_mse_pct:+.2f}%" if test_mse_diff != 0 else "Same"
    print(f"{'Test MSE':<20} {results_no_fs['test_mse']:<20.4f} {results_with_fs['test_mse']:<20.4f} {mse_improvement:<15}")
    
    # Test R²
    test_r2_diff = results_with_fs['test_r2'] - results_no_fs['test_r2']
    test_r2_improvement = f"{test_r2_diff:+.4f}"
    print(f"{'Test R²':<20} {results_no_fs['test_r2']:<20.4f} {results_with_fs['test_r2']:<20.4f} {test_r2_improvement:<15}")
    
    print(f"\n{'Summary':<20}")
    print(f"{'-'*75}")
    if test_rmse_diff < 0:
        print(f"Feature selection IMPROVED test RMSE by {abs(test_rmse_pct):.2f}%")
    elif test_rmse_diff > 0:
        print(f"Feature selection WORSENED test RMSE by {test_rmse_pct:.2f}%")
    else:
        print(f"Feature selection had NO EFFECT on test RMSE")
    
    print(f"  Features reduced from {n_feat_before} to {n_feat_after} ({feat_reduction:.1f}% reduction)")


def process_dataset(filepath, dataset_name, targets=['arousal', 'valence', 'dominance'], k=34):
    """Process entire dataset: load, split, train with/without FS, compare"""
    df, feature_cols, emotion_labels = load_and_prepare_data(filepath, dataset_name)
    
    all_results = []
    
    for target in targets:
        if target not in emotion_labels:
            print(f"\nWarning: Target '{target}' not found in dataset. Skipping.")
            continue
        
        print(f"\n\n{'#'*80}")
        print(f"# TARGET: {target.upper()}")
        print(f"{'#'*80}")
        
        X = df[feature_cols].values
        y = df[target].values
        
        X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)
        
        # Train WITHOUT feature selection
        results_no_fs = train_and_evaluate_ridge(
            X_train, y_train, X_val, y_val, X_test, y_test,
            target_name=target,
            use_feature_selection=False,
            feature_names=feature_cols,
            dataset_name=dataset_name
        )
        all_results.append(results_no_fs)
        
        # Train WITH feature selection
        results_with_fs = train_and_evaluate_ridge(
            X_train, y_train, X_val, y_val, X_test, y_test,
            target_name=target,
            use_feature_selection=True,
            fs_method='rfe',
            k=k,
            feature_names=feature_cols,
            dataset_name=dataset_name
        )
        all_results.append(results_with_fs)
        
        # Compare results
        compare_results(results_no_fs, results_with_fs, dataset_name)
    
    return all_results


def print_final_summary(all_results):
    """Print final summary of all experiments"""
    print(f"\n\n{'='*80}")
    print(f"FINAL SUMMARY - ALL RESULTS")
    print(f"{'='*80}")
    
    print(f"\n{'Target':<15} {'Features':<12} {'Best Alpha':<12} {'Train RMSE':<12} {'Test RMSE':<12} {'Test R²':<12}")
    print(f"{'-'*85}")
    
    for result in all_results:
        fs_label = f"{result['n_features']} (FS)" if result['feature_selection'] else f"{result['n_features']}"
        print(f"{result['target']:<15} {fs_label:<12} {result['best_alpha']:<12.4f} "
              f"{result['train_rmse']:<12.4f} {result['test_rmse']:<12.4f} {result['test_r2']:<12.4f}")


# ============================================================================
# COMBINED VISUALIZATION FUNCTIONS
# ============================================================================

def create_combined_visualizations():
    """Create comprehensive combined visualizations - minimal files"""
    print(f"\n\n{'='*80}")
    print("GENERATING COMBINED VISUALIZATIONS")
    print(f"{'='*80}\n")
    
    # Organize results by dataset
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
    
    # Create visualizations
    for dataset_name, targets_data in datasets.items():
        create_dataset_comprehensive_figure(dataset_name, targets_data)
    
    # Create overall summary
    create_overall_summary_figure(datasets)
    
    print(f"\nAll visualizations generated successfully!")
    print(f"  Total files created: {len(datasets) * 1 + 1}")


def create_dataset_comprehensive_figure(dataset_name, targets_data):
    """Create ONE comprehensive figure per dataset with all analyses"""
    targets = list(targets_data.keys())
    n_targets = len(targets)
    
    # Create figure with gridspec: 6 columns (CV, Pred-NoFS, Pred-FS, Res-NoFS, Res-FS, FeatImp)
    fig = plt.figure(figsize=(24, 5 * n_targets))
    gs = fig.add_gridspec(n_targets, 6, hspace=0.3, wspace=0.3)
    
    for i, target in enumerate(targets):
        data_no_fs = targets_data[target]['no_fs']
        data_fs = targets_data[target]['fs']
        
        if data_no_fs is None or data_fs is None:
            continue
        
        # Column 0: CV Results Comparison
        ax_cv = fig.add_subplot(gs[i, 0])
        for data, label, color in [(data_no_fs, 'No FS (68)', 'steelblue'), 
                                     (data_fs, 'FS (34)', 'coral')]:
            alphas = list(data['cv_results'].keys())
            avg_rmses = [data['cv_results'][alpha]['avg_rmse'] for alpha in alphas]
            ax_cv.plot(alphas, avg_rmses, marker='o', label=label, color=color, linewidth=2, markersize=4)
        
        ax_cv.set_xscale('log')
        ax_cv.set_xlabel('Alpha', fontsize=9)
        ax_cv.set_ylabel('CV RMSE', fontsize=9)
        ax_cv.set_title(f'{target.upper()}\nCross-Validation', fontsize=10, fontweight='bold')
        ax_cv.legend(fontsize=7)
        ax_cv.grid(True, alpha=0.3)
        
        # Column 1: Test Predictions Without FS
        ax_pred1 = fig.add_subplot(gs[i, 1])
        y_test, y_pred = data_no_fs['y_test'], data_no_fs['y_test_pred']
        ax_pred1.scatter(y_test, y_pred, alpha=0.5, s=25, edgecolors='black', linewidth=0.2)
        min_val, max_val = min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())
        ax_pred1.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=1.5)
        rmse, r2 = np.sqrt(mean_squared_error(y_test, y_pred)), r2_score(y_test, y_pred)
        ax_pred1.set_xlabel('Actual', fontsize=8)
        ax_pred1.set_ylabel('Predicted', fontsize=8)
        ax_pred1.set_title(f'Test (No FS)\nRMSE={rmse:.3f} R²={r2:.3f}', fontsize=9, fontweight='bold')
        ax_pred1.grid(True, alpha=0.3)
        
        # Column 2: Test Predictions With FS
        ax_pred2 = fig.add_subplot(gs[i, 2])
        y_test, y_pred = data_fs['y_test'], data_fs['y_test_pred']
        ax_pred2.scatter(y_test, y_pred, alpha=0.5, s=25, edgecolors='black', linewidth=0.2, color='coral')
        min_val, max_val = min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())
        ax_pred2.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=1.5)
        rmse, r2 = np.sqrt(mean_squared_error(y_test, y_pred)), r2_score(y_test, y_pred)
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
        
        # Column 5: Feature Importance Comparison
        ax_feat = fig.add_subplot(gs[i, 5])
        
        # Plot top 10 features from both models
        if data_no_fs['feature_names'] is not None:
            coeffs_no_fs = np.abs(data_no_fs['model'].coef_)
            top_idx = np.argsort(coeffs_no_fs)[-10:][::-1]
            top_features = [data_no_fs['feature_names'][i] for i in top_idx]
            top_coeffs = coeffs_no_fs[top_idx]
            
            y_pos = np.arange(len(top_features))
            colors = sns.color_palette('viridis', len(top_features))
            ax_feat.barh(y_pos, top_coeffs, color=colors, edgecolor='black', linewidth=0.5)
            ax_feat.set_yticks(y_pos)
            ax_feat.set_yticklabels(top_features, fontsize=7)
            ax_feat.set_xlabel('|Coefficient|', fontsize=8)
            ax_feat.set_title(f'Top 10 Features (No FS)', fontsize=9, fontweight='bold')
            ax_feat.invert_yaxis()
            ax_feat.grid(True, alpha=0.3, axis='x')
    
    fig.suptitle(f'Comprehensive Ridge Regression Analysis: {dataset_name}', 
                 fontsize=16, fontweight='bold', y=0.995)
    
    filename = f'{dataset_name.lower().replace("-", "")}_full_analysis.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {filename}")


def create_overall_summary_figure(datasets):
    """Create ONE overall summary figure comparing all datasets and targets"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Collect all data
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
                all_labels.append(f"{dataset_name[:4]}\n{target}")
                test_rmse_no_fs.append(data['no_fs']['test_rmse'])
                test_rmse_fs.append(data['fs']['test_rmse'])
                test_r2_no_fs.append(data['no_fs']['test_r2'])
                test_r2_fs.append(data['fs']['test_r2'])
                train_rmse_no_fs.append(data['no_fs']['train_rmse'])
                train_rmse_fs.append(data['fs']['train_rmse'])
    
    x = np.arange(len(all_labels))
    width = 0.35
    
    # Test RMSE Comparison
    axes[0, 0].bar(x - width/2, test_rmse_no_fs, width, label='No FS (68)', color='steelblue', edgecolor='black')
    axes[0, 0].bar(x + width/2, test_rmse_fs, width, label='FS (34)', color='coral', edgecolor='black')
    axes[0, 0].set_ylabel('Test RMSE', fontsize=11)
    axes[0, 0].set_title('Test RMSE Comparison', fontsize=12, fontweight='bold')
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(all_labels, fontsize=8)
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3, axis='y')
    
    # Test R² Comparison
    axes[0, 1].bar(x - width/2, test_r2_no_fs, width, label='No FS (68)', color='steelblue', edgecolor='black')
    axes[0, 1].bar(x + width/2, test_r2_fs, width, label='FS (34)', color='coral', edgecolor='black')
    axes[0, 1].set_ylabel('Test R²', fontsize=11)
    axes[0, 1].set_title('Test R² Comparison', fontsize=12, fontweight='bold')
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels(all_labels, fontsize=8)
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3, axis='y')
    
    # Train RMSE Comparison
    axes[1, 0].bar(x - width/2, train_rmse_no_fs, width, label='No FS (68)', color='steelblue', edgecolor='black')
    axes[1, 0].bar(x + width/2, train_rmse_fs, width, label='FS (34)', color='coral', edgecolor='black')
    axes[1, 0].set_ylabel('Train RMSE', fontsize=11)
    axes[1, 0].set_title('Train RMSE Comparison', fontsize=12, fontweight='bold')
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(all_labels, fontsize=8)
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3, axis='y')
    
    # RMSE Improvement Percentages
    improvements = [(fs - no_fs) / no_fs * 100 for fs, no_fs in zip(test_rmse_fs, test_rmse_no_fs)]
    colors_imp = ['green' if imp < 0 else 'red' for imp in improvements]
    axes[1, 1].bar(x, improvements, color=colors_imp, edgecolor='black')
    axes[1, 1].axhline(y=0, color='black', linestyle='-', linewidth=1)
    axes[1, 1].set_ylabel('RMSE Change (%)', fontsize=11)
    axes[1, 1].set_title('Feature Selection Impact on Test RMSE\\n(Negative = Improvement)', 
                         fontsize=12, fontweight='bold')
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(all_labels, fontsize=8)
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    filename = 'overall_model_comparison.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {filename}")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function"""
    print("\n" + "="*80)
    print(" "*20 + "RIDGE REGRESSION ANALYSIS")
    print(" "*15 + "with 5-Fold Cross-Validation")
    print(" "*15 + "and Feature Selection Comparison")
    print("="*80)
    
    # Define datasets
    datasets = [
        ('../Data/emosounds-3_preprocessed.csv', 'EmoSounds-3'),
        ('../Data/iadsed-2_preprocessed.csv', 'IADSED-2')
    ]
    
    # Define targets to predict
    targets = ['arousal', 'valence', 'dominance']
    
    # Number of features to select
    k_features = 34
    
    all_results = []
    
    # Process each dataset
    for filepath, dataset_name in datasets:
        try:
            results = process_dataset(filepath, dataset_name, targets=targets, k=k_features)
            all_results.extend(results)
        except FileNotFoundError:
            print(f"\nError: File '{filepath}' not found. Skipping {dataset_name}.")
        except Exception as e:
            print(f"\nError processing {dataset_name}: {str(e)}")
    
    # Print final summary
    if all_results:
        print_final_summary(all_results)
    
    # Create all combined visualizations
    create_combined_visualizations()
    
    print("\n" + "="*80)
    print(" "*25 + "ANALYSIS COMPLETE")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
