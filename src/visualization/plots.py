"""
Visualization utilities.
"""
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np


def plot_feature_importance(feature_importance, feature_names, title="Feature Importance", figsize=(12, 8)):
    """
    Plot feature importance.
    
    Args:
        feature_importance: Array of feature importance values
        feature_names: Array of feature names
        title: Plot title
        figsize: Figure size
    """
    # Create DataFrame for easier plotting
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': feature_importance
    }).sort_values('Importance', ascending=False)
    
    plt.figure(figsize=figsize)
    sns.barplot(x='Importance', y='Feature', data=importance_df)
    plt.title(title)
    plt.tight_layout()
    plt.show()
    
    return importance_df


def plot_distribution_comparison(train_data, test_data, columns, n_cols=3, figsize=(16, 12)):
    """
    Plot distribution comparison between train and test datasets.
    
    Args:
        train_data: Training DataFrame
        test_data: Testing DataFrame
        columns: List of columns to plot
        n_cols: Number of columns in the subplot grid
        figsize: Figure size
    """
    n_rows = (len(columns) + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten()
    
    for i, col in enumerate(columns):
        if i < len(axes):
            ax = axes[i]
            
            # Handle numeric vs categorical features
            if pd.api.types.is_numeric_dtype(train_data[col]):
                sns.histplot(train_data[col], color='blue', alpha=0.5, label='Train', ax=ax)
                sns.histplot(test_data[col], color='red', alpha=0.5, label='Test', ax=ax)
            else:
                # For categorical features
                train_counts = train_data[col].value_counts(normalize=True)
                test_counts = test_data[col].value_counts(normalize=True)
                
                # Combine and reindex to ensure both have the same categories
                all_categories = pd.concat([train_counts, test_counts]).index.unique()
                train_counts = train_counts.reindex(all_categories, fill_value=0)
                test_counts = test_counts.reindex(all_categories, fill_value=0)
                
                # Create a DataFrame for plotting
                comp_df = pd.DataFrame({
                    'Train': train_counts,
                    'Test': test_counts
                }).reset_index().melt(id_vars='index', var_name='Dataset', value_name='Proportion')
                
                sns.barplot(x='index', y='Proportion', hue='Dataset', data=comp_df, ax=ax)
                ax.tick_params(axis='x', rotation=45)
            
            ax.set_title(col)
            ax.legend()
    
    # Hide unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    
    plt.tight_layout()
    plt.show()
