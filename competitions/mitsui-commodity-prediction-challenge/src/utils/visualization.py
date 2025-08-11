import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


def plot_time_series(df: pd.DataFrame, 
                    date_col: str, 
                    value_cols: List[str], 
                    title: str = "Time Series Plot",
                    figsize: Tuple[int, int] = (15, 8)) -> None:
    """
    Plot multiple time series on the same chart
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    for col in value_cols:
        if col in df.columns:
            ax.plot(pd.to_datetime(df[date_col]), df[col], label=col, alpha=0.8)
    
    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Value', fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def plot_correlation_matrix(df: pd.DataFrame, 
                          title: str = "Correlation Matrix",
                          figsize: Tuple[int, int] = (12, 10)) -> None:
    """
    Plot correlation matrix heatmap
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    corr_matrix = df[numeric_cols].corr()
    
    fig, ax = plt.subplots(figsize=figsize)
    
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='coolwarm', center=0,
                square=True, linewidths=0.5, cbar_kws={"shrink": 0.8}, ax=ax)
    
    ax.set_title(title, fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()


def plot_distribution(df: pd.DataFrame, 
                     columns: List[str], 
                     title: str = "Distribution Plots",
                     figsize: Tuple[int, int] = (15, 10)) -> None:
    """
    Plot distribution of multiple columns
    """
    n_cols = min(3, len(columns))
    n_rows = (len(columns) + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    
    if n_rows == 1 and n_cols == 1:
        axes = [axes]
    elif n_rows == 1:
        axes = list(axes)
    else:
        axes = axes.flatten()
    
    for i, col in enumerate(columns):
        if col in df.columns and i < len(axes):
            ax = axes[i]
            
            ax.hist(df[col].dropna(), bins=50, alpha=0.7, edgecolor='black')
            ax.set_title(f'Distribution of {col}', fontweight='bold')
            ax.set_xlabel(col)
            ax.set_ylabel('Frequency')
            ax.grid(True, alpha=0.3)
    
    for i in range(len(columns), len(axes)):
        axes[i].set_visible(False)
    
    plt.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()


def plot_missing_values(df: pd.DataFrame, 
                       title: str = "Missing Values Analysis",
                       figsize: Tuple[int, int] = (12, 8)) -> None:
    """
    Visualize missing values in the dataset
    """
    missing_data = df.isnull().sum()
    missing_data = missing_data[missing_data > 0].sort_values(ascending=False)
    
    if len(missing_data) == 0:
        print("No missing values found in the dataset!")
        return
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    missing_data.plot(kind='bar', ax=ax1, color='coral')
    ax1.set_title('Missing Values Count', fontweight='bold')
    ax1.set_xlabel('Columns')
    ax1.set_ylabel('Missing Count')
    ax1.tick_params(axis='x', rotation=45)
    
    missing_percent = (missing_data / len(df)) * 100
    missing_percent.plot(kind='bar', ax=ax2, color='lightblue')
    ax2.set_title('Missing Values Percentage', fontweight='bold')
    ax2.set_xlabel('Columns')
    ax2.set_ylabel('Missing Percentage (%)')
    ax2.tick_params(axis='x', rotation=45)
    
    plt.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()


def plot_target_analysis(df: pd.DataFrame, 
                        target_col: str,
                        date_col: Optional[str] = None,
                        figsize: Tuple[int, int] = (15, 10)) -> None:
    """
    Comprehensive target variable analysis
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    axes[0, 0].hist(df[target_col].dropna(), bins=50, alpha=0.7, edgecolor='black')
    axes[0, 0].set_title(f'Distribution of {target_col}', fontweight='bold')
    axes[0, 0].set_xlabel(target_col)
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].boxplot(df[target_col].dropna())
    axes[0, 1].set_title(f'Box Plot of {target_col}', fontweight='bold')
    axes[0, 1].set_ylabel(target_col)
    axes[0, 1].grid(True, alpha=0.3)
    
    from scipy import stats
    stats.probplot(df[target_col].dropna(), dist="norm", plot=axes[1, 0])
    axes[1, 0].set_title(f'Q-Q Plot of {target_col}', fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3)
    
    if date_col and date_col in df.columns:
        axes[1, 1].plot(pd.to_datetime(df[date_col]), df[target_col], alpha=0.7)
        axes[1, 1].set_title(f'{target_col} Over Time', fontweight='bold')
        axes[1, 1].set_xlabel('Date')
        axes[1, 1].set_ylabel(target_col)
        axes[1, 1].tick_params(axis='x', rotation=45)
        axes[1, 1].grid(True, alpha=0.3)
    else:
        sorted_values = np.sort(df[target_col].dropna())
        cumulative = np.arange(1, len(sorted_values) + 1) / len(sorted_values)
        axes[1, 1].plot(sorted_values, cumulative)
        axes[1, 1].set_title(f'Cumulative Distribution of {target_col}', fontweight='bold')
        axes[1, 1].set_xlabel(target_col)
        axes[1, 1].set_ylabel('Cumulative Probability')
        axes[1, 1].grid(True, alpha=0.3)
    
    plt.suptitle(f'Target Variable Analysis: {target_col}', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()


def save_plot(filename: str, dpi: int = 300) -> None:
    """
    Save the current plot to file
    """
    plt.savefig(filename, dpi=dpi, bbox_inches='tight')
    print(f"Plot saved as {filename}")
