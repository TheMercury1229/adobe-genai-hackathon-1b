"""
Enhanced CSV-based trainer for PDF Heading Detection with proper feature weighting
This script includes comprehensive data validation and feature engineering to ensure
all features get proper importance weights.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.feature_selection import mutual_info_classif
import joblib
import json
import os
import warnings
warnings.filterwarnings('ignore')


def validate_and_analyze_features(df, feature_columns):
    """Comprehensive feature validation and analysis."""

    for col in feature_columns:
        if col not in df.columns:
            continue

        if df[col].nunique() <= 10:  # Show value counts for categorical-like features
            value_counts = df[col].value_counts()
            for val, count in value_counts.items():
                pct = (count / len(df)) * 100

        # Check if feature has any variance
        if df[col].nunique() == 1:
            print(f"   ⚠️  WARNING: {col} has no variance (constant value)")
        elif df[col].nunique() == 2 and set(df[col].unique()) == {0, 1}:
            print(f"   ✓ Binary feature with good distribution")

    return True


def enhanced_feature_engineering(df):
    """Enhanced feature engineering with proper weighting."""

    # Convert boolean columns more robustly
    boolean_columns = [
        'is_bold', 'is_italic', 'is_superscript', 'is_all_caps',
        'is_title_case', 'has_numbers', 'starts_with_number',
        'is_largest_font', 'is_above_avg_font'
    ]

    for col in boolean_columns:
        if col in df.columns:
            original_unique = df[col].unique()
            print(f"Converting {col}: {original_unique}")

            # More robust boolean conversion with safer string handling
            df[col] = df[col].astype(str).str.strip().str.upper()

            # Handle common boolean representations
            boolean_map = {
                'TRUE': 1, 'FALSE': 0, '1': 1, '0': 0,
                'YES': 1, 'NO': 0, 'T': 1, 'F': 0,
                '1.0': 1, '0.0': 0, 'NAN': 0, 'NONE': 0
            }

            df[col] = df[col].map(boolean_map)

            # Fill any unmapped values with 0 and ensure integer type
            df[col] = pd.to_numeric(
                df[col], errors='coerce').fillna(0).astype(int)

            final_unique = df[col].unique()
            value_counts = df[col].value_counts().to_dict()

    # Create composite features to give more weight to formatting

    # Strong formatting indicator
    formatting_features = ['is_bold', 'is_italic',
                           'is_all_caps', 'is_title_case']
    available_formatting = [
        col for col in formatting_features if col in df.columns]

    if available_formatting:
        df['formatting_score'] = df[available_formatting].sum(axis=1)
        print(
            f"Created formatting_score: {df['formatting_score'].value_counts().to_dict()}")

    # Font emphasis feature
    if 'is_bold' in df.columns and 'font_size_ratio' in df.columns:
        # Ensure numeric types before multiplication
        df['is_bold'] = pd.to_numeric(df['is_bold'], errors='coerce').fillna(0)
        df['font_size_ratio'] = pd.to_numeric(
            df['font_size_ratio'], errors='coerce').fillna(1.0)
        df['font_emphasis'] = df['is_bold'] * df['font_size_ratio']

        # Safe formatting with type check
        try:
            min_val = float(df['font_emphasis'].min())
            max_val = float(df['font_emphasis'].max())
            print(
                f"Created font_emphasis: range {min_val:.3f} to {max_val:.3f}")
        except (ValueError, TypeError):
            print(
                f"Created font_emphasis: {df['font_emphasis'].nunique()} unique values")

    # Position-based features (ensure relative_y and relative_x are numeric)
    if 'relative_y' in df.columns and 'relative_x' in df.columns:
        # Double-check these are numeric
        df['relative_y'] = pd.to_numeric(
            df['relative_y'], errors='coerce').fillna(0)
        df['relative_x'] = pd.to_numeric(
            df['relative_x'], errors='coerce').fillna(0)

        df['is_top_third'] = (df['relative_y'] < 0.33).astype(int)
        df['is_left_aligned'] = (df['relative_x'] < 0.1).astype(int)
    else:
        # Create default values if coordinates missing
        df['is_top_third'] = 0
        df['is_left_aligned'] = 0

    # Text length categories
    if 'char_count' in df.columns:
        # Ensure char_count is numeric
        df['char_count'] = pd.to_numeric(
            df['char_count'], errors='coerce').fillna(0)

        df['is_short_text'] = (df['char_count'] <= 5).astype(int)
        df['is_medium_text'] = ((df['char_count'] > 5) & (
            df['char_count'] <= 50)).astype(int)
        df['is_long_text'] = (df['char_count'] > 50).astype(int)

    else:
        # Create default values
        df['is_short_text'] = 0
        df['is_medium_text'] = 1  # Default to medium
        df['is_long_text'] = 0

    return df


def load_and_preprocess_csv(csv_path):
    """Enhanced CSV loading with better preprocessing."""

    # Load the CSV file
    df = pd.read_csv(csv_path, header=0)

    # Always check first few rows for potential header issues

    # More robust header detection - check if first row contains obvious header strings
    first_row_str = df.iloc[0].astype(str).str.lower()
    header_indicators = ['page_num', 'is_bold',
                         'label', 'font_size', 'position', 'text']

    if any(indicator in ' '.join(first_row_str.values) for indicator in header_indicators):
        print("⚠️  DETECTED: First row contains column names/headers. Removing it...")
        df = df.iloc[1:].reset_index(drop=True)

    # Additional check: if any numeric columns have string values, likely header issue
    numeric_test_cols = ['page_num', 'font_size_ratio', 'char_count']
    for col in numeric_test_cols:
        if col in df.columns:
            try:
                pd.to_numeric(df[col].iloc[0])
            except (ValueError, TypeError):
                print(
                    f"⚠️  DETECTED: Column '{col}' has non-numeric first value: '{df[col].iloc[0]}'")
                if len(df) > 1:  # Safety check
                    print("Removing problematic first row...")
                    df = df.iloc[1:].reset_index(drop=True)
                break

    # FIRST: Convert all numeric columns before any feature engineering
    numeric_columns = [
        'page_num', 'block_num', 'line_num', 'font_size', 'position_x', 'position_y',
        'line_width', 'line_height', 'page_width', 'page_height', 'char_count',
        'word_count', 'distance_from_left', 'distance_from_right',
        'distance_from_top', 'distance_from_bottom', 'space_above', 'space_below',
        'font_size_ratio'
    ]

    for col in numeric_columns:
        if col in df.columns:
            print(f"Converting numeric column '{col}'...")
            original_dtype = df[col].dtype

            # First convert to string and clean
            df[col] = df[col].astype(str).str.strip()

            # Convert to numeric
            df[col] = pd.to_numeric(df[col], errors='coerce')

            # Count and fill NaN values
            nan_count = df[col].isnull().sum()
            if nan_count > 0:
                fill_value = df[col].median() if col != 'page_num' else 1
                df[col] = df[col].fillna(fill_value)

    # Create normalized coordinates BEFORE feature engineering
    if 'page_width' in df.columns and 'page_height' in df.columns:
        if 'relative_x' not in df.columns:
            # Ensure position and page dimensions are numeric
            df['position_x'] = pd.to_numeric(
                df['position_x'], errors='coerce').fillna(0)
            df['page_width'] = pd.to_numeric(
                df['page_width'], errors='coerce').fillna(1)

            # Avoid division by zero
            df['page_width'] = df['page_width'].replace(0, 1)
            df['relative_x'] = df['position_x'] / df['page_width']
            df['relative_x'] = df['relative_x'].fillna(0)

        if 'relative_y' not in df.columns:
            # Ensure position and page dimensions are numeric
            df['position_y'] = pd.to_numeric(
                df['position_y'], errors='coerce').fillna(0)
            df['page_height'] = pd.to_numeric(
                df['page_height'], errors='coerce').fillna(1)

            # Avoid division by zero
            df['page_height'] = df['page_height'].replace(0, 1)
            df['relative_y'] = df['position_y'] / df['page_height']
            df['relative_y'] = df['relative_y'].fillna(0)

        # Ensure final coordinates are numeric and get safe stats
        df['relative_x'] = pd.to_numeric(
            df['relative_x'], errors='coerce').fillna(0)
        df['relative_y'] = pd.to_numeric(
            df['relative_y'], errors='coerce').fillna(0)

        # Safe range display
        try:
            x_min, x_max = float(df['relative_x'].min()), float(
                df['relative_x'].max())
            y_min, y_max = float(df['relative_y'].min()), float(
                df['relative_y'].max())
        except (ValueError, TypeError):
            print(
                f"  Sample values: x={df['relative_x'].head(3).tolist()}, y={df['relative_y'].head(3).tolist()}")

    # NOW do enhanced feature engineering with clean numeric data
    df = enhanced_feature_engineering(df)

    # Enhanced feature set with proper weighting
    feature_columns = [
        # Core positioning (keep these)
        'page_num', 'relative_x', 'relative_y',

        # Text characteristics
        'char_count', 'word_count',

        # Font and formatting (the main issue)
        'font_size_ratio', 'is_bold', 'is_italic', 'is_all_caps', 'is_title_case',
        'is_superscript', 'has_numbers', 'starts_with_number', 'is_above_avg_font',

        # Composite features for better discrimination
        'formatting_score', 'font_emphasis', 'is_top_third', 'is_left_aligned',
        'is_short_text', 'is_medium_text', 'is_long_text'
    ]

    # Keep only available features
    available_features = [col for col in feature_columns if col in df.columns]
    missing_features = [
        col for col in feature_columns if col not in df.columns]

    if missing_features:
        # Add missing features with default values
        for col in missing_features:
            df[col] = 0

    # Handle missing values
    for col in available_features:
        if df[col].isnull().sum() > 0:
            if col in ['is_bold', 'is_italic', 'is_all_caps', 'is_title_case', 'formatting_score']:
                df[col] = df[col].fillna(0)
            else:
                df[col] = df[col].fillna(df[col].median())

    # Validate features
    validate_and_analyze_features(df, available_features)

    return df, available_features


def train_enhanced_random_forest(df, feature_columns):
    """Train Random Forest with enhanced configuration for better feature utilization."""

    # Prepare features and target
    X = df[feature_columns].copy()
    y = df['label'].copy()

    # Show label distribution
    label_counts = y.value_counts()
    print("\nLabel distribution:")
    for label, count in label_counts.items():
        percentage = (count / len(df)) * 100
        print(f"  {label}: {count} ({percentage:.1f}%)")

    # Encode labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    # Calculate mutual information to understand feature relevance
    mi_scores = mutual_info_classif(X, y_encoded, random_state=42)
    mi_df = pd.DataFrame({
        'feature': feature_columns,
        'mutual_info': mi_scores
    }).sort_values('mutual_info', ascending=False)

    # Split data
    can_stratify = all(count >= 2 for count in pd.Series(
        y_encoded).value_counts().values)

    if can_stratify:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=0.2, stratify=y_encoded, random_state=42
        )
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=0.2, random_state=42
        )

    # Calculate balanced class weights
    classes = np.unique(y_encoded)
    class_weights = compute_class_weight(
        'balanced', classes=classes, y=y_encoded)
    class_weight_dict = {c: w for c, w in zip(classes, class_weights)}

    # Enhanced Random Forest configuration (memory-optimized)
    rf_model = RandomForestClassifier(
        n_estimators=300,           # Balanced trees for good performance vs size
        max_depth=15,               # Reasonable depth to prevent overfitting
        min_samples_split=3,        # Lower threshold for more splits
        min_samples_leaf=2,         # Slightly higher for generalization
        max_features=0.6,           # Good feature sampling
        n_jobs=-1,
        class_weight=class_weight_dict,
        bootstrap=True,
        oob_score=True,             # Out-of-bag scoring
        random_state=42,
        ccp_alpha=0.001            # Light pruning for smaller model
    )

    rf_model.fit(X_train, y_train)

    # Evaluate model
    train_score = rf_model.score(X_train, y_train)
    test_score = rf_model.score(X_test, y_test)
    oob_score = rf_model.oob_score_

    # Detailed evaluation
    y_pred = rf_model.predict(X_test)
    unique_labels = np.unique(np.concatenate([y_test, y_pred]))
    present_class_names = [label_encoder.classes_[i] for i in unique_labels]

    # Enhanced feature importance analysis
    feature_importance = pd.DataFrame({
        'feature': feature_columns,
        'importance': rf_model.feature_importances_,
        'mutual_info': mi_scores
    }).sort_values('importance', ascending=False)

    print("\n📈 FEATURE IMPORTANCE ANALYSIS:")
    print("="*50)
    for idx, row in feature_importance.iterrows():
        print(
            f"{row['feature']:20} | RF: {row['importance']:.4f} | MI: {row['mutual_info']:.4f}")

    # Identify problematic features
    zero_importance = feature_importance[feature_importance['importance'] == 0]
    if len(zero_importance) > 0:
        for feat in zero_importance['feature']:
            unique_vals = df[feat].nunique()
            val_counts = df[feat].value_counts()

    return rf_model, label_encoder, feature_importance
