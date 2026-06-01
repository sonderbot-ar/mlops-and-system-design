from session_4.src.transform import Transformer, balance_dataset
import pandas as pd
import os


def test_map_binary_column_to_int():
    transformer = Transformer()
    df = pd.DataFrame(
        {
            "Gender": ["Male", "Female", "Male", "Female"]
        }
    )

    expected_df = pd.DataFrame(
        {"Gender": [0, 1, 0, 1]}
    )

    transformed_df = transformer._map_binary_column_to_int(df)

    # Test the result against the expected DataFrame
    pd.testing.assert_frame_equal(transformed_df, expected_df)
    
    # Verify Gender column mapping
    assert transformed_df["Gender"].tolist() == [0, 1, 0, 1]



def create_df_balance():
    return pd.DataFrame(
        {
            "Age": [25, 30, 35, 40, 45, 50, 55, 60, 65, 70],
            "CreditScore": [600, 650, 700, 750, 800, 650, 700, 750, 800, 850],
            "Exited": [0, 0, 0, 0, 0, 1, 1, 1, 1, 1],  # Use Exited like real data
        }
    )


def test_balance_dataset():
    """Test that balance_dataset creates equal class distribution."""
    df_balance = create_df_balance()
    
    # Verify the test data has imbalance if we modify it
    balanced_df = balance_dataset(df_balance)
    
    # Check that classes are equal after balancing
    class_counts = balanced_df["Exited"].value_counts()
    assert class_counts[0] == class_counts[1], "Classes should be balanced"
    
    # Check that the balanced dataset has same size as the minority class * 2
    minority_size = df_balance["Exited"].value_counts().min()
    assert len(balanced_df) == minority_size * 2


def test_balance_with_unequal_classes():
    """Test balancing with actual imbalanced data (3:1 ratio)."""
    df_unequal = pd.DataFrame(
        {
            "Age": [25, 30, 35, 40, 45],
            "CreditScore": [600, 650, 700, 750, 800],
            "Exited": [0, 0, 0, 1, 1],  # 3:1 imbalance
        }
    )

    # Balance the dataset
    balanced_df = balance_dataset(df_unequal)

    # Check if classes are balanced
    class_counts = balanced_df["Exited"].value_counts()
    assert class_counts[0] == class_counts[1], "Classes should be equal after balancing"
    
    # Minority class had 2 samples, so balanced should have 4 total (2 per class)
    assert len(balanced_df) == 4, f"Expected 4 rows, got {len(balanced_df)}"


def test_balance_with_real_churn_data():
    data_path = "/Users/andresrodartee/mlops-and-system-design/session_4/Churn_Modelling_train_test copy.csv"
    
    df = pd.read_csv(data_path)

    before_counts = df["Exited"].value_counts()
    imbalance_ratio = before_counts[0] / before_counts[1]
        
    assert imbalance_ratio > 1, "Data should be imbalanced"
        
    balanced_df = balance_dataset(df)
        
    after_counts = balanced_df["Exited"].value_counts()
    assert after_counts[0] == after_counts[1], "Classes should be balanced"
        
    minority_class_size = before_counts.min()
    assert len(balanced_df) == minority_class_size * 2
