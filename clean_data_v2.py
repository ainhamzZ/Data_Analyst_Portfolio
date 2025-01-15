import pandas as pd
import argparse
import numpy as np
import sys
from sklearn.preprocessing import LabelEncoder



def clean_data(df):


# 1. Impute missing values for numerical columns with mean
    while True:
        try:
            # Ask how many numerical columns the user wants to select for imputing mean
            num_cols_mean = input(f"How many numerical columns would you like to select to impute mean: ")
            if not num_cols_mean.isdigit():
                raise ValueError(f"'{num_cols_mean}' is not a valid number. Please enter a valid number.")
            num_cols_mean = int(num_cols_mean)
            input_cols_mean = input(f"Please enter {num_cols_mean} numerical column name(s), separated by spaces to impute mean: ").split()
            if len(input_cols_mean) != num_cols_mean:
                raise ValueError(f"You must enter exactly {num_cols_mean} column names.")
            
            # Validate and impute mean for numerical columns
            for col in input_cols_mean:
                if col not in df.columns:
                    raise ValueError(f"'{col}' is not a valid numerical column.")
                if df[col].isnull().any():  # Check if the column has missing values
                    df[col].fillna(df[col].mean(), inplace=True)  # Impute with mean
                    print(f"Missing values in '{col}' filled with mean: {df[col].mean()}")
                else:
                    print(f"No missing values in '{col}'.")
            break  # Exit loop if successful

        except ValueError as e:
            print(f"Error: {e}. Please try again. ")

# 1. Impute missing values for numerical columns with median
    while True:
        try:
            # Ask how many numerical columns the user wants to select for imputing median
            num_cols_median = int(input(f"How many numerical columns would you like to select to impute median: "))
            input_cols_median = input(f"Please enter {num_cols_median} numerical column names, separated by spaces: ").split()
            if len(input_cols_median) != num_cols_median:
                raise ValueError(f"You must enter exactly {num_cols_median} column names.")
 
            # Validate and impute mean for numerical columns
            for col in input_cols_median:
                if col not in df.columns:
                    raise ValueError(f"'{col}' is not a valid numerical column.")
                if df[col].isnull().any():  # Check if the column has missing values
                    df[col].fillna(df[col].median(), inplace=True)  # Impute with median
                    print(f"Missing values in '{col}' filled with median: {df[col].median()}")
                else:
                    print(f"No missing values in '{col}'.")
            break  # Exit loop if successful

        except ValueError as e:
            print(f"Error: {e}. Please try again . ")            

    # 2. Fill missing values and encode for categorical columns
    while True:
        try:
            # Ask how many categorical columns the user wants to select
            cat_cols = int(input(f"How many categorical columns would you like to select: "))
            input_cols_mode = input(f"Please enter {cat_cols} categorical column names, separated by spaces: ").split()
            if len(input_cols_mode) != cat_cols:
                raise ValueError(f"You must enter exactly {cat_cols} column names.")
            
            # Validate and impute mode for categorical columns
            label_encoder = LabelEncoder()
            for col in input_cols_mode:
                if col not in df.columns:
                    raise ValueError(f"'{col}' is not a valid categorical column.")
                if df[col].isnull().any():  # Check if the column has missing values
                    df[col].fillna(df[col].mode()[0], inplace=True)  # Impute with mode
                    print(f"Missing values in '{col}' filled with mode: {df[col].mode()[0]}")
                else:
                    print(f"No missing values in '{col}'.")                    
                # Apply label encoding
                df[col] = label_encoder.fit_transform(df[col])
                print(f"'{col}' has been label encoded.")
            break  # Exit loop if successful

        except ValueError as e:
            print(f"Error: {e}. Please try again")
    
    return df

if __name__ == "__main__":
    # Set up argument parsing   
    parser = argparse.ArgumentParser(description="Clean and drop specified columns from a DataFrame")
    parser.add_argument(
    "--columns", 
    required=True, 
    help="Comma-separated column names to drop, e.g., 'Age,City'"
)
    parser.add_argument(
    "--input", 
    required=True, 
    help="Path to the CSV file to clean"
)
    parser.add_argument(
    "--output", 
    required=True, 
    help="Path to save the cleaned CSV file"
)
   
    # Parse arguments
    args = parser.parse_args()

    # Load the DataFrame
    print(f"Attempting to load file from: {args.input}")
    df = pd.read_csv(args.input)

    try:
        # Convert comma-separated string to a list of columns to drop. strip to remove whitespace after
        columns_to_drop = [col.strip() for col in args.columns.split(",")]
        
        # Check if all columns exist in the DataFrame
        invalid_columns = [col for col in columns_to_drop if col not in df.columns]
        if invalid_columns:
            raise ValueError(f"The following columns cannot be dropped as they do not exist in the DataFrame: {invalid_columns}")
        else:
            # Drop valid columns
            df = df.drop(columns=columns_to_drop)
            print(f"Successfully dropped columns: {columns_to_drop}")
    except ValueError as e:
        print(f'Error {e}. Please try again') 
        sys.exit(1) 

    # Clean the DataFrame (impute missing values)
    df = clean_data(df)

    # Save the cleaned DataFrame to the specified output path
    df.to_csv(args.output, index=False)
    
    print(f"Cleaned data saved to {args.output}")
