import pandas as pd
import argparse
import numpy as np
import sys
from sklearn.preprocessing import LabelEncoder,MinMaxScaler,StandardScaler
from ydata_profiling import ProfileReport
import os
import openpyxl

def fix_data_types(df):
    """
    This function checks and fixes data types in a DataFrame:
    - Converts object columns to string if they contain only text.
    - Converts numeric columns to int or float if needed.
    - Converts date columns to datetime.
    - Replaces invalid values with NaN.
    """
    
    for col in df.columns:
        # Detect missing values
        missing_values = df[col].isna().sum()
        
        # Attempt to infer column data type
        if df[col].dtype == 'datetime64':  
            try:
                df[col] = pd.to_datetime(df[col], errors='coerce')  # Try converting to datetime
                if df[col].notna().sum() > 0:  # If at least one value was successfully converted
                    print(f" Column '{col}' converted to datetime.")
                    continue
            except Exception:
                pass  
        
        elif df[col].dtype == 'object':
            # Check if column contains only strings
            if all(isinstance(x, str) or pd.isna(x) for x in df[col]):
                df[col] = df[col].astype(str)
                print(f" Column '{col}' converted to string.")
            else:
                print(f" Warning: Column '{col}' contains mixed types.")

        elif df[col].dtype in ['int64', 'float64']:  # Already numeric
            df[col] = pd.to_numeric(df[col], errors='coerce')  # Ensure no incorrect types
            print(f" Column '{col}' remains numeric.")

        else:
            print(f" Column '{col}' is already correct ({df[col].dtype}).")

        # Report missing values after conversion
        if missing_values > 0:
            print(f" Column '{col}' has {missing_values} missing values.")



def clean_data(df):


# Drop duplicate rows
    while True:
        try :
            df_no_duplicates = df.drop_duplicates()
            dropped_count = len(df) - len(df_no_duplicates)
            print(f'{dropped_count} duplicate rows dropped!')
            break  # Exit loop if successful

        except ValueError as e:
            print(f"Error: {e}")

# 1. Impute missing values for numerical columns with mean
    while True:
        try:
            # Ask how many numerical columns the user wants to select for imputing mean
            num_cols_mean = input(f"How many numerical columns would you like to select to impute mean: ")
            if not num_cols_mean.isdigit():
                raise ValueError(f"'{num_cols_mean}' is not a valid number. Please enter a valid number.")
            elif num_cols_mean == '0':
                print("No columns selected to impute mean")
            else:
                num_cols_mean = int(num_cols_mean)
                input_cols_mean = input(f"Please enter {num_cols_mean} numerical column name(s), separated by spaces to impute mean: ").split()
                if len(input_cols_mean) != num_cols_mean:
                    raise ValueError(f"You must enter exactly {num_cols_mean} column names.")
                else:
                    # Validate and impute mean for numerical columns
                    for col in input_cols_mean:
                        if col not in df_no_duplicates.columns:
                            raise ValueError(f"'{col}' is not a valid numerical column.")
                        elif not df_no_duplicates[col].isnull().any():  # Check if the column has missing values
                            print(f"No missing values in '{col}'.") 
                        else: 
                            df_no_duplicates[col].fillna(df_no_duplicates[col].mean(), inplace=True)  # Impute with mean
                            print(f"Missing values in '{col}' filled with mean: {df_no_duplicates[col].mean()}")
           
            break  # Exit loop if successful

        except ValueError as e:
            print(f"Error: {e}")

# 2. Impute missing values for numerical columns with median
    while True:
        try:
            # Ask how many numerical columns the user wants to select for imputing median
            num_cols_median = input(f"How many numerical columns would you like to select to impute median: ")
            if not num_cols_median.isdigit():
                raise ValueError(f"'{num_cols_median}' is not a valid number. Please enter a valid number.")
            elif num_cols_median == '0':
                print("No columns selected to impute median")
            else:
                num_cols_median = int(num_cols_median)
                input_cols_median = input(f"Please enter {num_cols_median} numerical column names, separated by spaces: ").split()
                if len(input_cols_median) != num_cols_median:
                    raise ValueError(f"You must enter exactly {num_cols_median} column names.")
                else:
                    # Validate and impute mean for numerical columns
                    for col in input_cols_median:
                        if col not in df_no_duplicates.columns:
                            raise ValueError(f"'{col}' is not a valid numerical column.")
                        elif not df_no_duplicates[col].isnull().any():  # Check if the column has missing values
                            print(f"No missing values in '{col}'.") 
                        else: 
                            df_no_duplicates[col].fillna(df_no_duplicates[col].median(), inplace=True)  # Impute with median
                            print(f"Missing values in '{col}' filled with median: {df_no_duplicates[col].median()}")

            break  # Exit loop if successful

        except ValueError as e:
            print(f"Error: {e}")            

    # 3. Fill missing values and encode for categorical columns
    while True:
        try:
            # Ask how many categorical columns the user wants to select
            cat_cols = input(f"How many categorical columns would you like to select: ")
            if not cat_cols.isdigit():
                raise ValueError(f"'{cat_cols}' is not a valid number. Please enter a valid number.")
            elif cat_cols == '0':
                print("No columns selected to impute mode")  
            else:
                cat_cols = int(cat_cols)                                         
                input_cols_mode = input(f"Please enter {cat_cols} categorical column names, separated by spaces: ").split()
                if len(input_cols_mode) != cat_cols:
                    raise ValueError(f"You must enter exactly {cat_cols} column names.")
                else:     
                    # Validate and impute mode for categorical columns
                    label_encoder = LabelEncoder()
                    for col in input_cols_mode:
                        if col not in df_no_duplicates.columns:
                            raise ValueError(f"'{col}' is not a valid categorical column.")
                        elif not df_no_duplicates[col].isnull().any():  # Check if the column has missing values
                            print(f"No missing values in '{col}'.")                        
                        else:
                            df_no_duplicates[col].fillna(df_no_duplicates[col].mode()[0], inplace=True)  # Impute with mode
                            print(f"Missing values in '{col}' filled with mode: {df_no_duplicates[col].mode()[0]}")                 
                            # Apply label encoding
                            df_no_duplicates[col] = label_encoder.fit_transform(df_no_duplicates[col])
                            print(f"'{col}' has been label encoded.")
            break  # Exit loop if successful

        except ValueError as e:
            print(f"Error: {e}")

    # 4. Min-Max Scaling
    while True:
        try:
            # Ask how many columns the user wants to select for mix-max normmalisation
            minmax_cols = input(f"How many columns would you like to apply min-max normalisation: ")
            if not minmax_cols.isdigit():
                raise ValueError(f"'{minmax_cols}' is not a valid number. Please enter a valid number.")
            elif minmax_cols == '0':
                print("No columns selected for min-max normalisation")  
            else:
                minmax_cols = int(minmax_cols)                                         
                input_minmax_cols = input(f"Please enter {minmax_cols} column names, separated by spaces: ").split()
                if len(input_minmax_cols) != minmax_cols:
                    raise ValueError(f"You must enter exactly {minmax_cols} column names.")
                else:
                    scaler = MinMaxScaler()
                    df[input_minmax_cols] = scaler.fit_transform(df[input_minmax_cols])     
                    print(f"'Min-Max normlisation has been applied to the following columns {input_minmax_cols}'.")
            break  # Exit loop if successful

        except ValueError as e:
            print(f"Error: {e}")

    # 5. Z-score (Standardization)
    while True:
        try:
            # Ask how many columns the user wants to select for Z-score normmalisation
            z_cols = input(f"How many columns would you like to apply Z-score normalisation: ")
            if not z_cols.isdigit():
                raise ValueError(f"'{z_cols}' is not a valid number. Please enter a valid number.")
            elif z_cols == '0':
                print("No columns selected for Z-score normalisation")  
            else:
                z_cols = int(z_cols)                                         
                input_z_cols = input(f"Please enter {z_cols} column names, separated by spaces: ").split()
                if len(input_z_cols) != z_cols:
                    raise ValueError(f"You must enter exactly {z_cols} column names.")
                else:
                    scaler = StandardScaler()
                    df[input_z_cols] = scaler.fit_transform(df[input_z_cols])     
                    print(f"'Z-score (Standardization) normlisation has been applied to the following columns {input_z_cols}'.")
            break  # Exit loop if successful

        except ValueError as e:
            print(f"Error: {e}")


    return df

def load_file(file_path):

    while True:
        try:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f'The file path : "{file_path}" does not exist.') 
            else:
                ext = os.path.splitext(file_path)[-1].lower()  # Get the file extension    
                if ext == ".csv":
                    return pd.read_csv(file_path) 
                elif ext == ".txt":
                    return pd.read_csv(file_path,sep=',')                 
                elif ext in [".xls", ".xlsx"]:
                    return pd.read_excel(file_path, engine="openpyxl")
                elif ext == ".json":
                    return pd.read_json(file_path)
                elif ext == ".parquet":
                    return pd.read_parquet(file_path) 
                elif ext == ".html":
                    return pd.read_html(file_path)[0]  # Reads first table from HTML 
                elif ext == ".xml":
                    return pd.read_xml(file_path)
                elif ext == ".zip":
                    return pd.read_csv(file_path, compression="zip")  # Assumes ZIP contains a CSV
                else:
                    raise ValueError(f"Unsupported file format: {ext}. Please try CSV, Excel, JSON, Parquet, HTML, XML, or ZIP.")
        
        except FileNotFoundError as e:
            print(f"Error: {e}")
            file_path = input("Please enter the correct file path: ")
   

if __name__ == "__main__":
    # Set up argument parsing   
    parser = argparse.ArgumentParser(description="Clean and drop specified columns from a DataFrame")
    parser.add_argument(
    "--columns", 
    required=False, 
    help="Comma-separated column names to drop, e.g., 'Age,City'"
)
    parser.add_argument(
    "--input", 
    required=True, 
    help="Path to the data file to clean"
)
    parser.add_argument(
    "--output", 
    required=True, 
    help="Path to save the cleaned data file"
)
   
    # Parse arguments
    args = parser.parse_args()

    # Load the DataFrame
    print(f"Attempting to load file from: {args.input}")
    file_path = f"{args.input}"
    df = load_file(file_path)
    profile = ProfileReport(df,explorative=True)
    profile.to_file("report.html")

    while True:
        
        if not args.columns:
            print("No columns dropped")
            break
        else:
            # Convert comma-separated string to a list of columns to drop. strip to remove whitespace after
            columns_to_drop = [col.strip() for col in args.columns.split(",")]
        
            # Check if all columns exist in the DataFrame
            invalid_columns = [col for col in columns_to_drop if col not in df.columns]
            if invalid_columns:
                print(f"The following columns do not exist in the DataFrame: {', '.join(invalid_columns)}")
                args.columns = input("Please re-enter the correct comma-separated columns to drop: ")
            else:
             # Drop valid columns
                df = df.drop(columns=columns_to_drop)
                print(f"Successfully dropped columns: {', '.join(columns_to_drop)}")
                break

    # Clean the DataFrame (impute missing values)
    df = clean_data(df)

    # Save the cleaned DataFrame to the specified output path
    df.to_csv(args.output, index=False)
    
    print(f"Cleaned data saved to {args.output}")
