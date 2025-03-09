import pandas as pd
import argparse
import numpy as np
import sys
from ydata_profiling import ProfileReport
import os
import openpyxl
from io import StringIO
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import base64

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
                    return pd.read_excel(file_path, engine="openpyxl", header=0)
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

def data_summary(df):

    # Generate the HTML content from df.head() and df.describe()
    html_head = df.head().to_html()

    html_describe = df.describe().to_html()

    # Generate Data Information from df.info
    buffer = StringIO() # creates in-mmeory buffer
    df.info(buf=buffer)
    info_str = buffer.getvalue()

    # Generate Unique Values in Columns
    unique_list=[]
    for i in df.columns:
        unique_val = df[i].unique() 
        unique_count = df[i].nunique()  # number of unique values
        unique_list.append({
        'Column': i,
        'Unique Values': unique_val,
        'Number of Unique Values': unique_count
    })
    unique_df = pd.DataFrame(unique_list)
    unique_df_html = unique_df.to_html() 

    # Count total duplicate rows
    duplicate_count = df.duplicated().sum()

    # Find duplicates in DF
    duplicates = df[df.duplicated()]
    duplicates_df_html = duplicates.to_html()
    
    # Generate NULL Values in Columns
    null_list=[]
    for i in df.columns:
        i_null = len(df[pd.isnull(df[i])])
        null_list.append({
            'Column': i,
            'Number of NULLs': i_null
        })
    null_df = pd.DataFrame(null_list)
    null_df_html = null_df.to_html()


    while True :
        try :
            # Step 1: Calculate Q1, Q3, and IQR
            col_outlier = input("How many columns are you investigating for outliers: ")
            if not col_outlier.isdigit():
                raise ValueError(f"'{col_outlier}' is not a valid number. Please enter a valid number.")
            elif col_outlier == '0':
                print("No columns selected for outliers")
            else:
                col_outlier = int(col_outlier)
                input_col_outlier = input(f"Please enter {col_outlier} column name(s), separated by spaces for outliers: ").split()
                if len(input_col_outlier) != col_outlier:
                    raise ValueError(f"You must enter exactly {col_outlier} column names.")
                else:
                    outlier_col_list = [] 
                    for col in input_col_outlier:
                        if col not in df.columns:
                            raise ValueError(f"'{col}' is not found in table.")
                        else:                                              
                            Q1 = df[col].quantile(0.25)
                            Q3 = df[col].quantile(0.75)
                            IQR = Q3 - Q1
                            lower_bound = Q1 - 1.5 * IQR
                            upper_bound = Q3 + 1.5 * IQR
                            outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
                            outlier_col_list.append({
                                'Column': col,
                                'Q1' : Q1,
                                'Q3' : Q3,
                                'IQR' : IQR,
                                'Lower bound' : lower_bound,
                                'Upper bound' : upper_bound,
                                'Outliers' : outliers
                            })
                            # Step 3: Plot Boxplot for Outliers and Save it as an Image
                            plt.figure(figsize=(6, 4))
                            sns.boxplot(x=df[col], color='salmon')
                            plt.title(f"Outliers in {col}", fontsize=14)
                            plt.xlabel(col)

                            # Save the plot to a BytesIO object (in-memory image)
                            img_buffer = BytesIO()
                            plt.savefig(img_buffer, format='png')
                            img_buffer.seek(0)

                            # Convert the image to base64 encoding for embedding in HTML
                            img_base64 = base64.b64encode(img_buffer.getvalue()).decode('utf-8')
                            img_buffer.close()

                    outliers_df = pd.DataFrame(outlier_col_list)
                    outliers_df_html = outliers_df.to_html()
                                       

            break
        except ValueError as e:
            print(f"Error: {e}")

    while True :
        try :
            # Step 1: Calculate Z-Score
            col_Z_score = input("How many columns are you computing for Z-Score: ")
            if not col_Z_score.isdigit():
                raise ValueError(f"'{col_Z_score}' is not a valid number. Please enter a valid number.")
            elif col_Z_score == '0':
                print("No columns selected for outliers")
            else:
                col_Z_score = int(col_Z_score)
                input_col_Z = input(f"Please enter {col_Z_score} column name(s), separated by spaces for outliers: ").split()
                if len(input_col_Z) != col_Z_score:
                    raise ValueError(f"You must enter exactly {col_Z_score} column names.")
                else:
                    col_z_list = []
                    for col in input_col_Z:
                        if col not in df.columns:
                            raise ValueError(f"'{col}' is not found in table.")
                        else:
                            mean = df[col].mean()
                            std_dev = df[col].std()                           
                            z_scores = (df[col] - mean) / std_dev 
                            col_z_list.append({
                                'Column': col,
                                'Mean' : mean,
                                'Standard Deviation' : std_dev,
                                'Z-score' : z_scores.tolist()
                            })
                    outliers_z_df = pd.DataFrame(col_z_list)
                    outliers_z_df_html = outliers_z_df.to_html()
                                       

            break
        except ValueError as e:
            print(f"Error: {e}")


    # Create a new report (HTML file)
    with open(f"{args.output}", "w") as file:
        file.write("<html><head><title>Data Report</title></head><body>\n")  # Add HTML structure
        file.write("<h2>First 5 Rows of Data</h2>\n")  # Heading
        file.write(html_head)  # Insert the DataFrame as HTML table
        file.write("<br />")
        file.write("<hr>\n")  # Add a separator      
        file.write("<h2>DataFrame Information</h2>\n")
        file.write("<pre>" + info_str + "</pre>\n")  # Use <pre> tag for formatted text
        file.write("<br />")
        file.write("<hr>\n")
        file.write("<h2>Number of columns</h2>\n")
        file.write(str(len(df.columns)))
        file.write("<br />")
        file.write("<hr>\n")
        file.write("<h2>Number of records</h2>\n")
        file.write(str(len(df)))
        file.write("<br />")
        file.write("<hr>\n")
        file.write("<h2>Summary Statistics for DataFrame</h2>\n")
        file.write(html_describe)
        file.write("<br />")
        file.write("<hr>\n")
        file.write("<h2>Unique Values in Columns</h2>\n")
        file.write(unique_df_html)
        file.write("<br />")
        file.write("<hr>\n")
        file.write("<h2>Number of Duplicates in Columns</h2>\n")
        file.write(str(duplicate_count))
        file.write("<br />")
        file.write("<h2>Duplicates in Columns</h2>\n")
        file.write(duplicates_df_html)
        file.write("<br />")
        file.write("<hr>\n")        
        file.write("<h2>NULL Values in Columns</h2>\n")
        file.write(null_df_html)
        file.write("<br />")
        file.write("<h2>Outliers in Columns</h2>\n")
        file.write(outliers_df_html)
        file.write("<h3>Boxplot of UnitPrice (with Outliers):</h3>")
        file.write(f"<img src='data:image/png;base64,{img_base64}' alt='Boxplot' />\n")        
        file.write("<hr>\n")       
        file.write("<br />")
        file.write("<hr>\n")
        file.write(outliers_z_df_html)
        file.write("<br />")                
        file.write("<p><b>End of report</b></p>\n")
        file.write("</body></html>")  # Close the HTML tags

    print("New HTML report created successfully!")

    return args.output



if __name__ == "__main__":
    # Set up argument parsing   
    parser = argparse.ArgumentParser(description="Clean and drop specified columns from a DataFrame")

    parser.add_argument(
    "--input", 
    required=True, 
    help="Path to the data file for insights"
)
    parser.add_argument(
    "--output", 
    required=True, 
    help="Path to save the insights"
)
   
    # Parse arguments
    args = parser.parse_args()

    # Load the DataFrame
    print(f"Attempting to load file from: {args.input}")
    file_path = f"{args.input}"
    df = load_file(file_path)
    profile = ProfileReport(df,explorative=True)
    profile.to_file("report.html")

    # Insights
    df_info = data_summary(df)

    # Save the insights to the specified output path
    print(df_info)
    print(f"Data insights saved to {args.output}")


def data_summary_wout_args(df):

    # Generate the HTML content from df.head() and df.describe()
    html_head = df.head().to_html()

    html_describe = df.describe().to_html()

    # Generate Data Information from df.info
    buffer = StringIO() # creates in-mmeory buffer
    df.info(buf=buffer)
    info_str = buffer.getvalue()

    # Generate Unique Values in Columns
    unique_list=[]
    for i in df.columns:
        unique_val = df[i].unique() 
        unique_count = df[i].nunique()  # number of unique values
        unique_list.append({
        'Column': i,
        'Unique Values': unique_val,
        'Number of Unique Values': unique_count
    })
    unique_df = pd.DataFrame(unique_list)
    unique_df_html = unique_df.to_html()

    # Count total duplicate rows
    duplicate_count = df.duplicated().sum()

    # Find duplicates in DF
    duplicates = df[df.duplicated()]
    duplicates_df_html = duplicates.to_html() 

    # Generate NULL Values in Columns
    null_list=[]
    for i in df.columns:
        i_null = len(df[pd.isnull(df[i])])
        null_list.append({
            'Column': i,
            'Number of NULLs': i_null
        })
    null_df = pd.DataFrame(null_list)
    null_df_html = null_df.to_html()

    while True :
        try :
            # Step 1: Calculate Q1, Q3, and IQR
            col_outlier = input("How many columns are you investigating for outliers: ")
            if not col_outlier.isdigit():
                raise ValueError(f"'{col_outlier}' is not a valid number. Please enter a valid number.")
            elif col_outlier == '0':
                print("No columns selected for outliers")
            else:
                col_outlier = int(col_outlier)
                input_col_outlier = input(f"Please enter {col_outlier} column name(s), separated by spaces for outliers: ").split()
                if len(input_col_outlier) != col_outlier:
                    raise ValueError(f"You must enter exactly {col_outlier} column names.")
                else:
                    outlier_col_list=[]
                    for col in input_col_outlier:
                        if col not in df.columns:
                            raise ValueError(f"'{col}' is not found in table.")
                        else:                           
                            Q1 = df[col].quantile(0.25)
                            Q3 = df[col].quantile(0.75)
                            IQR = Q3 - Q1
                            lower_bound = Q1 - 1.5 * IQR
                            upper_bound = Q3 + 1.5 * IQR
                            outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
                            outlier_col_list.append({
                                'Column': col,
                                'Q1' : Q1,
                                'Q3' : Q3,
                                'IQR' : IQR,
                                'Lower bound' : lower_bound,
                                'Upper bound' : upper_bound,
                                'Outliers' : outliers
                            })

                            # Step 3: Plot Boxplot for Outliers and Save it as an Image
                            plt.figure(figsize=(6, 4))
                            sns.boxplot(x=df[col], color='salmon')
                            plt.title(f"Outliers in {col}", fontsize=14)
                            plt.xlabel(col)

                            # Save the plot to a BytesIO object (in-memory image)
                            img_buffer = BytesIO()
                            plt.savefig(img_buffer, format='png')
                            img_buffer.seek(0)

                            # Convert the image to base64 encoding for embedding in HTML
                            img_base64 = base64.b64encode(img_buffer.getvalue()).decode('utf-8')
                            img_buffer.close()

                    outliers_df = pd.DataFrame(outlier_col_list)
                    outliers_df_html = outliers_df.to_html()
                                       

            break
        except ValueError as e:
            print(f"Error: {e}")

    while True :
        try :
            # Step 1: Calculate Z-Score
            col_Z_score = input("How many columns are you computing for Z-Score: ")
            if not col_Z_score.isdigit():
                raise ValueError(f"'{col_Z_score}' is not a valid number. Please enter a valid number.")
            elif col_Z_score == '0':
                print("No columns selected for outliers")
            else:
                col_Z_score = int(col_Z_score)
                input_col_Z = input(f"Please enter {col_Z_score} column name(s), separated by spaces for outliers: ").split()
                if len(input_col_Z) != col_Z_score:
                    raise ValueError(f"You must enter exactly {col_Z_score} column names.")
                else:
                    col_z_list = []
                    for col in input_col_Z:
                        if col not in df.columns:
                            raise ValueError(f"'{col}' is not found in table.")
                        else:
                            mean = df[col].mean()
                            std_dev = df[col].std()                           
                            z_scores = (df[col] - mean) / std_dev 
                            col_z_list.append({
                                'Column': col,
                                'Mean' : mean,
                                'Standard Deviation' : std_dev,
                                'Z-Score' : z_scores.tolist()
                            })
                    outliers_z_df = pd.DataFrame(col_z_list)
                    outliers_z_df_html = outliers_z_df.to_html()
                                       

            break
        except ValueError as e:
            print(f"Error: {e}")

    user_output = input("Type in report name to be generated followed by .html : ")
    # Create a new report (HTML file)
    with open(f"{user_output}", "w") as file:
        file.write("<html><head><title>Data Report</title></head><body>\n")  # Add HTML structure
        file.write("<h2>First 5 Rows of Data</h2>\n")  # Heading
        file.write(html_head)  # Insert the DataFrame as HTML table
        file.write("<br />")
        file.write("<hr>\n")  # Add a separator      
        file.write("<h2>DataFrame Information</h2>\n")
        file.write("<pre>" + info_str + "</pre>\n")  # Use <pre> tag for formatted text
        file.write("<br />")
        file.write("<hr>\n")
        file.write("<h2>Number of columns</h2>\n")
        file.write(str(len(df.columns)))
        file.write("<br />")
        file.write("<hr>\n")
        file.write("<h2>Number of records</h2>\n")
        file.write(str(len(df)))
        file.write("<br />")
        file.write("<hr>\n")
        file.write("<h2>Summary Statistics for DataFrame</h2>\n")
        file.write(html_describe)
        file.write("<br />")
        file.write("<hr>\n")
        file.write("<h2>Unique Values in Columns</h2>\n")
        file.write(unique_df_html)
        file.write("<br />")
        file.write("<hr>\n")
        file.write("<h2>Number of Duplicates in Columns</h2>\n")
        file.write(str(duplicate_count))
        file.write("<br />")
        file.write("<h2>Duplicates in Columns</h2>\n")
        file.write(duplicates_df_html)
        file.write("<br />")
        file.write("<hr>\n")
        file.write("<h2>NULL Values in Columns</h2>\n")
        file.write(null_df_html)
        file.write("<br />")
        file.write("<h2>Outliers in Columns</h2>\n")
        file.write(outliers_df_html)   
        file.write("<h3>Boxplot of UnitPrice (with Outliers):</h3>")
        file.write(f"<img src='data:image/png;base64,{img_base64}' alt='Boxplot' />\n")        
        file.write("<hr>\n")
        file.write("<br />")
        file.write(outliers_z_df_html)
        file.write("<br />")                   
        file.write("<p><b>End of report</b></p>\n")
        file.write("</body></html>")  # Close the HTML tags

    print("New HTML report created successfully!")

    return user_output
