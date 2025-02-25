import pandas as pd
import argparse
import numpy as np
import sys
from ydata_profiling import ProfileReport
import os
import openpyxl
from io import StringIO

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
        file.write("<h2>NULL Values in Columns</h2>\n")
        file.write(null_df_html)
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
        file.write("<h2>NULL Values in Columns</h2>\n")
        file.write(null_df_html)
        file.write("<br />")
        file.write("<p><b>End of report</b></p>\n")
        file.write("</body></html>")  # Close the HTML tags

    print("New HTML report created successfully!")

    return user_output
