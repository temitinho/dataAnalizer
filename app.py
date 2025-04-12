import streamlit as st
import pandas as pd
import numpy as np
import io
import sqlite3
from sklearn.datasets import load_iris, fetch_california_housing
import datetime
import plotly.express as px
import plotly.graph_objects as go
import statsmodels.api as sm
from statsmodels.formula.api import ols
import matplotlib.pyplot as plt
import seaborn as sns

if "df" not in st.session_state:
    st.session_state.df = pd.DataFrame() 
#df = pd.DataFrame()
# Set page configuration
st.set_page_config(
    page_title="Interactive Data Analysis Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
) 

@st.cache_data
def cached_data_processing(file_content, file_extension, **kwargs):
    """Cache data processing to avoid reloading the same file multiple times"""
    if file_extension == "csv":
        return pd.read_csv(io.BytesIO(file_content), **kwargs)
    elif file_extension in ["xlsx", "xls"]:
        return pd.read_excel(io.BytesIO(file_content), **kwargs)
    elif file_extension == "json":
        return pd.read_json(io.BytesIO(file_content), **kwargs)
    elif file_extension == "parquet":
        return pd.read_parquet(io.BytesIO(file_content))
    elif file_extension in ["db", "sqlite"]:
        # For SQLite files, we need a table name
        table_name = kwargs.get("table_name")
        if not table_name:
            raise ValueError("Table name must be provided for SQLite files")
            
        # Create a temporary file to store the database
        temp_db_path = "temp_cached_database.db"
        with open(temp_db_path, "wb") as f:
            f.write(file_content)
            
        # Connect to the database and read the specified table
        conn = sqlite3.connect(temp_db_path)
        df = pd.read_sql(f"SELECT * FROM {table_name}", conn)
        conn.close()
        
        return df
    return None
def process_uploaded_file(uploaded_file):
    # Get file extension
    file_extension = uploaded_file.name.split(".")[-1].lower()
    
    # Read file content
    file_content = uploaded_file.getvalue()
    
    # Process based on file type
    if file_extension == "csv":
        # Allow user to specify CSV parameters
        delimiter = st.selectbox("Select delimiter", [",", ";", "\\t", "|", " "], index=0)
        encoding = st.selectbox("Select encoding", ["utf-8", "latin-1", "iso-8859-1"], index=0)
        
        # Convert tab delimiter for proper reading
        if delimiter == "\\t":
            delimiter = "\t"
            
        # Use cached function to read CSV file
        df = cached_data_processing(
            file_content, 
            file_extension, 
            delimiter=delimiter, 
            encoding=encoding
        )
        
    elif file_extension in ["xlsx", "xls"]:
        # For Excel files, allow sheet selection
        # We need to save the file temporarily to get sheet names
        temp_excel_path = "temp_excel_file." + file_extension
        with open(temp_excel_path, "wb") as f:
            f.write(file_content)
            
        xls = pd.ExcelFile(temp_excel_path)
        sheet_name = st.selectbox("Select sheet", xls.sheet_names)
        
        # Use cached function to read Excel file
        df = cached_data_processing(file_content, file_extension, sheet_name=sheet_name)
        
    elif file_extension == "json":
        # For JSON files, allow different orientations
        orientation = st.selectbox(
            "Select JSON orientation", 
            ["records", "columns", "index", "split", "table"],
            index=0,
            help="How is your JSON data structured?"
        )
        
        # Use cached function to read JSON file
        df = cached_data_processing(file_content, file_extension, orient=orientation)
        
    elif file_extension == "parquet":
        # Use cached function to read parquet file
        df = cached_data_processing(file_content, file_extension)
        
    elif file_extension in ["db", "sqlite"]:
        # SQLite databases need special handling and can't use the cached function as-is
        # Save the uploaded file to a temporary location
        temp_db_path = "temp_database.db"
        with open(temp_db_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
            
        # Connect to the database
        conn = sqlite3.connect(temp_db_path)
        
        # Get list of tables
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = [table[0] for table in cursor.fetchall()]
        
        if not tables:
            st.error("No tables found in the database.")
            return None
            
        # Let user select a table
        selected_table = st.selectbox("Select a table", tables)
        
        # Read the selected table
        df = pd.read_sql(f"SELECT * FROM {selected_table}", conn)
        conn.close()
    elif file_extension in ["db", "sqlite"]:
        # Get the file content
        file_content = uploaded_file.getbuffer()
        
        # Save temporarily to get table list
        temp_db_path = "temp_database.db"
        with open(temp_db_path, "wb") as f:
            f.write(file_content)
            
        # Connect to the database
        conn = sqlite3.connect(temp_db_path)
        
        # Get list of tables
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = [table[0] for table in cursor.fetchall()]
        conn.close()
        
        if not tables:
            st.error("No tables found in the database.")
            return None
            
        # Let user select a table
        selected_table = st.selectbox("Select a table", tables)
        
        # Use the cached function to read the selected table
        df = cached_data_processing(file_content, file_extension, table_name=selected_table)
    else:
        st.error(f"Unsupported file format: {file_extension}")
        return None
    
    # Display data preview
    st.subheader("Data Preview")
    st.dataframe(df.head())
    
    # Display data information
    st.subheader("Data Information")
    buffer = io.StringIO()
    df.info(buf=buffer)
    info_str = buffer.getvalue()
    st.text(info_str)
    
    # Display basic statistics
    st.subheader("Basic Statistics")
    st.write(df.describe())
    
    return df
# Main data loading function
def load_data_page():
    st.title("Interactive Data Analysis Dashboard")
    
    st.header("1. Data Loading")
    
    data_source = st.radio(
        "Select data source",
        ["Upload a file", "Use sample data"],
        horizontal=True,
        key="data_source_radio"
    )
    
    global df
    df = None
    
    if data_source == "Upload a file":
        # File upload widget
        uploaded_file = st.file_uploader(
            "Upload your data file", 
            type=["csv", "xlsx", "xls", "json", "parquet", "db", "sqlite"],
            help="Upload a data file to begin analysis"
        )
        
        if uploaded_file is not None:
            # Display file details
            file_details = {
                "Filename": uploaded_file.name,
                "File size": f"{uploaded_file.size / 1024:.2f} KB",
                "File type": uploaded_file.type
            }
            
            st.write("File Details:")
            for key, value in file_details.items():
                st.write(f"- {key}: {value}")
            
            # Process the uploaded file based on its extension
            try:
                df = process_uploaded_file(uploaded_file)
            except Exception as e:
                st.error(f"Error loading file: {e}")
    else:
        # Sample data option
        sample_option = st.selectbox(
            "Select sample dataset",
            ["Iris Flower Dataset", "California Housing", "Diabetes Dataset", "Boston Housing", "Titanic Passengers"],
            index=0,
        )
        
        df = load_sample_data(sample_option)
    
    return df


def load_sample_data(sample_option):
    if sample_option == "Iris Flower Dataset":
        from sklearn.datasets import load_iris
        data = load_iris()
        df = pd.DataFrame(data.data, columns=data.feature_names)
        df['target'] = data.target
        
    elif sample_option == "Titanic Passengers":
        url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
        df = pd.read_csv(url) 
        
    elif sample_option == "Boston Housing":
        from sklearn.datasets import load_boston
        data = load_boston()
        df = pd.DataFrame(data.data, columns=data.feature_names)
        df['PRICE'] = data.target
    elif sample_option == "California Housing":
        data = fetch_california_housing()
        df = pd.DataFrame(data.data, columns=data.feature_names)
        df['target'] = data.target 
    elif sample_option == "Diabetes Dataset":   
        from sklearn.datasets import load_diabetes
        data = load_diabetes()
        df = pd.DataFrame(data.data, columns=data.feature_names)
        df['target'] = data.target
    else:
        st.error("Invalid sample dataset selected.")
        return None 
    
    # Display data preview
    st.subheader("Data Preview")
    st.dataframe(df.head())
    
    # Display data information
    st.subheader("Data Information")
    buffer = io.StringIO()
    df.info(buf=buffer)
    info_str = buffer.getvalue()
    st.text(info_str)
    
    # Display basic statistics
    st.subheader("Basic Statistics")
    st.write(df.describe())
    
    return df

def columns_operations(df): 
    st.title("Column Operations")
    
    st.write("### Preview of data")
    st.dataframe(df.head())
    
    # Get numeric columns - update this dynamically to include newly added columns
    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if len(numeric_columns) < 2:
        st.error("The file must contain at least 2 numeric columns for operations.")
        return
    
    # Add operation
    st.write("### Add columns")
    add_cols = st.multiselect("Select columns to add", numeric_columns, max_selections=None)
    add_result_name = st.text_input("Name for addition result column", "sum_result")
    
    if add_cols and len(add_cols) >= 2 and st.button("Add Columns"):
        df[add_result_name] = df[add_cols].sum(axis=1)
        st.session_state.df = df  # Update the stored dataframe
        st.success(f"Created new column '{add_result_name}' by adding {len(add_cols)} columns")
        
        # Get updated numeric columns after adding the new column
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        st.dataframe(df.head())
    
    # Subtraction operation
    st.write("### Subtract columns")
    # Get the latest numeric columns
    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    minuend = st.selectbox("Select column to subtract from (minuend)", numeric_columns)
    subtrahend = st.selectbox("Select column to subtract (subtrahend)", 
                            [col for col in numeric_columns if col != minuend] if len(numeric_columns) > 1 else numeric_columns)
    sub_result_name = st.text_input("Name for subtraction result column", "difference_result")
    
    if minuend and subtrahend and st.button("Subtract Columns"):
        df[sub_result_name] = df[minuend] - df[subtrahend]
        st.session_state.df = df  # Update the stored dataframe
        st.success(f"Created new column '{sub_result_name}' by subtracting {subtrahend} from {minuend}")
        st.dataframe(df.head())
    
    # Multiply operation
    st.write("### Multiply columns")
    # Get updated numeric columns
    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    mult_cols = st.multiselect("Select columns to multiply", numeric_columns, max_selections=None)
    mult_result_name = st.text_input("Name for multiplication result column", "product_result")
    
    if mult_cols and len(mult_cols) >= 2 and st.button("Multiply Columns"):
        df[mult_result_name] = df[mult_cols].prod(axis=1)
        st.session_state.df = df  # Update the stored dataframe
        st.success(f"Created new column '{mult_result_name}' by multiplying {len(mult_cols)} columns")
        st.dataframe(df.head())
    
    # Division operation
    st.write("### Divide columns")
    # Get updated numeric columns
    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    col1 = st.selectbox("Select numerator column", numeric_columns)
    col2 = st.selectbox("Select denominator column", 
                        [col for col in numeric_columns if col != col1] if len(numeric_columns) > 1 else numeric_columns)
    div_result_name = st.text_input("Name for division result column", "division_result")
    
    if col1 and col2 and st.button("Divide Columns"):
        # Handle division by zero gracefully
        df[div_result_name] = df[col1] / df[col2].replace(0, np.nan)
        st.session_state.df = df  # Update the stored dataframe
        st.success(f"Created new column '{div_result_name}' by dividing {col1} by {col2}")
        st.dataframe(df.head())
    
    # Filter Data
    st.write("### Filter Data")
    # Get all columns for filtering
    all_columns = df.columns.tolist()
    filter_col = st.selectbox("Select column to filter", all_columns)
    
    # Determine filter type based on column data type
    if filter_col in numeric_columns:
        st.write(f"Filtering numeric column: {filter_col}")
        min_val = float(df[filter_col].min())
        max_val = float(df[filter_col].max())
        
        filter_type = st.radio("Filter type", ["Range", "Greater than", "Less than", "Equal to"])
        
        if filter_type == "Range":
            filter_range = st.slider(f"Select range for {filter_col}", 
                                    min_value=min_val, 
                                    max_value=max_val,
                                    value=(min_val, max_val))
            
            if st.button("Apply Range Filter"):
                filtered_df = df[(df[filter_col] >= filter_range[0]) & (df[filter_col] <= filter_range[1])]
                st.session_state.df = filtered_df
                st.success(f"Data filtered where {filter_col} is between {filter_range[0]} and {filter_range[1]}")
                st.write(f"Filtered data: {len(filtered_df)} rows (from {len(df)} rows)")
                st.dataframe(filtered_df.head(10))
        
        elif filter_type == "Greater than":
            threshold = st.slider(f"Select minimum value for {filter_col}", 
                                min_value=min_val, 
                                max_value=max_val,
                                value=min_val)
            
            if st.button("Apply Greater Than Filter"):
                filtered_df = df[df[filter_col] > threshold]
                st.session_state.df = filtered_df
                st.success(f"Data filtered where {filter_col} > {threshold}")
                st.write(f"Filtered data: {len(filtered_df)} rows (from {len(df)} rows)")
                st.dataframe(filtered_df.head(10))
        
        elif filter_type == "Less than":
            threshold = st.slider(f"Select maximum value for {filter_col}", 
                                min_value=min_val, 
                                max_value=max_val,
                                value=max_val)
            
            if st.button("Apply Less Than Filter"):
                filtered_df = df[df[filter_col] < threshold]
                st.session_state.df = filtered_df
                st.success(f"Data filtered where {filter_col} < {threshold}")
                st.write(f"Filtered data: {len(filtered_df)} rows (from {len(df)} rows)")
                st.dataframe(filtered_df.head(10))
        
        elif filter_type == "Equal to":
            # For exact match with numeric values, get unique values
            unique_values = sorted(df[filter_col].unique().tolist())
            if len(unique_values) > 20:  # If too many values, use a slider instead
                selected_value = st.slider(f"Select value for {filter_col}", 
                                        min_value=min_val, 
                                        max_value=max_val)
            else:
                selected_value = st.selectbox(f"Select value for {filter_col}", unique_values)
            
            if st.button("Apply Equal To Filter"):
                filtered_df = df[df[filter_col] == selected_value]
                st.session_state.df = filtered_df
                st.success(f"Data filtered where {filter_col} = {selected_value}")
                st.write(f"Filtered data: {len(filtered_df)} rows (from {len(df)} rows)")
                st.dataframe(filtered_df.head(10))
    
    else:  # Non-numeric column
        st.write(f"Filtering categorical column: {filter_col}")
        unique_values = df[filter_col].unique().tolist()
        selected_values = st.multiselect(f"Select values to keep from {filter_col}", unique_values)
        
        if selected_values and st.button("Apply Category Filter"):
            filtered_df = df[df[filter_col].isin(selected_values)]
            st.session_state.df = filtered_df
            st.success(f"Data filtered where {filter_col} is in {', '.join(str(v) for v in selected_values)}")
            st.write(f"Filtered data: {len(filtered_df)} rows (from {len(df)} rows)")
            st.dataframe(filtered_df.head(10))
    
    # Reset filters button
    if st.button("Reset All Filters"):
        # Restore original data but keep calculated columns
        original_df = st.session_state.original_df.copy()
        for col in df.columns:
            if col not in original_df.columns:
                original_df[col] = df[col]
        st.session_state.df = original_df
        st.success("All filters reset to original data (calculated columns preserved)")
        st.dataframe(original_df.head())    
    
    # Sort Data - at the end
    st.write("### Sort Data")
    # Get final updated numeric columns
    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    sort_col = st.selectbox("Select column to sort by", numeric_columns)
    sort_order = st.radio("Sort order", ["Ascending", "Descending"])
    
    if sort_col and st.button("Sort Data"):
        ascending = sort_order == "Ascending"
        df = df.sort_values(by=sort_col, ascending=ascending)
        st.session_state.df = df  # Update the stored dataframe
        st.success(f"Data sorted by '{sort_col}' in {sort_order.lower()} order")
        st.dataframe(df.head(10))
    
    # Download the modified CSV
    st.write("### Download Data")
    if st.button("Download Modified CSV"):
        csv = df.to_csv(index=False)
        st.download_button(
            label="Download CSV",
            data=csv,
            file_name="modified_data.csv",
            mime="text/csv"
        )
def delete_columns(df):
    st.subheader("Delete Columns")
    #df = st.session_state.df 
    if df is not None:
        # Get list of columns
        columns = df.columns.tolist()
        
        # Let user select columns to delete
        columns_to_delete = st.multiselect(
            "Select columns to delete",
            options=columns,
            help="Select one or more columns to remove from the dataset"
        )
        
        # Create a button to confirm deletion
        if columns_to_delete and st.button("Delete Selected Columns"):
            # Create a copy of the dataframe with selected columns removed
            df_modified = df.drop(columns=columns_to_delete)
            
            # Show success message
            st.success(f"Deleted {len(columns_to_delete)} columns: {', '.join(columns_to_delete)}")
            
            # Display the modified dataframe
            st.subheader("Modified Dataset")
            st.dataframe(df_modified.head())
            
            # Return the modified dataframe
            return df_modified
    
    # If no columns were deleted, return the original dataframe
    return df
def change_data_types(df):
    st.subheader("Change Data Types")
    
    if df is not None:
        # Get list of columns
        columns = df.columns.tolist()
        
        # Create columns for layout
        col1, col2 = st.columns(2)
        
        with col1:
            # Let user select a column
            selected_column = st.selectbox(
                "Select column to change data type",
                options=columns,
                help="Select a column to change its data type"
            )
        
        with col2:
            # Get current data type
            current_type = str(df[selected_column].dtype)
            
            # Let user select a new data type
            new_type = st.selectbox(
                f"Change data type (current: {current_type})",
                options=["int64", "float64", "str", "category", "datetime64", "bool"],
                help="Select the new data type for the column"
            )
        
        # Show current values
        st.write(f"Current values (type: {current_type}):")
        st.write(df[selected_column].head())
        
        # Create a button to confirm data type change
        if st.button("Change Data Type"):
            # Create a copy of the dataframe
            df_modified = df.copy()
            
            try:
                # Apply the data type conversion
                if new_type == "int64":
                    df_modified[selected_column] = df_modified[selected_column].astype(int)
                elif new_type == "float64":
                    df_modified[selected_column] = df_modified[selected_column].astype(float)
                elif new_type == "str":
                    df_modified[selected_column] = df_modified[selected_column].astype(str)
                elif new_type == "category":
                    df_modified[selected_column] = df_modified[selected_column].astype('category')
                elif new_type == "datetime64":
                    df_modified[selected_column] = pd.to_datetime(df_modified[selected_column])
                elif new_type == "bool":
                    df_modified[selected_column] = df_modified[selected_column].astype(bool)
                
                # Show success message
                st.success(f"Changed data type of '{selected_column}' from {current_type} to {new_type}")
                
                # Show the modified values
                st.write(f"Modified values (type: {df_modified[selected_column].dtype}):")
                st.write(df_modified[selected_column].head())
                
                # Return the modified dataframe
                return df_modified
            except Exception as e:
                st.error(f"Error changing data type: {e}")
    
    # If no data type was changed, return the original dataframe
    return df
def data_manipulation_page(df):
    st.header("2. Data Manipulation")
    
    st.session_state.df = df
    
    if df is None:
        st.warning("Please load data first.")
        return None
    
    # Create tabs for different manipulation operations
    tab1, tab2, tab3, tab4 = st.tabs(["Delete Columns", "Change Data Types", "Analyze Correlations", "Columns Operations"])
    
    # # Store the current state of the dataframe
    # if "current_df" not in st.session_state:
    #     st.session_state.current_df = df.copy()
    
    # Tab 1: Delete Columns
    with tab1:
        # Get list of columns
        columns = st.session_state.df.columns.tolist()
        
        # Let user select columns to delete
        columns_to_delete = st.multiselect(
            "Select columns to delete",
            options=columns,
            help="Select one or more columns to remove from the dataset"
        )
        
        # Create a button to confirm deletion
        if columns_to_delete and st.button("Delete Selected Columns"):
            # Create a copy of the dataframe with selected columns removed
            st.session_state.df = st.session_state.df.drop(columns=columns_to_delete)
            
            # Show success message
            st.success(f"Deleted {len(columns_to_delete)} columns: {', '.join(columns_to_delete)}")
    
    # Tab 2: Change Data Types
    with tab2:
        # Get list of columns
        columns = st.session_state.df.columns.tolist()
        
        # Create columns for layout
        col1, col2 = st.columns(2)
        
        with col1:
            # Let user select a column
            selected_column = st.selectbox(
                "Select column to change data type",
                options=columns,
                help="Select a column to change its data type"
            )
        
        with col2:
            # Get current data type
            current_type = str(st.session_state.df[selected_column].dtype)
            
            # Let user select a new data type
            new_type = st.selectbox(
                f"Change data type (current: {current_type})",
                options=["int64", "float64", "str", "category", "datetime64", "bool"],
                help="Select the new data type for the column"
            )
        
        # Show current values
        st.write(f"Current values (type: {current_type}):")
        st.write(st.session_state.df[selected_column].head())
        
        # Create a button to confirm data type change
        if st.button("Change Data Type"):
            try:
                # Apply the data type conversion
                if new_type == "int64":
                    st.session_state.df[selected_column] = st.session_state.current_df[selected_column].astype(int)
                elif new_type == "float64":
                    st.session_state.df[selected_column] = st.session_state.current_df[selected_column].astype(float)
                elif new_type == "str":
                    st.session_state.df[selected_column] = st.session_state.current_df[selected_column].astype(str)
                elif new_type == "category":
                    st.session_state.df[selected_column] = st.session_state.current_df[selected_column].astype('category')
                elif new_type == "datetime64":
                    st.session_state.df[selected_column] = pd.to_datetime(st.session_state.current_df[selected_column])
                elif new_type == "bool":
                    st.session_state.df[selected_column] = st.session_state.current_df[selected_column].astype(bool)
                
                # Show success message
                st.success(f"Changed data type of '{selected_column}' from {current_type} to {new_type}")
                
                # Show the modified values
                st.write(f"Modified values (type: {st.session_state.current_df[selected_column].dtype}):")
                st.write(st.session_state.df[selected_column].head())
            except Exception as e:
                st.error(f"Error changing data type: {e}")
    
    # Tab 3: Analyze Correlations
    with tab3:
        # Get list of numeric columns
        numeric_columns = st.session_state.df.select_dtypes(include=['number']).columns.tolist()
        
        if len(numeric_columns) < 2:
            st.warning("Need at least 2 numeric columns to calculate correlations.")
        else:
            # Let user select columns for correlation analysis
            selected_columns = st.multiselect(
                "Select columns for correlation analysis",
                options=numeric_columns,
                default=numeric_columns[:min(5, len(numeric_columns))],
                help="Select two or more numeric columns to analyze their correlations"
            )
            
            if len(selected_columns) < 2:
                st.warning("Please select at least 2 columns for correlation analysis.")
            else:
                # Calculate correlation matrix
                correlation_matrix = st.session_state.df[selected_columns].corr()
                
                # Display correlation matrix
                st.subheader("Correlation Matrix")
                st.dataframe(correlation_matrix.style.background_gradient(cmap='coolwarm'))
                
                # Create a heatmap visualization
                st.subheader("Correlation Heatmap")
                
                import plotly.express as px
                fig = px.imshow(
                    correlation_matrix,
                    text_auto=True,
                    aspect="auto",
                    color_continuous_scale='RdBu_r',
                    title="Correlation Heatmap"
                )
                st.plotly_chart(fig, use_container_width=True)
    with tab4:
        df = st.session_state['dataframe']
        columns = df.columns.tolist()
        columns_operations(df)
        
        
    # Display the current state of the dataframe
    st.subheader("Current Dataset")
    st.dataframe(st.session_state.df.head())
   
    return st.session_state.df

def data_analize_page(df):
    st.header("3. Data Analysis")
    
    if df is None:
        st.warning("Please load data first.")
        return None
    
    # Create tabs for different analysis operations
    tab1, tab2, tab3 = st.tabs(["Analyze Correlations", "Group By", "Value Counts"])
    
    # Tab 1: Analyze Correlations
    with tab1:
        df = analyze_correlations_page(df)
    
    # Tab 2: Group By
    with tab2:
        numeric_columns = df.select_dtypes(include=['number']).columns.tolist()
        categorical_columns = df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        df = analyze_groupby(df, numeric_columns, categorical_columns)
    
    # Tab 3: Value Counts
    with tab3:
        df = st.session_state['dataframe']
        value_counts(df)
    
def analyze_correlations_page(df):
    st.subheader("Analyze Correlations")
    
    if df is not None:
        # Get list of numeric columns
        numeric_columns = df.select_dtypes(include=['number']).columns.tolist()
        
        if len(numeric_columns) < 2:
            st.warning("Need at least 2 numeric columns to calculate correlations.")
            return df
        
        # Let user select columns for correlation analysis
        selected_columns = st.multiselect(
            "Select columns for correlation analysis",
            options=numeric_columns,
            default=numeric_columns[:min(5, len(numeric_columns))],
            help="Select two or more numeric columns to analyze their correlations"
        )
        
        if len(selected_columns) < 2:
            st.warning("Please select at least 2 columns for correlation analysis.")
        else:
            # Calculate correlation matrix
            correlation_matrix = df[selected_columns].corr()
            
            # Display correlation matrix
            st.subheader("Correlation Matrix")
            st.dataframe(correlation_matrix.style.background_gradient(cmap='coolwarm'))
            
            # Create a heatmap visualization
            st.subheader("Correlation Heatmap")
            
            import plotly.express as px
            fig = px.imshow(
                correlation_matrix,
                text_auto=True,
                aspect="auto",
                color_continuous_scale='RdBu_r',
                title="Correlation Heatmap"
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Provide interpretation of correlations
            st.subheader("Correlation Interpretation")
            
            st.write("""
            **Interpretation Guide:**
            - Values close to 1 indicate strong positive correlation (as one variable increases, the other also increases)
            - Values close to -1 indicate strong negative correlation (as one variable increases, the other decreases)
            - Values close to 0 indicate little to no linear correlation
            """)
            
            # Highlight strongest correlations
            st.subheader("Strongest Correlations")
            
            # Get the upper triangle of the correlation matrix (excluding diagonal)
            upper_tri = correlation_matrix.where(np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool))
            
            # Stack the data and sort by absolute correlation
            stacked = upper_tri.stack().reset_index()
            stacked.columns = ['Variable 1', 'Variable 2', 'Correlation']
            stacked['Abs Correlation'] = stacked['Correlation'].abs()
            stacked = stacked.sort_values('Abs Correlation', ascending=False)
            
            # Display top correlations
            if not stacked.empty:
                st.dataframe(stacked.head(10))
                
                # Scatter plot for top correlation
                if len(stacked) > 0:
                    top_var1 = stacked.iloc[0]['Variable 1']
                    top_var2 = stacked.iloc[0]['Variable 2']
                    top_corr = stacked.iloc[0]['Correlation']
                    
                    st.write(f"**Scatter plot for top correlation: {top_var1} vs {top_var2} (r = {top_corr:.4f})**")
                    
                    fig = px.scatter(
                        df, 
                        x=top_var1, 
                        y=top_var2, 
                        trendline="ols",
                        title=f"{top_var1} vs {top_var2} (r = {top_corr:.4f})"
                    )
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.write("No correlations found in the selected columns.")
    
    return df
def analyze_groupby(df, numeric_columns, categorical_columns):
    st.title("Group by Data Analyzer")
    
    if df is None:
        st.warning("Please load data first.")
        return None
    
    columns = df.columns.tolist()
    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()        
    # Allow selecting multiple columns for groupby
    groupby_cols = st.multiselect(
        "Select columns to group by:",
        options=columns,
        default=None
    )
    
    # Select a numeric column for filtering
    if numeric_columns:
        st.subheader("Filter Data")
        filter_col = st.selectbox(
            "Select a numeric column to filter:",
            options=numeric_columns
        )
        
        # Get min and max values for the selected column
        min_val = float(df[filter_col].min())
        max_val = float(df[filter_col].max())
        
        # Slider for threshold value
        threshold = st.slider(
            f"Show values where {filter_col} is greater than:",
            min_value=min_val,
            max_value=max_val,
            value=min_val,
            step=(max_val - min_val) / 100
        )
        
        # Manual input option
        use_manual = st.checkbox("Enter threshold manually")
        if use_manual:
            manual_threshold = st.number_input(
                f"Enter threshold value for {filter_col}:",
                value=float(threshold),
                min_value=min_val,
                max_value=max_val
            )
            threshold = manual_threshold
        
        # Process and display results
        if st.button("Analyze Data"):
            if groupby_cols:
                # Filter data based on threshold
                filtered_df = df[df[filter_col] > threshold]
                
                # Group by selected columns
                grouped = filtered_df.groupby(groupby_cols).agg({
                    filter_col: ['count', 'mean', 'sum', 'min', 'max']
                })
                
                st.subheader("Analysis Results")
                st.write(f"Data grouped by: {', '.join(groupby_cols)}")
                st.write(f"Showing records where {filter_col} > {threshold}")
                st.write(f"Number of records after filtering: {len(filtered_df)}")
                
                # Display grouped data
                st.dataframe(grouped)
                
                # Provide download option for results
                csv = grouped.to_csv()
                st.download_button(
                    label="Download results as CSV",
                    data=csv,
                    file_name="grouped_data_results.csv",
                    mime="text/csv",
                )
                
                # Show basic visualization
                st.subheader("Visualization")
                
                # Only create chart if not too many groups
                if len(grouped) <= 20:
                    chart_data = grouped.reset_index()
                    st.bar_chart(chart_data, x=groupby_cols[0], y=f"({filter_col}, mean)")
                else:
                    st.write("Too many groups to display chart (limited to 20)")
            else:
                st.error("Please select at least one column to group by")
    else:
        st.warning("No numeric columns found in the dataset for filtering operations")
def value_counts(df):   
    st.title("Value Counts Analysis")
    
    if df is None:
        st.warning("Please load data first.")
        return None
   
    # Get column information
    columns = df.columns.tolist()
    
    # Select column for value counts
    selected_column = st.selectbox(
        "Select a column to view value counts:",
        options=columns
    )
    
    # Show value counts
    if selected_column:
        st.subheader(f"Value Counts for: {selected_column}")
        
        # Determine if the column is numeric
        is_numeric = pd.api.types.is_numeric_dtype(df[selected_column])
        
        # Options for display
        display_options = st.radio(
            "Display options:",
            ["Top values", "All values", "Custom range"],
            horizontal=True
        )
        
        if display_options == "Top values":
            top_n = st.slider("Number of top values to show:", 5, 50, 10)
            limit = top_n
        elif display_options == "Custom range":
            start = st.number_input("Start index:", 0, step=1)
            end = st.number_input("End index:", start+10, step=1)
            limit = slice(start, end)
        else:  # All values
            limit = None
        
        # Get value counts
        if is_numeric and st.checkbox("Bin numeric values", value=False):
            bins = st.slider("Number of bins:", 5, 100, 10)
            value_counts = pd.cut(df[selected_column], bins=bins).value_counts()
            st.write(f"Values binned into {bins} groups")
        else:
            # For categorical or unbinned numeric data
            value_counts = df[selected_column].value_counts()
        
        # Apply limits
        if limit is not None:
            if isinstance(limit, int):
                displayed_counts = value_counts.head(limit)
            else:  # slice
                displayed_counts = value_counts.iloc[limit]
        else:
            displayed_counts = value_counts
        
        # Display counts as table
        st.write(f"**Count of unique values: {len(value_counts)}**")
        st.dataframe(displayed_counts.reset_index().rename(
            columns={"index": selected_column, 0: "Count"}
        ))
        
        # Display percentage
        if st.checkbox("Show percentages", value=True):
            percentages = value_counts / len(df) * 100
            if limit is not None:
                if isinstance(limit, int):
                    displayed_percentages = percentages.head(limit)
                else:  # slice
                    displayed_percentages = percentages.iloc[limit]
            else:
                displayed_percentages = percentages
            
            st.dataframe(displayed_percentages.reset_index().rename(
                columns={"index": selected_column, 0: "Percentage (%)"}
            ).round(2))
        
        # Visualization
        st.subheader("Visualization")
        chart_type = st.radio(
            "Chart type:",
            ["Bar chart", "Pie chart", "Horizontal bar chart"],
            horizontal=True
        )
        
        # Create figure and axis
        fig, ax = plt.subplots(figsize=(10, 6))
        
        if chart_type == "Bar chart":
            displayed_counts.plot(kind='bar', ax=ax)
        elif chart_type == "Horizontal bar chart":
            displayed_counts.plot(kind='barh', ax=ax)
        else:  # Pie chart
            if len(displayed_counts) > 15:
                st.warning("Too many categories for a pie chart. Showing top 15.")
                displayed_counts = displayed_counts.head(15)
            displayed_counts.plot(kind='pie', ax=ax, autopct='%1.1f%%')
        
        plt.title(f"Value Counts for {selected_column}")
        plt.tight_layout()
        st.pyplot(fig)
        
        # Download option
        csv = displayed_counts.reset_index().to_csv(index=False)
        st.download_button(
            label="Download value counts as CSV",
            data=csv,
            file_name=f"{selected_column}_value_counts.csv",
            mime="text/csv",
        )
        
    # Multi-column comparison
    st.subheader("Compare Multiple Columns")
    if st.checkbox("Show value counts for multiple columns"):
        multi_columns = st.multiselect(
            "Select columns to compare:",
            options=columns,
            default=[]
        )
        
        if multi_columns:
            max_values = st.slider("Max unique values to display per column:", 5, 30, 10)
            
            # Create multi-column layout
            cols = st.columns(len(multi_columns))
            
            for i, col_name in enumerate(multi_columns):
                with cols[i]:
                    st.write(f"**{col_name}**")
                    col_counts = df[col_name].value_counts().head(max_values)
                    st.dataframe(col_counts.reset_index().rename(
                        columns={"index": col_name, 0: "Count"}
                    ), height=400)
    
            
def create_line_chart(df, numeric_columns, categorical_columns):
    st.subheader("Line Chart")
    
    
    if df is not None:
        # Get list of columns
        columns = df.columns.tolist()
        numeric_columns = df.select_dtypes(include=['number']).columns.tolist()
        
        # Create columns for layout
        col1, col2 = st.columns(2)
        
        with col1:
            # Let user select x-axis column
            x_axis = st.selectbox(
                "Select X-axis column",
                options=columns,
                key="line_x_axis",  # Add unique key
                help="Select the column for the X-axis"
            )
        
        with col2:
            # Let user select y-axis column(s)
            y_axes = st.multiselect(
                "Select Y-axis column(s)",
                options=numeric_columns,
                default=numeric_columns[0] if numeric_columns else None,
                help="Select one or more numeric columns for the Y-axis"
            )
        
        # Additional options
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Chart title
            chart_title = st.text_input(
                "Chart Title",
                value=f"Line Chart: {', '.join(y_axes)} vs {x_axis}" if y_axes else "Line Chart"
            )
        
        with col2:
            # Line mode
            line_mode = st.selectbox(
                "Line Mode",
                options=["lines", "lines+markers", "markers"],
                index=1,
                key="line_mode",
                help="Select the style of the line"
            )
        
        with col3:
            # Color scheme
            color_scheme = st.selectbox(
                "Color Scheme",
                options=["plotly", "viridis", "plasma", "inferno", "magma", "cividis"],
                index=0,
                key="line_color_scheme",
                help="Select the color scheme for the chart"
            )
        
        # Create a button to generate the chart
        if y_axes and st.button("Generate Line Chart"):
            try:
                # Create the line chart
                import plotly.express as px
                
                # Check if x-axis is datetime
                if pd.api.types.is_datetime64_any_dtype(df[x_axis]):
                    # Sort by date for time series
                    df_sorted = df.sort_values(by=x_axis)
                    
                    fig = px.line(
                        df_sorted,
                        x=x_axis,
                        y=y_axes,
                        title=chart_title,
                        color_discrete_sequence=px.colors.sequential.Viridis if color_scheme != "plotly" else None,
                        line_shape="linear",
                        markers=(line_mode != "lines")
                    )
                else:
                    fig = px.line(
                        df,
                        x=x_axis,
                        y=y_axes,
                        title=chart_title,
                        color_discrete_sequence=px.colors.sequential.Viridis if color_scheme != "plotly" else None,
                        line_shape="linear",
                        markers=(line_mode != "lines")
                    )
                
                # Update line mode
                if line_mode == "markers":
                    fig.update_traces(mode="markers")
                elif line_mode == "lines":
                    fig.update_traces(mode="lines")
                else:
                    fig.update_traces(mode="lines+markers")
                
                # Display the chart
                st.plotly_chart(fig, use_container_width=True)
                
                # Option to download the chart
                st.download_button(
                    label="Download Chart as HTML",
                    data=fig.to_html(),
                    file_name="line_chart.html",
                    mime="text/html"
                )
            except Exception as e:
                st.error(f"Error generating line chart: {e}")
def create_heatmap(df, numeric_columns, categorical_columns):
    st.subheader("Heatmap")
    
    # Add unique keys to all widgets by prefixing with "heatmap_"
    
    # Example for column selection widgets
    columns_to_include = st.multiselect(
        "Select columns to include in heatmap",
        options=numeric_columns,
        default=numeric_columns[:min(5, len(numeric_columns))],
        key="heatmap_columns_to_include"
    )
    
    # Color scheme selection
    color_scheme = st.selectbox(
        "Color scheme",
        options=["viridis", "plasma", "inferno", "magma", "cividis", "blues", "reds"],
        index=0,
        key="heatmap_color_scheme"
    )
    
    # Method selection
    correlation_method = st.selectbox(
        "Correlation method",
        options=["pearson", "kendall", "spearman"],
        index=0,
        help="Method of correlation calculation",
        key="heatmap_correlation_method"
    )
    
    # Any other widget needs a unique key
    show_values = st.checkbox(
        "Show correlation values", 
        value=True,
        key="heatmap_show_values"
    )
    
    # Example for sliders
    fig_height = st.slider(
        "Figure height", 
        min_value=400, 
        max_value=1000, 
        value=600,
        key="heatmap_fig_height"
    )
    
    # Example for other select boxes
    mask_option = st.selectbox(
        "Mask options",
        options=["None", "Upper triangle", "Lower triangle"],
        index=0,
        key="heatmap_mask_option"
    )
    
    if df is not None:
        # Get list of numeric columns
        numeric_columns = df.select_dtypes(include=['number']).columns.tolist()
        
        if len(numeric_columns) < 2:
            st.warning("Need at least 2 numeric columns to create a heatmap.")
            return
        
        # Let user select columns for the heatmap
        selected_columns = st.multiselect(
            "Select columns for heatmap",
            options=numeric_columns,
            default=numeric_columns[:min(8, len(numeric_columns))],
            key="heatmap_selected_columns",
            help="Select two or more numeric columns for the heatmap"
        )
        
        if len(selected_columns) < 2:
            st.warning("Please select at least 2 columns for the heatmap.")
            return
        
        # Additional options
        col1, col2 = st.columns(2)
        
        with col1:
            # Heatmap type
            heatmap_type = st.selectbox(
                "Heatmap Type",
                options=["Correlation", "Values"],
                index=0,
                key="heatmap_type",
                help="Select the type of heatmap to create"
            )
        
        with col2:
            # Color scheme
            color_scheme = st.selectbox(
                "Color Scheme",
                options=["RdBu_r", "Viridis", "Plasma", "Inferno", "Magma", "Cividis"],
                index=0,
                key="heatmap_color_scheme2",
                help="Select the color scheme for the heatmap"
            )
        
        # Create a button to generate the heatmap
        if st.button("Generate Heatmap"):
            try:
                import plotly.express as px
                import plotly.graph_objects as go
                
                # Create the heatmap based on the selected type
                if heatmap_type == "Correlation":
                    # Calculate correlation matrix
                    corr_matrix = df[selected_columns].corr()
                    
                    # Create heatmap
                    fig = px.imshow(
                        corr_matrix,
                        text_auto=True,
                        aspect="auto",
                        color_continuous_scale=color_scheme.lower(),
                        title="Correlation Heatmap"
                    )
                else:  # Values
                    # For values heatmap, we need categorical columns for rows and columns
                    categorical_columns = df.select_dtypes(include=['object', 'category']).columns.tolist()
                    
                    if len(categorical_columns) < 2:
                        st.warning("Need at least 2 categorical columns for a values heatmap.")
                        
                        # Create a simple heatmap of the values
                        fig = px.imshow(
                            df[selected_columns],
                            text_auto=True,
                            aspect="auto",
                            color_continuous_scale=color_scheme.lower(),
                            title="Values Heatmap"
                        )
                    else:
                        # Let user select categorical columns for rows and columns
                        row_col, col_col = st.columns(2)
                        
                        with row_col:
                            row_column = st.selectbox(
                                "Select column for rows",
                                options=categorical_columns,
                                key="heatmap_row_column",
                                help="Select the categorical column for the rows",  
                                index=0
                            )
                        
                        with col_col:
                            col_column = st.selectbox(
                                "Select column for columns",
                                options=[c for c in categorical_columns if c != row_column],
                                key="heatmap_col_column",
                                index=0
                            )
                        
                        # Let user select a value column
                        value_column = st.selectbox(
                            "Select column for values",
                            options=numeric_columns,
                            key="heatmap_value_column",
                            index=0
                        )
                        
                        # Create a pivot table
                        pivot_table = df.pivot_table(
                            index=row_column,
                            columns=col_column,
                            values=value_column,
                            aggfunc='mean'
                        )
                        
                        # Create heatmap
                        fig = px.imshow(
                            pivot_table,
                            text_auto=True,
                            aspect="auto",
                            color_continuous_scale=color_scheme.lower(),
                            title=f"Heatmap of {value_column} by {row_column} and {col_column}"
                        )
                
                # Display the heatmap
                st.plotly_chart(fig, use_container_width=True)
                
                # Option to download the chart
                st.download_button(
                    label="Download Heatmap as HTML",
                    data=fig.to_html(),
                    file_name="heatmap.html",
                    mime="text/html"
                )
            except Exception as e:
                st.error(f"Error generating heatmap: {e}")
def create_bar_graph(df, numeric_columns, categorical_columns):
    st.subheader("Bar Graph")
    
   
    color_scheme = st.selectbox(
        "Color Scheme",
        options=["blues", "viridis", "rocket", "mako", "crest", "magma"],
        index=0,
        help="Select the color scheme for the chart",
        key="bar_color_scheme"  # Add unique key
    )
    
    if df is not None:
        # Get list of columns
        columns = df.columns.tolist()
        numeric_columns = df.select_dtypes(include=['number']).columns.tolist()
        categorical_columns = df.select_dtypes(include=['object', 'category']).columns.tolist() + \
                             [c for c in df.columns if df[c].nunique() < 10]
        
        # Create columns for layout
        col1, col2 = st.columns(2)
        
        with col1:
            # Let user select x-axis column
            x_axis = st.selectbox(
                "Select X-axis column (categorical)",
                options=categorical_columns if categorical_columns else columns,
                key="bar_x_axis",  # Add unique key
                help="Select the categorical column for the X-axis"
            )
        
        with col2:
            # Let user select y-axis column
            y_axis = st.selectbox(
                "Select Y-axis column (numeric)",
                options=numeric_columns,
                key="bar_y_axis",  # Add unique key
                help="Select the numeric column for the Y-axis"
            )
        
        # Additional options
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Chart title
            chart_title = st.text_input(
                "Chart Title",
                value=f"Bar Graph: {y_axis} by {x_axis}"
            )
        
        with col2:
            # Orientation
            orientation = st.selectbox(
                "Orientation",
                options=["Vertical", "Horizontal"],
                index=0,
                key="bar_graph_orientation",
                help="Select the orientation of the bars"
            )
        
        with col3:
            # Color scheme
            color_scheme = st.selectbox(
                "Color Scheme",
                options=["plotly", "viridis", "plasma", "inferno", "magma", "cividis"],
                index=0,
                key="bar_graph_color_scheme",
                help="Select the color scheme for the chart"
            )
        
        # Additional options
        col1, col2 = st.columns(2)
        
        with col1:
            # Aggregation function
            agg_func = st.selectbox(
                "Aggregation Function",
                options=["sum", "mean", "count", "min", "max"],
                index=0,
                key="bar_graph_agg_func",
                help="Select how to aggregate the Y-axis values"
            )
        
        with col2:
            # Group by column
            group_by = st.selectbox(
                "Group By (optional)",
                options=["None"] + categorical_columns,
                index=0,
                key="bar_graph_group_by",
                help="Select a column to group the bars by"
            )
        
        # Create a button to generate the chart
        if st.button("Generate Bar Graph"):
            try:
                import plotly.express as px
                
                # Prepare the data
                if group_by != "None":
                    # Group by two categorical columns
                    if agg_func == "sum":
                        df_agg = df.groupby([x_axis, group_by])[y_axis].sum().reset_index()
                    elif agg_func == "mean":
                        df_agg = df.groupby([x_axis, group_by])[y_axis].mean().reset_index()
                    elif agg_func == "count":
                        df_agg = df.groupby([x_axis, group_by])[y_axis].count().reset_index()
                    elif agg_func == "min":
                        df_agg = df.groupby([x_axis, group_by])[y_axis].min().reset_index()
                    elif agg_func == "max":
                        df_agg = df.groupby([x_axis, group_by])[y_axis].max().reset_index()
                    
                    # Create the grouped bar chart
                    if orientation == "Vertical":
                        fig = px.bar(
                            df_agg,
                            x=x_axis,
                            y=y_axis,
                            color=group_by,
                            title=chart_title,
                            barmode="group",
                            color_discrete_sequence=px.colors.sequential.Viridis if color_scheme != "plotly" else None
                        )
                    else:  # Horizontal
                        fig = px.bar(
                            df_agg,
                            y=x_axis,
                            x=y_axis,
                            color=group_by,
                            title=chart_title,
                            barmode="group",
                            orientation='h',
                            color_discrete_sequence=px.colors.sequential.Viridis if color_scheme != "plotly" else None
                        )
                else:
                    # Group by one categorical column
                    if agg_func == "sum":
                        df_agg = df.groupby(x_axis)[y_axis].sum().reset_index()
                    elif agg_func == "mean":
                        df_agg = df.groupby(x_axis)[y_axis].mean().reset_index()
                    elif agg_func == "count":
                        df_agg = df.groupby(x_axis)[y_axis].count().reset_index()
                    elif agg_func == "min":
                        df_agg = df.groupby(x_axis)[y_axis].min().reset_index()
                    elif agg_func == "max":
                        df_agg = df.groupby(x_axis)[y_axis].max().reset_index()
                    
                    # Create the bar chart
                    if orientation == "Vertical":
                        fig = px.bar(
                            df_agg,
                            x=x_axis,
                            y=y_axis,
                            title=chart_title,
                            color_discrete_sequence=px.colors.sequential.Viridis if color_scheme != "plotly" else None
                        )
                    else:  # Horizontal
                        fig = px.bar(
                            df_agg,
                            y=x_axis,
                            x=y_axis,
                            title=chart_title,
                            orientation='h',
                            color_discrete_sequence=px.colors.sequential.Viridis if color_scheme != "plotly" else None
                        )
                
                # Display the chart
                st.plotly_chart(fig, use_container_width=True)
                
                # Option to download the chart
                st.download_button(
                    label="Download Bar Graph as HTML",
                    data=fig.to_html(),
                    file_name="bar_graph.html",
                    mime="text/html"
                )
            except Exception as e:
                st.error(f"Error generating bar graph: {e}")
def create_scatter_plot(df, numeric_columns, categorical_columns):
    """
    Create an interactive scatter plot visualization.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The dataframe containing the data to visualize
    numeric_columns : list
        List of numeric column names in the dataframe
    categorical_columns : list
        List of categorical column names in the dataframe
    """
    st.subheader("Scatter Plot")
   
    if len(numeric_columns) < 2:
        st.warning("Need at least 2 numeric columns to create a scatter plot.")
        return
    
    # Create columns for layout
    col1, col2 = st.columns(2)
    
    with col1:
        # Let user select x-axis column
        x_axis = st.selectbox(
            "Select X-axis column",
            options=numeric_columns,
            index=0,
            key="scatter_x_axis",
            help="Select the numeric column for the X-axis"
        )
    
    with col2:
        # Let user select y-axis column
        y_axis = st.selectbox(
            "Select Y-axis column",
            options=[col for col in numeric_columns if col != x_axis],
            index=0,
            key="scatter_y_axis",
            help="Select the numeric column for the Y-axis"
        )
    
    # Additional options
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Color by
        if categorical_columns:
            color_by = st.selectbox(
                "Color by",
                options=["None"] + categorical_columns,
                index=0,
                key="scatter_color_by",
                help="Select a categorical column to color the points"
            )
        else:
            color_by = "None"
    
    with col2:
        # Size by
        size_by = st.selectbox(
            "Size by",
            options=["None"] + numeric_columns,
            index=0,
            key="scatter_size_by",
            help="Select a numeric column to determine the size of points"
        )
    
    with col3:
        # Opacity
        opacity = st.slider(
            "Opacity",
            min_value=0.1,
            max_value=1.0,
            value=0.7,
            step=0.1,
            key="scatter_opacity",
            help="Adjust the opacity of the points"
        )
    
    # Additional options
    col1, col2 = st.columns(2)
    
    with col1:
        # Chart title
        chart_title = st.text_input(
            "Chart Title",
            value=f"Scatter Plot: {y_axis} vs {x_axis}"
        )
    
    with col2:
        # Trendline
        trendline = st.selectbox(
            "Trendline",
            options=["None", "OLS", "Lowess"],
            index=0,
            key="scatter_trendline",
            help="Add a trendline to the scatter plot"
        )
    
    # Create a button to generate the chart
    if st.button("Generate Scatter Plot"):
        try:
            # Prepare parameters
            color_param = None if color_by == "None" else color_by
            size_param = None if size_by == "None" else size_by
            trendline_param = None if trendline == "None" else trendline.lower()
            
            # Create the scatter plot
            fig = px.scatter(
                df,
                x=x_axis,
                y=y_axis,
                color=color_param,
                size=size_param,
                opacity=opacity,
                title=chart_title,
                trendline=trendline_param,
                labels={
                    x_axis: x_axis,
                    y_axis: y_axis
                }
            )
            
            # Update layout for better appearance
            fig.update_layout(
                plot_bgcolor='white',
                xaxis=dict(
                    showgrid=True,
                    gridcolor='lightgray',
                    zeroline=True,
                    zerolinecolor='lightgray'
                ),
                yaxis=dict(
                    showgrid=True,
                    gridcolor='lightgray',
                    zeroline=True,
                    zerolinecolor='lightgray'
                )
            )
            
            # Display the chart
            st.plotly_chart(fig, use_container_width=True)
            
            # Option to download the chart
            st.download_button(
                label="Download Chart as HTML",
                data=fig.to_html(),
                file_name="scatter_plot.html",
                mime="text/html"
            )
            
            # Display correlation
            if x_axis != y_axis:
                correlation = df[x_axis].corr(df[y_axis])
                st.info(f"Correlation between {x_axis} and {y_axis}: {correlation:.4f}")
        
        except Exception as e:
            st.error(f"Error generating scatter plot: {e}")
def create_pie_chart(df, numeric_columns, categorical_columns):
    """
    Create an interactive pie chart visualization.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The dataframe containing the data to visualize
    numeric_columns : list
        List of numeric column names in the dataframe
    categorical_columns : list
        List of categorical column names in the dataframe
    """
    st.subheader("Pie Chart")
    
    if not categorical_columns:
        st.warning("Need at least one categorical column to create a pie chart.")
        return
    
    if not numeric_columns:
        st.warning("Need at least one numeric column to create a pie chart.")
        return
    
    # Create columns for layout
    col1, col2 = st.columns(2)
    
    with col1:
        # Let user select names column (categories)
        names_column = st.selectbox(
            "Select category column",
            options=categorical_columns,
            index=0,
            key="pie_names_column",
            help="Select the categorical column for pie chart segments"
        )
    
    with col2:
        # Let user select values column
        values_column = st.selectbox(
            "Select values column",
            options=numeric_columns,
            index=0,
            key="pie_values_column",
            help="Select the numeric column for segment sizes"
        )
    
    # Additional options
    col1, col2 = st.columns(2)
    
    with col1:
        # Chart title
        chart_title = st.text_input(
            "Chart Title",
            value=f"Pie Chart: {values_column} by {names_column}"
        )
    
    with col2:
        # Chart type
        chart_type = st.selectbox(
            "Chart Type",
            options=["Pie Chart", "Donut Chart"],
            index=0,
            key="pie_chart_type",
            help="Select the type of chart"
        )
    
    # Additional options
    col1, col2 = st.columns(2)
    
    with col1:
        # Sort values
        sort_values = st.checkbox(
            "Sort segments by value",
            value=True,
            key="pie_sort_values",
            help="Sort the pie segments by their values"
        )
    
    with col2:
        # Pull out largest segment
        pull_out = st.checkbox(
            "Pull out largest segment",
            value=False,
            key="pie_pull_out",
            help="Pull out the largest segment for emphasis"
        )
    
    # Create a button to generate the chart
    if st.button("Generate Pie Chart"):
        try:
            # Group the data
            grouped_df = df.groupby(names_column)[values_column].sum().reset_index()
            
            # Sort if requested
            if sort_values:
                grouped_df = grouped_df.sort_values(values_column, ascending=False)
            
            # Create pull values if requested
            if pull_out:
                max_index = grouped_df[values_column].idxmax()
                pull = [0.1 if i == max_index else 0 for i in range(len(grouped_df))]
            else:
                pull = None
            
            # Create the pie chart
            if chart_type == "Pie Chart":
                fig = px.pie(
                    grouped_df,
                    names=names_column,
                    values=values_column,
                    title=chart_title,
                    hole=0
                )
            else:  # Donut Chart
                fig = px.pie(
                    grouped_df,
                    names=names_column,
                    values=values_column,
                    title=chart_title,
                    hole=0.4
                )
            
            # Apply pull if requested
            if pull:
                fig.update_traces(pull=pull)
            
            # Update layout for better appearance
            fig.update_layout(
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=-0.3,
                    xanchor="center",
                    x=0.5
                )
            )
            
            # Display the chart
            st.plotly_chart(fig, use_container_width=True)
            
            # Option to download the chart
            st.download_button(
                label="Download Chart as HTML",
                data=fig.to_html(),
                file_name="pie_chart.html",
                mime="text/html"
            )
            
            # Display data table
            with st.expander("View Data Table"):
                # Add percentage column
                grouped_df['Percentage'] = (grouped_df[values_column] / grouped_df[values_column].sum() * 100).round(2)
                grouped_df['Percentage'] = grouped_df['Percentage'].astype(str) + '%'
                
                st.dataframe(grouped_df, use_container_width=True)
        
        except Exception as e:
            st.error(f"Error generating pie chart: {e}")
def create_histogram(df, numeric_columns, categorical_columns):
    """
    Create an interactive histogram visualization.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The dataframe containing the data to visualize
    numeric_columns : list
        List of numeric column names in the dataframe
    """
    st.subheader("Histogram")
    
    x_column = st.selectbox(
        "X-axis",
        options=numeric_columns + categorical_columns,
        key="histogram_x_column"  # Add unique key
    )
    
    y_column = st.selectbox(
        "Y-axis",
        options=numeric_columns,
        key="histogram_y_column"  # Add unique key
    )
    
    if not numeric_columns:
        st.warning("Need at least one numeric column to create a histogram.")
        return
    
    # Let user select column
    column = st.selectbox(
        "Select column",
        options=numeric_columns,
        index=0,
        help="Select the numeric column for the histogram"
    )
    
    # Additional options
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Number of bins
        nbins = st.slider(
            "Number of bins",
            min_value=5,
            max_value=100,
            value=20,
            step=5,
            help="Adjust the number of bins in the histogram"
        )
    
    with col2:
        # Histogram type
        hist_type = st.selectbox(
            "Histogram Type",
            options=["Count", "Probability Density", "Cumulative"],
            index=0,
            help="Select the type of histogram"
        )
    
    with col3:
        # Chart title
        chart_title = st.text_input(
            "Chart Title",
            value=f"Histogram of {column}"
        )
    
    # Additional options
    col1, col2 = st.columns(2)
    
    with col1:
        # Show rug plot
        show_rug = st.checkbox(
            "Show rug plot",
            value=False,
            help="Add a rug plot to show individual data points"
        )
    
    with col2:
        # Show KDE
        show_kde = st.checkbox(
            "Show KDE (density curve)",
            value=True,
            help="Add a kernel density estimate curve"
        )
    
    # Create a button to generate the chart
    if st.button("Generate Histogram"):
        try:
            # Determine histogram normalization
            if hist_type == "Count":
                histnorm = None
            elif hist_type == "Probability Density":
                histnorm = "probability density"
            else:  # Cumulative
                histnorm = "probability"
            
            # Create the histogram
            fig = px.histogram(
                df,
                x=column,
                nbins=nbins,
                histnorm=histnorm,
                title=chart_title,
                marginal="rug" if show_rug else None,
                opacity=0.7
            )
            
            # Add KDE if requested
            if show_kde:
                # Calculate KDE
                from scipy.stats import gaussian_kde
                import numpy as np
                
                # Get data without NaN values
                data = df[column].dropna()
                
                # Calculate KDE
                kde = gaussian_kde(data)
                
                # Create x range for KDE
                x_range = np.linspace(data.min(), data.max(), 1000)
                
                # Calculate KDE values
                y_kde = kde(x_range)
                
                # Scale KDE to match histogram height
                if histnorm is None:
                    # For count histogram, scale to match the highest bin
                    hist_values, bin_edges = np.histogram(data, bins=nbins)
                    scale_factor = hist_values.max() / y_kde.max()
                    y_kde = y_kde * scale_factor
                elif histnorm == "probability density":
                    # For density histogram, no scaling needed
                    pass
                else:  # Cumulative
                    # For cumulative, scale to 0-1 range
                    y_kde = np.cumsum(y_kde) / np.sum(y_kde)
                
                # Add KDE line
                fig.add_trace(
                    go.Scatter(
                        x=x_range,
                        y=y_kde,
                        mode='lines',
                        name='KDE',
                        line=dict(color='red', width=2)
                    )
                )
            
            # Update layout for better appearance
            fig.update_layout(
                plot_bgcolor='white',
                xaxis=dict(
                    title=column,
                    showgrid=True,
                    gridcolor='lightgray',
                    zeroline=True,
                    zerolinecolor='lightgray'
                ),
                yaxis=dict(
                    title="Count" if histnorm is None else "Density" if histnorm == "probability density" else "Cumulative Probability",
                    showgrid=True,
                    gridcolor='lightgray',
                    zeroline=True,
                    zerolinecolor='lightgray'
                )
            )
            
            # Display the chart
            st.plotly_chart(fig, use_container_width=True)
            
            # Option to download the chart
            st.download_button(
                label="Download Chart as HTML",
                data=fig.to_html(),
                file_name="histogram.html",
                mime="text/html"
            )
            
            # Display statistics
            with st.expander("View Statistics"):
                stats = df[column].describe()
                st.dataframe(stats)
                
                # Calculate additional statistics
                from scipy import stats as scipy_stats
                
                skewness = scipy_stats.skew(df[column].dropna())
                kurtosis = scipy_stats.kurtosis(df[column].dropna())
                
                st.write(f"Skewness: {skewness:.4f}")
                st.write(f"Kurtosis: {kurtosis:.4f}")
                
                # Interpret skewness
                if abs(skewness) < 0.5:
                    st.write("The distribution is approximately symmetric.")
                elif skewness < 0:
                    st.write("The distribution is negatively skewed (left-tailed).")
                else:
                    st.write("The distribution is positively skewed (right-tailed).")
        
        except Exception as e:
            st.error(f"Error generating histogram: {e}")
def create_box_plot(df, numeric_columns, categorical_columns):
    """
    Create an interactive box plot visualization.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The dataframe containing the data to visualize
    numeric_columns : list
        List of numeric column names in the dataframe
    categorical_columns : list
        List of categorical column names in the dataframe
    """
    st.subheader("Box Plot")
    
    if not numeric_columns:
        st.warning("Need at least one numeric column to create a box plot.")
        return
    
    # Create columns for layout
    col1, col2 = st.columns(2)
    
    with col1:
        # Let user select y-axis column (numeric)
        y_axis = st.selectbox(
            "Select Y-axis column (numeric)",
            options=numeric_columns,
            index=0,
            help="Select the numeric column for the Y-axis"
        )
    
    with col2:
        # Let user select x-axis column (categorical, optional)
        if categorical_columns:
            x_axis = st.selectbox(
                "Select X-axis column (categorical, optional)",
                options=["None"] + categorical_columns,
                index=0,
                help="Select the categorical column for the X-axis"
            )
        else:
            x_axis = "None"
    
    # Additional options
    col1, col2 = st.columns(2)
    
    with col1:
        # Chart title
        chart_title = st.text_input(
            "Chart Title",
            value=f"Box Plot of {y_axis}" if x_axis == "None" else f"Box Plot of {y_axis} by {x_axis}"
        )
    
    with col2:
        # Color by
        if categorical_columns and x_axis != "None":
            color_options = ["Same as X-axis"] + [col for col in categorical_columns if col != x_axis]
            color_by = st.selectbox(
                "Color by",
                options=color_options,
                index=0,
                help="Select a categorical column to color the boxes"
            )
            
            if color_by == "Same as X-axis":
                color_by = x_axis
        else:
            color_by = None
    
    # Additional options
    col1, col2 = st.columns(2)
    
    with col1:
        # Show points
        points = st.selectbox(
            "Show individual points",
            options=["Outliers only", "All points", "None"],
            index=0,
            help="Control the display of individual data points"
        )
    
    with col2:
        # Orientation
        orientation = st.selectbox(
            "Orientation",
            options=["Vertical", "Horizontal"],
            index=0,
            help="Select the orientation of the box plot"
        )
    
    # Create a button to generate the chart
    if st.button("Generate Box Plot"):
        try:
            # Prepare parameters
            x_param = None if x_axis == "None" else x_axis
            color_param = color_by
            
            # Set points parameter
            if points == "Outliers only":
                points_param = "outliers"
            elif points == "All points":
                points_param = "all"
            else:  # None
                points_param = False
            
            # Create the box plot
            if orientation == "Vertical":
                fig = px.box(
                    df,
                    x=x_param,
                    y=y_axis,
                    color=color_param,
                    title=chart_title,
                    points=points_param
                )
            else:  # Horizontal
                fig = px.box(
                    df,
                    y=x_param,
                    x=y_axis,
                    color=color_param,
                    title=chart_title,
                    points=points_param,
                    orientation='h'
                )
            
            # Update layout for better appearance
            fig.update_layout(
                plot_bgcolor='white',
                xaxis=dict(
                    showgrid=True,
                    gridcolor='lightgray',
                    zeroline=True,
                    zerolinecolor='lightgray'
                ),
                yaxis=dict(
                    showgrid=True,
                    gridcolor='lightgray',
                    zeroline=True,
                    zerolinecolor='lightgray'
                )
            )
            
            # Display the chart
            st.plotly_chart(fig, use_container_width=True)
            
            # Option to download the chart
            st.download_button(
                label="Download Chart as HTML",
                data=fig.to_html(),
                file_name="box_plot.html",
                mime="text/html"
            )
            
            # Display statistics
            if x_param is not None:
                with st.expander("View Statistics by Group"):
                    # Group statistics
                    group_stats = df.groupby(x_param)[y_axis].describe()
                    st.dataframe(group_stats)
                    
                    # ANOVA test if more than one group
                    if len(df[x_param].unique()) > 1:
                        try:
                            from scipy import stats as scipy_stats
                            
                            # Prepare data for ANOVA
                            groups = [df[df[x_param] == group][y_axis].dropna() for group in df[x_param].unique()]
                            
                            # Run ANOVA
                            f_stat, p_value = scipy_stats.f_oneway(*groups)
                            
                            st.write("### One-way ANOVA Test")
                            st.write(f"F-statistic: {f_stat:.4f}")
                            st.write(f"p-value: {p_value:.4f}")
                            
                            if p_value < 0.05:
                                st.write("The groups are significantly different (p < 0.05).")
                            else:
                                st.write("There is no significant difference between the groups (p >= 0.05).")
                        except Exception as e:
                            st.warning(f"Could not perform ANOVA test: {e}")
            else:
                with st.expander("View Statistics"):
                    stats = df[y_axis].describe()
                    st.dataframe(stats)
        
        except Exception as e:
            st.error(f"Error generating box plot: {e}")
def create_violin_plot(df, numeric_columns, categorical_columns):
    """
    Create an interactive violin plot visualization.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The dataframe containing the data to visualize
    numeric_columns : list
        List of numeric column names in the dataframe
    categorical_columns : list
        List of categorical column names in the dataframe
    """
    st.subheader("Violin Plot")
    
    x_column = st.selectbox(
        "X-axis",
        options=numeric_columns + categorical_columns,
        key="violin_x_column"  # Add unique key
    )
    
    y_column = st.selectbox(
        "Y-axis",
        options=numeric_columns,
        key="violin_y_column"  # Add unique key
    )
    
    if not numeric_columns:
        st.warning("Need at least one numeric column to create a violin plot.")
        return
    
    # Create columns for layout
    col1, col2 = st.columns(2)
    
    with col1:
        # Let user select y-axis column (numeric)
        y_axis = st.selectbox(
            "Select Y-axis column (numeric)",
            options=numeric_columns,
            index=0,
            key="violin_y_axis",  # Add unique key
            help="Select the numeric column for the Y-axis"
        )
    
    with col2:
        # Let user select x-axis column (categorical, optional)
        if categorical_columns:
            x_axis = st.selectbox(
                "Select X-axis column (categorical, optional)",
                options=["None"] + categorical_columns,
                index=0,
                key="violin_x_axis",  # Add unique key
                help="Select the categorical column for the X-axis"
            )
        else:
            x_axis = "None"
    
    # Additional options
    col1, col2 = st.columns(2)
    
    with col1:
        # Chart title
        chart_title = st.text_input(
            "Chart Title",
            value=f"Violin Plot of {y_axis}" if x_axis == "None" else f"Violin Plot of {y_axis} by {x_axis}"
        )
    
    with col2:
        # Color by
        if categorical_columns and x_axis != "None":
            color_options = ["Same as X-axis"] + [col for col in categorical_columns if col != x_axis]
            color_by = st.selectbox(
                "Color by",
                options=color_options,
                index=0,
                key="violin_color_by",  # Add unique key
                help="Select a categorical column to color the violins"
            )
            
            if color_by == "Same as X-axis":
                color_by = x_axis
        else:
            color_by = None
    
    # Additional options
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Show box
        show_box = st.checkbox(
            "Show box plot inside",
            value=True,
            key="violin_show_box",  # Add unique key
            help="Show a box plot inside the violin plot"
        )
    
    with col2:
        # Show points
        points = st.selectbox(
            "Show individual points",
            options=["None", "All points", "Outliers only"],
            index=0,
            key="violin_points",  # Add unique key
            help="Control the display of individual data points"
        )
    
    with col3:
        # Orientation
        orientation = st.selectbox(
            "Orientation",
            options=["Vertical", "Horizontal"],
            index=0,
            key="violin_orientation",  # Add unique key
            help="Select the orientation of the violin plot"
        )
    
    # Create a button to generate the chart
    if st.button("Generate Violin Plot"):
        try:
            # Prepare parameters
            x_param = None if x_axis == "None" else x_axis
            color_param = color_by
            
            # Set points parameter
            if points == "All points":
                points_param = "all"
            elif points == "Outliers only":
                points_param = "outliers"
            else:  # None
                points_param = False
            
            # Set box parameter
            box_param = True if show_box else False
            
            # Create the violin plot
            if orientation == "Vertical":
                fig = px.violin(
                    df,
                    x=x_param,
                    y=y_axis,
                    color=color_param,
                    title=chart_title,
                    box=box_param,
                    points=points_param
                )
            else:  # Horizontal
                fig = px.violin(
                    df,
                    y=x_param,
                    x=y_axis,
                    color=color_param,
                    title=chart_title,
                    box=box_param,
                    points=points_param,
                    orientation='h'
                )
            
            # Update layout for better appearance
            fig.update_layout(
                plot_bgcolor='white',
                xaxis=dict(
                    showgrid=True,
                    gridcolor='lightgray',
                    zeroline=True,
                    zerolinecolor='lightgray'
                ),
                yaxis=dict(
                    showgrid=True,
                    gridcolor='lightgray',
                    zeroline=True,
                    zerolinecolor='lightgray'
                )
            )
            
            # Display the chart
            st.plotly_chart(fig, use_container_width=True)
            
            # Option to download the chart
            st.download_button(
                label="Download Chart as HTML",
                data=fig.to_html(),
                file_name="violin_plot.html",
                mime="text/html"
            )
            
            # Display statistics
            if x_param is not None:
                with st.expander("View Statistics by Group"):
                    # Group statistics
                    group_stats = df.groupby(x_param)[y_axis].describe()
                    st.dataframe(group_stats)
                    
                    # Kernel density estimates
                    st.write("### Kernel Density Estimates")
                    
                    # Create a separate plot for KDE comparison
                    import plotly.figure_factory as ff
                    
                    # Prepare data for KDE
                    kde_data = []
                    group_names = []
                    
                    for group in df[x_param].unique():
                        group_data = df[df[x_param] == group][y_axis].dropna()
                        if len(group_data) > 0:
                            kde_data.append(group_data)
                            group_names.append(str(group))
                    
                    if len(kde_data) > 0:
                        # Create KDE plot
                        fig_kde = ff.create_distplot(
                            kde_data,
                            group_names,
                            bin_size=0.2,
                            show_hist=False,
                            show_rug=False
                        )
                        
                        fig_kde.update_layout(
                            title="Kernel Density Estimates by Group",
                            xaxis_title=y_axis,
                            yaxis_title="Density",
                            plot_bgcolor='white',
                            xaxis=dict(
                                showgrid=True,
                                gridcolor='lightgray',
                                zeroline=True,
                                zerolinecolor='lightgray'
                            ),
                            yaxis=dict(
                                showgrid=True,
                                gridcolor='lightgray',
                                zeroline=True,
                                zerolinecolor='lightgray'
                            )
                        )
                        
                        st.plotly_chart(fig_kde, use_container_width=True)
            else:
                with st.expander("View Statistics"):
                    stats = df[y_axis].describe()
                    st.dataframe(stats)
        
        except Exception as e:
            st.error(f"Error generating violin plot: {e}")
def create_3d_scatter_plot(df, numeric_columns, categorical_columns):
    """
    Create an interactive 3D scatter plot visualization.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The dataframe containing the data to visualize
    numeric_columns : list
        List of numeric column names in the dataframe
    categorical_columns : list
        List of categorical column names in the dataframe
    """
    st.subheader("3D Scatter Plot")
    
    # For axis selections
    x_column = st.selectbox(
        "X-axis",
        options=numeric_columns,
        index=0 if numeric_columns else 0,
        key="scatter3d_x_column"
    )
    
    y_column = st.selectbox(
        "Y-axis",
        options=numeric_columns,
        index=min(1, len(numeric_columns)-1) if numeric_columns else 0,
        key="scatter3d_y_column"
    )
    
    z_column = st.selectbox(
        "Z-axis",
        options=numeric_columns,
        index=min(2, len(numeric_columns)-1) if numeric_columns else 0,
        key="scatter3d_z_column"
    )
    
    # For color selection
    if categorical_columns:
        color_by = st.selectbox(
            "Color by",
            options=["None"] + categorical_columns,
            index=0,
            key="scatter3d_color_by"
        )
    else:
        color_by = "None"
    
    # For marker size selection
    size_by = st.selectbox(
        "Size by",
        options=["None"] + numeric_columns,
        index=0,
        key="scatter3d_size_by"
    )
    
    # For marker size scaling
    marker_size = st.slider(
        "Marker size",
        min_value=1,
        max_value=20,
        value=5,
        key="scatter3d_marker_size"
    )
    
    # For opacity selection
    opacity = st.slider(
        "Opacity",
        min_value=0.1,
        max_value=1.0,
        value=0.7,
        step=0.1,
        key="scatter3d_opacity2"
    )
    
    # For color scheme selection
    color_scheme = st.selectbox(
        "Color scheme",
        options=["viridis", "plasma", "inferno", "magma", "cividis", "blues", "reds"],
        index=0,
        key="scatter3d_color_scheme"
    )
    
    
    if len(numeric_columns) < 3:
        st.warning("Need at least 3 numeric columns to create a 3D scatter plot.")
        return
    
    # Create columns for layout
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Let user select x-axis column
        x_axis = st.selectbox(
            "Select X-axis column",
            options=numeric_columns,
            index=0,
            key="scatter3d_x_axis",
            help="Select the numeric column for the X-axis"
        )
    
    with col2:
        # Let user select y-axis column
        y_axis = st.selectbox(
            "Select Y-axis column",
            options=[col for col in numeric_columns if col != x_axis],
            index=0,
            key="scatter3d_y_axis",
            help="Select the numeric column for the Y-axis"
        )
    
    with col3:
        # Let user select z-axis column
        z_axis = st.selectbox(
            "Select Z-axis column",
            options=[col for col in numeric_columns if col != x_axis and col != y_axis],
            index=0,
            key="scatter3d_z_axis",
            help="Select the numeric column for the Z-axis"
        )
    
    # Additional options
    col1, col2 = st.columns(2)
    
    with col1:
        # Color by
        if categorical_columns:
            color_by = st.selectbox(
                "Color by",
                options=["None"] + categorical_columns,
                index=0,
                key="scatter3d_color_by2",
                help="Select a categorical column to color the points"
            )
        else:
            color_by = "None"
    
    with col2:
        # Size by
        size_by = st.selectbox(
            "Size by",
            options=["None"] + [col for col in numeric_columns if col not in [x_axis, y_axis, z_axis]],
            index=0,
            key="scatter3d_size_by2",
            help="Select a numeric column to determine the size of points"
        )
    
    # Additional options
    col1, col2 = st.columns(2)
    
    with col1:
        # Chart title
        chart_title = st.text_input(
            "Chart Title",
            key="scatter3d_chart_title",
            value=f"3D Scatter Plot: {x_axis}, {y_axis}, {z_axis}"
        )
    
    with col2:
        # Opacity
        opacity = st.slider(
            "Opacity",
            min_value=0.1,
            max_value=1.0,
            value=0.8,
            step=0.1,
            key="scatter3d_opacity",
            help="Adjust the opacity of the points"
        )
    
    # Create a button to generate the chart
    if st.button("Generate 3D Scatter Plot"):
        try:
            # Prepare parameters
            color_param = None if color_by == "None" else color_by
            size_param = None if size_by == "None" else size_by
            
            # Create the 3D scatter plot
            fig = px.scatter_3d(
                df,
                x=x_axis,
                y=y_axis,
                z=z_axis,
                color=color_param,
                size=size_param,
                opacity=opacity,
                title=chart_title,
                labels={
                    x_axis: x_axis,
                    y_axis: y_axis,
                    z_axis: z_axis
                }
            )
            
            # Update layout for better appearance
            fig.update_layout(
                scene=dict(
                    xaxis_title=x_axis,
                    yaxis_title=y_axis,
                    zaxis_title=z_axis,
                    xaxis=dict(
                        backgroundcolor="rgb(230, 230, 230)",
                        gridcolor="white",
                        showbackground=True,
                        zerolinecolor="white"
                    ),
                    yaxis=dict(
                        backgroundcolor="rgb(230, 230, 230)",
                        gridcolor="white",
                        showbackground=True,
                        zerolinecolor="white"
                    ),
                    zaxis=dict(
                        backgroundcolor="rgb(230, 230, 230)",
                        gridcolor="white",
                        showbackground=True,
                        zerolinecolor="white"
                    )
                ),
                margin=dict(r=10, l=10, b=10, t=50)
            )
            
            # Display the chart
            st.plotly_chart(fig, use_container_width=True)
            
            # Option to download the chart
            st.download_button(
                label="Download Chart as HTML",
                data=fig.to_html(),
                file_name="3d_scatter_plot.html",
                mime="text/html"
            )
            
            # Display correlations
            with st.expander("View Correlations"):
                corr_matrix = df[[x_axis, y_axis, z_axis]].corr()
                st.dataframe(corr_matrix.style.background_gradient(cmap='coolwarm'))
                
                st.write(f"Correlation between {x_axis} and {y_axis}: {corr_matrix.loc[x_axis, y_axis]:.4f}")
                st.write(f"Correlation between {x_axis} and {z_axis}: {corr_matrix.loc[x_axis, z_axis]:.4f}")
                st.write(f"Correlation between {y_axis} and {z_axis}: {corr_matrix.loc[y_axis, z_axis]:.4f}")
        
        except Exception as e:
            st.error(f"Error generating 3D scatter plot: {e}")
def create_other_visualizations(df, numeric_columns, categorical_columns):
    st.subheader("Other Visualizations")
    
    if df is not None:
        # Let user select visualization type
        viz_type = st.selectbox(
            "Select Visualization Type",
            options=["Scatter Plot", "Pie Chart", "Histogram", "Box Plot", "Violin Plot", "3D Scatter Plot"],
            index=0,
            key="other_viz_type",
            help="Select the type of visualization to create"
        )
        
        # Get list of columns
        columns = df.columns.tolist()
        numeric_columns = df.select_dtypes(include=['number']).columns.tolist()
        categorical_columns = df.select_dtypes(include=['object', 'category']).columns.tolist() + \
                             [c for c in df.columns if df[c].nunique() < 10]
        
        # Create visualization based on selected type
        if viz_type == "Scatter Plot":
            create_scatter_plot(df, numeric_columns, categorical_columns)
        elif viz_type == "Pie Chart":
            create_pie_chart(df, numeric_columns, categorical_columns)
        elif viz_type == "Histogram":
            create_histogram(df, numeric_columns, categorical_columns)
        elif viz_type == "Box Plot":
            create_box_plot(df, numeric_columns, categorical_columns)
        elif viz_type == "Violin Plot":
            create_violin_plot(df, numeric_columns, categorical_columns)
        elif viz_type == "3D Scatter Plot":
            create_3d_scatter_plot(df, numeric_columns, categorical_columns)
def data_visualization_page(df):
    st.header(" Data Visualization")
    
    if df is None:
        st.warning("Please load data first.")
        return
    if df is not None:
        # Let user select visualization type
        
        # Get list of columns
        columns = df.columns.tolist()
        numeric_columns = df.select_dtypes(include=['number']).columns.tolist()
        categorical_columns = df.select_dtypes(include=['object', 'category']).columns.tolist() + \
                             [c for c in df.columns if df[c].nunique() < 10]
        
    # Create tabs for different visualization types
    tab1, tab2, tab3, tab4 = st.tabs(["Line Charts", "Heatmaps", "Bar Graphs", "Other Visualizations"])
    
    # Tab 1: Line Charts
    with tab1:
        create_line_chart(df, numeric_columns, categorical_columns)
    
    # Tab 2: Heatmaps
    with tab2:
        create_heatmap(df, numeric_columns, categorical_columns)
    
    # Tab 3: Bar Graphs
    with tab3:
        create_bar_graph(df, numeric_columns, categorical_columns)
    
    # Tab 4: Other Visualizations
    with tab4:
        create_other_visualizations(df, numeric_columns, categorical_columns)

def about_page():    
    pass    

def create_responsive_layout():
    # Set page configuration for responsive design
    st.set_page_config(
        page_title="Interactive Data Analysis Dashboard",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Add custom CSS for responsive design
    st.markdown("""
    <style>
    /* Responsive layout adjustments */
    @media (max-width: 768px) {
        .main .block-container {
            padding-left: 1rem;
            padding-right: 1rem;
        }
        
        /* Adjust column widths on small screens */
        [data-testid="column"] {
            width: 100% !important;
            flex: 1 1 100% !important;
            margin-bottom: 1rem;
        }
    }
    
    /* Card styling */
    .card {
        padding: 1.5rem;
        border-radius: 0.5rem;
        background-color: #f8f9fa;
        box-shadow: 0 0.125rem 0.25rem rgba(0, 0, 0, 0.075);
        margin-bottom: 1rem;
    }
    
    /* Button styling */
    .stButton > button {
        width: 100%;
    }
    
    /* Improve table readability */
    .dataframe {
        font-size: 0.8rem;
    }
    
    /* Improve sidebar navigation */
    .sidebar .sidebar-content {
        background-color: #f8f9fa;
    }
    </style>
    """, unsafe_allow_html=True)
# Initialize session state
def initialize_session_state():
    if "data" not in st.session_state:
        st.session_state.data = None
    
    if "filtered_data" not in st.session_state:
        st.session_state.filtered_data = None
    
    if "activity_log" not in st.session_state:
        st.session_state.activity_log = []
    
    if "current_page" not in st.session_state:
        st.session_state.current_page = "Home"
def log_activity(activity):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    st.session_state.activity_log.append(f"[{timestamp}] {activity}")
def home_page():
    st.title("Interactive Data Analysis Dashboard")
    
    # Create a clean, modern layout
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.write("""
        ## Welcome to Your Data Analysis Hub
        
        This interactive dashboard provides powerful tools to explore, manipulate, and visualize your data.
        
        ### Key Features:
        - Load data from various file formats
        - Clean and transform your data
        - Create stunning visualizations
        - Perform advanced analysis
        - Export results and insights
        
        Get started by uploading your data in the **Data Loading** section.
        """)
       
def create_sidebar():

    with st.sidebar:
        st.title("Data Analysis Dashboard")
        
        # Navigation
        page = st.radio(
            "Navigation",
            ["Home", "Data Loading", "Data Manipulation", "Data Visualization", "Analysis", "About"],
            index=0,
            key="navigation_radio"
        )
        
        # Global filters (only show if data is loaded)
        if st.session_state.data is not None:
            st.subheader("Global Filters")
            
            # Get list of categorical columns
            categorical_columns = st.session_state.data.select_dtypes(include=['object', 'category']).columns.tolist()
            
            # Add filters for categorical columns
            if categorical_columns:
                # Let user select a column to filter by
                filter_column = st.selectbox(
                    "Filter by Column",
                    options=["None"] + categorical_columns
                )
                
                # If a column is selected, create a multiselect filter
                if filter_column != "None":
                    # Get unique values for the selected column
                    unique_values = st.session_state.data[filter_column].unique().tolist()
                    
                    # Create a multiselect filter
                    selected_values = st.multiselect(
                        f"Select {filter_column} Values",
                        options=unique_values,
                        default=unique_values
                    )
                    
                    # Apply the filter
                    if selected_values:
                        st.session_state.filtered_data = st.session_state.data[
                            st.session_state.data[filter_column].isin(selected_values)
                        ]
                    else:
                        st.session_state.filtered_data = st.session_state.data
                else:
                    st.session_state.filtered_data = st.session_state.data
            
            # Add date range filter if datetime columns exist
            datetime_columns = [
                col for col in st.session_state.data.columns 
                if pd.api.types.is_datetime64_any_dtype(st.session_state.data[col])
            ]
            
            if datetime_columns:
                # Let user select a datetime column
                date_column = st.selectbox(
                    "Filter by Date",
                    options=["None"] + datetime_columns
                )
                
                # If a column is selected, create a date range filter
                if date_column != "None":
                    # Get min and max dates
                    min_date = st.session_state.data[date_column].min()
                    max_date = st.session_state.data[date_column].max()
                    
                    # Create a date range slider
                    date_range = st.date_input(
                        "Select Date Range",
                        value=(min_date, max_date),
                        min_value=min_date,
                        max_value=max_date
                    )
                    
                    # Apply the filter if two dates are selected
                    if len(date_range) == 2:
                        start_date, end_date = date_range
                        st.session_state.filtered_data = st.session_state.filtered_data[
                            (st.session_state.filtered_data[date_column] >= pd.Timestamp(start_date)) &
                            (st.session_state.filtered_data[date_column] <= pd.Timestamp(end_date))
                        ]
            
            # Display filter status
            if "filtered_data" in st.session_state:
                st.write(f"Showing {len(st.session_state.filtered_data)} of {len(st.session_state.data)} rows")
        
        # About section
        st.subheader("About")
        st.write("Interactive Data Analysis Dashboard v1.0")
        st.write("Created with Streamlit")
    
    return page
def create_main_content(page):
    # Check if dataframe exists in session state
    if 'dataframe' not in st.session_state:
        st.session_state['dataframe'] = None
        
    df = st.session_state['dataframe']
    
    if page == "Home":
        home_page()
    elif page == "Data Loading":
        # Update the dataframe in session state with the loaded data
        df = load_data_page()
        st.session_state['dataframe'] = df
    elif page == "Data Manipulation":
        data_manipulation_page(df)
    elif page == "Data Visualization":
        data_visualization_page(df)
    elif page == "Analysis":
        data_analize_page(df)    
    elif page == "About":
        about_page()
# Main dashboard application
def run_dashboard():
    # Initialize session state
    initialize_session_state()
    
    # Set up responsive layout
    #create_responsive_layout() 
    
    # Create sidebar and get selected page
    page = create_sidebar()
    
    # Update current page in session state
    if page != st.session_state.current_page:
        st.session_state.current_page = page
        log_activity(f"Navigated to {page} page")
    
    # Create main content based on selected page
    create_main_content(page)

# Run the dashboard
if __name__ == "__main__":
    run_dashboard()
