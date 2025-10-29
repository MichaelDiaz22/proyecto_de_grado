import streamlit as st
import pandas as pd
import joblib
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Load the saved pipeline, scaler, and fitted PCA
try:
    pipeline = joblib.load('best_model_pipeline.joblib')
    fitted_pca = joblib.load('pca.joblib') # Assuming pca.joblib was saved
    
    # Corrección: Verificar y extraer correctamente los componentes del pipeline
    if hasattr(pipeline, 'steps'):
        # Obtener todos los steps del pipeline
        loaded_scaler = None
        loaded_model = None
        
        for step_name, step_object in pipeline.steps:
            if isinstance(step_object, StandardScaler):
                loaded_scaler = step_object
            # Asumir que el último step es el modelo
            loaded_model = step_object
        
        # Verificar que tenemos ambos componentes
        if loaded_scaler is None:
            st.error("Error: No se encontró StandardScaler en el pipeline.")
            st.stop()
        
        if loaded_model is None:
            st.error("Error: No se encontró el modelo en el pipeline.")
            st.stop()
            
    else:
        st.error("Error: El pipeline cargado no tiene la estructura esperada.")
        st.stop()

except FileNotFoundError as e:
    st.error(f"Error: No se encontraron los archivos necesarios. {e}")
    st.error("Asegúrate de que los archivos 'best_model_pipeline.joblib' y 'pca.joblib' están en el mismo directorio.")
    st.stop()
except Exception as e:
    st.error(f"Error al cargar los modelos: {e}")
    st.stop()

st.title('Proyección de Consumo del Próximo Mes')

st.write("""
Esta aplicación utiliza un modelo de Machine Learning entrenado para proyectar el consumo
del próximo mes basado en datos históricos de productos y almacenes.
""")

uploaded_file = st.file_uploader("Sube tu archivo Excel con los datos históricos", type=['xlsx'])

if uploaded_file is not None:
    try:
        # Read the Excel file
        df_uploaded = pd.read_excel(uploaded_file)
        
        # Mostrar información del archivo cargado
        st.write(f"Archivo cargado: {uploaded_file.name}")
        st.write(f"Dimensiones del dataset: {df_uploaded.shape}")

        # --- Preprocessing steps (must match the training pipeline) ---

        # 1. Handle missing values and drop columns as done in the notebook
        # Drop columns with all missing values ('2025-10', '2025-11', '2025-12') and 'CUM'
        cols_to_drop = ['2025-10', '2025-11', '2025-12', 'CUM']
        df_processed = df_uploaded.drop(columns=[col for col in cols_to_drop if col in df_uploaded.columns])

        # Drop rows with missing values in specific columns
        cols_to_dropna = [
            'CODIGO_ALMACEN', 'ALMACEN', 'CODIGO_PRODUCTO', 'DESCRIPCION_PRODUCTO',
            'TIPO_PRODUCTO', 'VALOR_PROMEDIO', 'VALOR_FINAL'
        ] + [col for col in df_processed.columns if col.startswith('2024-') or (col.startswith('2025-') and col not in ['2025-10', '2025-11', '2025-12'])]
        
        cols_to_dropna = [col for col in cols_to_dropna if col in df_processed.columns]
        df_processed = df_processed.dropna(subset=cols_to_dropna)

        # Impute missing values for 'CODIGO_PADRE' and 'DESCRIPCION_PADRE'
        if 'CODIGO_PADRE' in df_processed.columns:
             df_processed['CODIGO_PADRE'] = df_processed['CODIGO_PADRE'].fillna('No aplica')
        if 'DESCRIPCION_PADRE' in df_processed.columns:
            df_processed['DESCRIPCION_PADRE'] = df_processed['DESCRIPCION_PADRE'].fillna('No aplica')

        # 2. Create new aggregated features (must match the notebook)
        monthly_cols_2024 = [f'2024-{i:02d}' for i in range(1, 13)]
        monthly_cols_2025 = [f'2025-{i:02d}' for i in range(1, 10)] # Up to 2025-9 based on notebook
        all_monthly_cols = monthly_cols_2024 + monthly_cols_2025

        # Ensure monthly columns are numeric, fillna(0) for aggregation
        for col in all_monthly_cols:
            if col in df_processed.columns:
                df_processed[col] = pd.to_numeric(df_processed[col], errors='coerce').fillna(0)
            else:
                # Add missing monthly columns and fill with 0 if they are not in the uploaded data
                df_processed[col] = 0

        # Create aggregated columns, ensuring they are added only if they don't exist or are recalculated
        df_processed['TOTAL_CANTIDAD_2024'] = df_processed[monthly_cols_2024].sum(axis=1)
        df_processed['PROMEDIO_CANTIDAD_2024'] = df_processed[monthly_cols_2024].mean(axis=1)
        df_processed['TOTAL_CANTIDAD_2025'] = df_processed[monthly_cols_2025].sum(axis=1)
        df_processed['PROMEDIO_CANTIDAD_2025'] = df_processed[monthly_cols_2025].mean(axis=1)

        # 3. Select the numerical columns used for PCA
        numerical_cols_for_pca = ['VALOR_PROMEDIO', 'VALOR_FINAL', 'TOTAL_CANTIDAD_2024', 'PROMEDIO_CANTIDAD_2024', 'TOTAL_CANTIDAD_2025', 'PROMEDIO_CANTIDAD_2025'] + \
                                 monthly_cols_2024 + monthly_cols_2025

        # Ensure all numerical columns for PCA exist in the processed data
        numerical_cols_for_pca = [col for col in numerical_cols_for_pca if col in df_processed.columns]

        # Verificar que tenemos columnas numéricas
        if len(numerical_cols_for_pca) == 0:
            st.error("Error: No se encontraron las columnas numéricas necesarias para el modelo.")
            st.stop()

        # Separate the numerical data for scaling and PCA
        df_numerical_processed = df_processed[numerical_cols_for_pca]

        # 4. Apply the loaded StandardScaler (VERIFICAR QUE ESTÁ AJUSTADO)
        if not hasattr(loaded_scaler, 'mean_'):
            st.error("Error: El StandardScaler no está ajustado. Debe ser entrenado antes de su uso.")
            st.stop()
            
        df_scaled_processed = loaded_scaler.transform(df_numerical_processed)
        df_scaled_processed = pd.DataFrame(df_scaled_processed, columns=numerical_cols_for_pca, index=df_processed.index)

        # 5. Apply the loaded PCA transformation
        df_pca_processed = fitted_pca.transform(df_scaled_processed)
        df_pca_processed = pd.DataFrame(df_pca_processed, columns=[f'PC{i+1}' for i in range(fitted_pca.n_components_)], index=df_processed.index)

        # Identify and include binary columns from the uploaded data
        binary_cols = [col for col in df_processed.columns if col not in numerical_cols_for_pca and df_processed[col].nunique() <= 2]
        df_binary_processed = df_processed[binary_cols]

        # Concatenate PCA components with binary columns
        df_final_features = pd.concat([df_pca_processed, df_binary_processed], axis=1)

        # --- Make predictions using the loaded model ---
        predictions = loaded_model.predict(df_final_features)

        # Add the predictions to the original processed DataFrame
        df_processed['Proyección de consumo próximo mes (Predicción)'] = predictions

        # --- Output the results ---

        st.subheader("Predicciones Generadas")
        st.write(f"Se han generado predicciones para {len(df_processed)} registros.")
        st.write("Se ha añadido la columna 'Proyección de consumo próximo mes (Predicción)' al archivo.")
        
        # Mostrar estadísticas básicas de las predicciones
        st.write(f"**Estadísticas de las predicciones:**")
        st.write(f"- Mínimo: {predictions.min():.2f}")
        st.write(f"- Máximo: {predictions.max():.2f}")
        st.write(f"- Promedio: {predictions.mean():.2f}")
        
        st.dataframe(df_processed.head()) # Use st.dataframe for Streamlit

        # Provide download link for the updated Excel file
        output_filename = "historico_con_proyeccion.xlsx"
        # Use ExcelWriter to save the DataFrame to an Excel file
        with pd.ExcelWriter(output_filename, engine='openpyxl') as writer:
            df_processed.to_excel(writer, index=False)

        with open(output_filename, 'rb') as f:
            st.download_button(
                label="Descargar archivo Excel con predicciones",
                data=f,
                file_name=output_filename,
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

    except Exception as e:
        st.error(f"Ocurrió un error durante el procesamiento: {e}")
        st.error("Por favor, verifica que el archivo tenga el formato correcto.")

else:
    st.info("Por favor, sube un archivo Excel para comenzar.")
