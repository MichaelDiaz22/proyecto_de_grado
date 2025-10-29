import streamlit as st
import pandas as pd
import joblib
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Load the saved pipeline, scaler, and fitted PCA
try:
    # Intentar cargar los modelos
    pipeline = joblib.load('best_model_pipeline.joblib')
    fitted_pca = joblib.load('pca.joblib')
    
    st.success("✅ Modelos cargados exitosamente")
    
    # DEBUG: Mostrar información del pipeline
    st.write("**Información del Pipeline:**")
    st.write(f"Steps del pipeline: {[name for name, _ in pipeline.steps]}")
    
    # Extraer componentes del pipeline de manera más robusta
    loaded_scaler = None
    loaded_model = None
    
    for step_name, step_object in pipeline.steps:
        st.write(f"Step: {step_name}, Tipo: {type(step_object)}")
        if hasattr(step_object, 'transform') and hasattr(step_object, 'fit'):
            if hasattr(step_object, 'mean_') or hasattr(step_object, 'scale_'):
                loaded_scaler = step_object
                st.success(f"✅ Scaler encontrado en step: {step_name}")
            else:
                loaded_model = step_object
                st.success(f"✅ Modelo encontrado en step: {step_name}")
    
    # Si no encontramos el scaler en el pipeline, intentar cargarlo por separado
    if loaded_scaler is None:
        st.warning("⚠️ No se encontró scaler en el pipeline, intentando cargar por separado...")
        try:
            loaded_scaler = joblib.load('scaler.joblib')
            st.success("✅ Scaler cargado por separado")
        except:
            st.error("❌ No se pudo cargar el scaler")
    
    if loaded_model is None:
        st.error("❌ No se pudo cargar el modelo")
        st.stop()
        
    if loaded_scaler is None:
        st.error("❌ No se pudo cargar el scaler")
        st.stop()

except FileNotFoundError as e:
    st.error(f"❌ Error: No se encontraron los archivos necesarios. {e}")
    st.error("Asegúrate de que los archivos 'best_model_pipeline.joblib' y 'pca.joblib' están en el mismo directorio.")
    st.stop()
except Exception as e:
    st.error(f"❌ Error al cargar los modelos: {e}")
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
        st.write(f"📊 Archivo cargado: {uploaded_file.name}")
        st.write(f"📐 Dimensiones del dataset: {df_uploaded.shape}")
        st.write("**Primeras filas del dataset:**")
        st.dataframe(df_uploaded.head())

        # --- Preprocessing steps (must match the training pipeline) ---

        # 1. Handle missing values and drop columns as done in the notebook
        cols_to_drop = ['2025-10', '2025-11', '2025-12', 'CUM']
        df_processed = df_uploaded.drop(columns=[col for col in cols_to_drop if col in df_uploaded.columns])
        st.write(f"📐 Después de eliminar columnas: {df_processed.shape}")

        # Drop rows with missing values in specific columns
        cols_to_dropna = [
            'CODIGO_ALMACEN', 'ALMACEN', 'CODIGO_PRODUCTO', 'DESCRIPCION_PRODUCTO',
            'TIPO_PRODUCTO', 'VALOR_PROMEDIO', 'VALOR_FINAL'
        ] + [col for col in df_processed.columns if col.startswith('2024-') or 
             (col.startswith('2025-') and col not in ['2025-10', '2025-11', '2025-12'])]
        
        cols_to_dropna = [col for col in cols_to_dropna if col in df_processed.columns]
        df_processed = df_processed.dropna(subset=cols_to_dropna)
        st.write(f"📐 Después de eliminar filas con NaN: {df_processed.shape}")

        # Impute missing values for 'CODIGO_PADRE' and 'DESCRIPCION_PADRE'
        if 'CODIGO_PADRE' in df_processed.columns:
             df_processed['CODIGO_PADRE'] = df_processed['CODIGO_PADRE'].fillna('No aplica')
        if 'DESCRIPCION_PADRE' in df_processed.columns:
            df_processed['DESCRIPCION_PADRE'] = df_processed['DESCRIPCION_PADRE'].fillna('No aplica')

        # 2. Create new aggregated features
        monthly_cols_2024 = [f'2024-{i:02d}' for i in range(1, 13)]
        monthly_cols_2025 = [f'2025-{i:02d}' for i in range(1, 10)]
        all_monthly_cols = monthly_cols_2024 + monthly_cols_2025

        # Ensure monthly columns are numeric, fillna(0) for aggregation
        for col in all_monthly_cols:
            if col in df_processed.columns:
                df_processed[col] = pd.to_numeric(df_processed[col], errors='coerce').fillna(0)
            else:
                df_processed[col] = 0

        # Create aggregated columns
        df_processed['TOTAL_CANTIDAD_2024'] = df_processed[monthly_cols_2024].sum(axis=1)
        df_processed['PROMEDIO_CANTIDAD_2024'] = df_processed[monthly_cols_2024].mean(axis=1)
        df_processed['TOTAL_CANTIDAD_2025'] = df_processed[monthly_cols_2025].sum(axis=1)
        df_processed['PROMEDIO_CANTIDAD_2025'] = df_processed[monthly_cols_2025].mean(axis=1)

        st.write("✅ Características agregadas creadas")

        # 3. Select the numerical columns used for PCA
        numerical_cols_for_pca = ['VALOR_PROMEDIO', 'VALOR_FINAL', 'TOTAL_CANTIDAD_2024', 
                                 'PROMEDIO_CANTIDAD_2024', 'TOTAL_CANTIDAD_2025', 
                                 'PROMEDIO_CANTIDAD_2025'] + monthly_cols_2024 + monthly_cols_2025

        numerical_cols_for_pca = [col for col in numerical_cols_for_pca if col in df_processed.columns]
        
        st.write(f"🔢 Columnas numéricas para PCA: {len(numerical_cols_for_pca)}")
        st.write(numerical_cols_for_pca)

        if len(numerical_cols_for_pca) == 0:
            st.error("❌ Error: No se encontraron las columnas numéricas necesarias para el modelo.")
            st.stop()

        # Separate the numerical data for scaling and PCA
        df_numerical_processed = df_processed[numerical_cols_for_pca]

        # 4. Verificar y aplicar el StandardScaler
        st.write("🔧 Aplicando StandardScaler...")
        
        # Verificar que el scaler esté ajustado
        if not hasattr(loaded_scaler, 'mean_'):
            st.warning("⚠️ El scaler no está ajustado. Ajustando con los datos actuales...")
            # En caso de emergencia, ajustar con los datos actuales
            loaded_scaler.fit(df_numerical_processed)
            st.success("✅ Scaler ajustado con los datos actuales")
        
        df_scaled_processed = loaded_scaler.transform(df_numerical_processed)
        df_scaled_processed = pd.DataFrame(df_scaled_processed, 
                                         columns=numerical_cols_for_pca, 
                                         index=df_processed.index)
        st.write("✅ Scaler aplicado exitosamente")

        # 5. Apply the loaded PCA transformation
        st.write("🔧 Aplicando PCA...")
        df_pca_processed = fitted_pca.transform(df_scaled_processed)
        df_pca_processed = pd.DataFrame(df_pca_processed, 
                                      columns=[f'PC{i+1}' for i in range(fitted_pca.n_components_)], 
                                      index=df_processed.index)
        st.write("✅ PCA aplicado exitosamente")

        # Identify and include binary columns
        binary_cols = [col for col in df_processed.columns if col not in numerical_cols_for_pca and df_processed[col].nunique() <= 2]
        df_binary_processed = df_processed[binary_cols]
        
        st.write(f"🔢 Columnas binarias identificadas: {len(binary_cols)}")

        # Concatenate PCA components with binary columns
        df_final_features = pd.concat([df_pca_processed, df_binary_processed], axis=1)

        # --- Make predictions using the loaded model ---
        st.write("🔮 Generando predicciones...")
        predictions = loaded_model.predict(df_final_features)
        st.write("✅ Predicciones generadas exitosamente")

        # Add the predictions to the original processed DataFrame
        df_processed['Proyección de consumo próximo mes (Predicción)'] = predictions

        # --- Output the results ---
        st.subheader("🎯 Predicciones Generadas")
        st.write(f"📊 Se han generado predicciones para {len(df_processed)} registros.")
        
        # Mostrar estadísticas básicas de las predicciones
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Mínimo", f"{predictions.min():.2f}")
        with col2:
            st.metric("Máximo", f"{predictions.max():.2f}")
        with col3:
            st.metric("Promedio", f"{predictions.mean():.2f}")
        
        st.write("**Vista previa de los resultados:**")
        st.dataframe(df_processed[['CODIGO_PRODUCTO', 'DESCRIPCION_PRODUCTO', 'Proyección de consumo próximo mes (Predicción)']].head())

        # Provide download link for the updated Excel file
        output_filename = "historico_con_proyeccion.xlsx"
        
        with pd.ExcelWriter(output_filename, engine='openpyxl') as writer:
            df_processed.to_excel(writer, index=False)

        with open(output_filename, 'rb') as f:
            st.download_button(
                label="📥 Descargar archivo Excel con predicciones",
                data=f,
                file_name=output_filename,
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

    except Exception as e:
        st.error(f"❌ Ocurrió un error durante el procesamiento: {e}")
        import traceback
        st.error(f"Detalles del error: {traceback.format_exc()}")

else:
    st.info("📁 Por favor, sube un archivo Excel para comenzar.")
