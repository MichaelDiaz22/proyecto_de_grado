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
    
    # Extraer componentes del pipeline
    loaded_scaler = None
    loaded_model = None
    
    for step_name, step_object in pipeline.steps:
        st.write(f"Step: {step_name}, Tipo: {type(step_object)}")
        if step_name == 'standardscaler':
            loaded_scaler = step_object
            st.success(f"✅ Scaler encontrado en step: {step_name}")
        elif step_name == 'mlpregressor':
            loaded_model = step_object
            st.success(f"✅ Modelo encontrado en step: {step_name}")
    
    # Verificar que tenemos ambos componentes
    if loaded_scaler is None:
        st.error("❌ No se encontró el scaler en el pipeline")
        st.stop()
        
    if loaded_model is None:
        st.error("❌ No se encontró el modelo en el pipeline")
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

        # 2. Normalizar nombres de columnas de meses para que coincidan con el entrenamiento
        # Convertir formato 2024-01 a 2024-1, 2024-02 a 2024-2, etc.
        month_columns_mapping = {}
        for col in df_processed.columns:
            if col.startswith('2024-') or col.startswith('2025-'):
                try:
                    year, month = col.split('-')
                    # Convertir mes con ceros a la izquierda a formato sin ceros
                    month_normalized = str(int(month))
                    new_col_name = f"{year}-{month_normalized}"
                    month_columns_mapping[col] = new_col_name
                except:
                    continue
        
        # Renombrar las columnas
        df_processed = df_processed.rename(columns=month_columns_mapping)
        st.write("✅ Nombres de columnas de meses normalizados")

        # 3. Crear nuevas características agregadas
        monthly_cols_2024 = [f'2024-{i}' for i in range(1, 13)]
        monthly_cols_2025 = [f'2025-{i}' for i in range(1, 10)]
        all_monthly_cols = monthly_cols_2024 + monthly_cols_2025

        # Asegurar que las columnas mensuales sean numéricas y manejar valores faltantes
        for col in all_monthly_cols:
            if col in df_processed.columns:
                df_processed[col] = pd.to_numeric(df_processed[col], errors='coerce').fillna(0)
            else:
                # Si la columna no existe, crearla con ceros
                df_processed[col] = 0

        # Crear columnas agregadas
        df_processed['TOTAL_CANTIDAD_2024'] = df_processed[monthly_cols_2024].sum(axis=1)
        df_processed['PROMEDIO_CANTIDAD_2024'] = df_processed[monthly_cols_2024].mean(axis=1)
        df_processed['TOTAL_CANTIDAD_2025'] = df_processed[monthly_cols_2025].sum(axis=1)
        df_processed['PROMEDIO_CANTIDAD_2025'] = df_processed[monthly_cols_2025].mean(axis=1)

        st.write("✅ Características agregadas creadas")

        # 4. Seleccionar columnas numéricas para PCA y eliminar columnas vacías
        numerical_cols_for_pca = ['VALOR_PROMEDIO', 'VALOR_FINAL', 'TOTAL_CANTIDAD_2024', 
                                 'PROMEDIO_CANTIDAD_2024', 'TOTAL_CANTIDAD_2025', 
                                 'PROMEDIO_CANTIDAD_2025'] + monthly_cols_2024 + monthly_cols_2025

        # Filtrar columnas que existen en los datos procesados
        numerical_cols_for_pca = [col for col in numerical_cols_for_pca if col in df_processed.columns]
        
        # Identificar y eliminar columnas vacías o con varianza cero
        non_empty_cols = []
        empty_cols = []
        
        for col in numerical_cols_for_pca:
            if col in df_processed.columns:
                # Verificar si la columna no está vacía y tiene varianza > 0
                if df_processed[col].notna().any() and df_processed[col].var() > 0:
                    non_empty_cols.append(col)
                else:
                    empty_cols.append(col)
        
        numerical_cols_for_pca = non_empty_cols
        
        st.write(f"🔢 Columnas numéricas para PCA: {len(numerical_cols_for_pca)}")
        st.write(f"🗑️ Columnas vacías eliminadas: {len(empty_cols)}")
        if empty_cols:
            st.write(f"Columnas eliminadas: {empty_cols}")

        if len(numerical_cols_for_pca) == 0:
            st.error("❌ Error: No se encontraron columnas numéricas válidas para el modelo.")
            st.stop()

        # Separar los datos numéricos para escalado y PCA
        df_numerical_processed = df_processed[numerical_cols_for_pca]

        # 5. Aplicar el StandardScaler
        st.write("🔧 Aplicando StandardScaler...")
        
        # Verificar que el scaler esté ajustado
        if not hasattr(loaded_scaler, 'mean_'):
            st.warning("⚠️ El scaler no está ajustado. Ajustando con los datos actuales...")
            loaded_scaler.fit(df_numerical_processed)
            st.success("✅ Scaler ajustado con los datos actuales")
        
        df_scaled_processed = loaded_scaler.transform(df_numerical_processed)
        df_scaled_processed = pd.DataFrame(df_scaled_processed, 
                                         columns=numerical_cols_for_pca, 
                                         index=df_processed.index)
        st.write("✅ Scaler aplicado exitosamente")

        # 6. Aplicar la transformación PCA
        st.write("🔧 Aplicando PCA...")
        
        # Verificar que las columnas coincidan con las del entrenamiento
        pca_expected_features = fitted_pca.feature_names_in_ if hasattr(fitted_pca, 'feature_names_in_') else None
        
        if pca_expected_features is not None:
            missing_features = set(pca_expected_features) - set(numerical_cols_for_pca)
            extra_features = set(numerical_cols_for_pca) - set(pca_expected_features)
            
            if missing_features:
                st.warning(f"⚠️ Características faltantes para PCA: {list(missing_features)}")
            if extra_features:
                st.warning(f"⚠️ Características adicionales no usadas en PCA: {list(extra_features)}")
            
            # Reordenar columnas para que coincidan con el entrenamiento
            df_scaled_processed = df_scaled_processed.reindex(columns=pca_expected_features, fill_value=0)
        
        df_pca_processed = fitted_pca.transform(df_scaled_processed)
        df_pca_processed = pd.DataFrame(df_pca_processed, 
                                      columns=[f'PC{i+1}' for i in range(fitted_pca.n_components_)], 
                                      index=df_processed.index)
        st.write("✅ PCA aplicado exitosamente")

        # 7. Identificar e incluir columnas binarias
        binary_cols = [col for col in df_processed.columns if col not in numerical_cols_for_pca and df_processed[col].nunique() <= 2]
        df_binary_processed = df_processed[binary_cols]
        
        st.write(f"🔢 Columnas binarias identificadas: {len(binary_cols)}")

        # Concatenar componentes PCA con columnas binarias
        df_final_features = pd.concat([df_pca_processed, df_binary_processed], axis=1)

        # --- Hacer predicciones usando el modelo cargado ---
        st.write("🔮 Generando predicciones...")
        predictions = loaded_model.predict(df_final_features)
        st.write("✅ Predicciones generadas exitosamente")

        # Agregar las predicciones al DataFrame procesado original
        df_processed['Proyección de consumo próximo mes (Predicción)'] = predictions

        # --- Mostrar resultados ---
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
        columns_to_show = ['CODIGO_PRODUCTO', 'DESCRIPCION_PRODUCTO', 'Proyección de consumo próximo mes (Predicción)']
        available_columns = [col for col in columns_to_show if col in df_processed.columns]
        st.dataframe(df_processed[available_columns].head())

        # Proporcionar enlace de descarga para el archivo Excel actualizado
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
