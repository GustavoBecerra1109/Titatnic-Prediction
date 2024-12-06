import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Configuración de la página
st.set_page_config(page_title="Predictor de Supervivencia del Titanic", layout="wide")

# Cargar modelos y datos de preprocesamiento
@st.cache_resource
def load_all():
    files_to_load = {
        'best_rf_model.pkl': 'Random Forest model',
        'best_gb_model.pkl': 'Gradient Boosting model',
        'scaler.pkl': 'Standard Scaler',
        'preprocessing_info.pkl': 'Preprocessing information',
        'model_info.pkl': 'Model information'
    }
    
    loaded_data = {}
    for filename, description in files_to_load.items():
        try:
            with open(filename, 'rb') as f:
                loaded_data[filename] = pickle.load(f)
        except Exception as e:
            st.error(f"Error loading {description}: {str(e)}")
            raise
    
    return (loaded_data['best_rf_model.pkl'], 
            loaded_data['best_gb_model.pkl'],
            loaded_data['scaler.pkl'],
            loaded_data['preprocessing_info.pkl'],
            loaded_data['model_info.pkl'])

# Cargar todos los modelos y datos
rf_model, gb_model, scaler, preprocessing_info, model_info = load_all()

# Función para procesar los datos de entrada
def process_input_data(input_data):
    # Crear DataFrame con las características correctas
    df = pd.DataFrame([input_data])
    
    # Asegurarse de que todas las características necesarias estén presentes
    for feature in preprocessing_info['feature_names']:
        if feature not in df.columns:
            df[feature] = 0
    
    # Reordenar columnas según el orden original
    df = df[preprocessing_info['feature_names']]
    
    return df

st.title('Predicción de Supervivencia en el Titanic')

# Crear tabs
tab1, tab2 = st.tabs(["Predicción", "Información del Modelo"])

with tab1:
    st.write("""
    ### Ingrese los datos del pasajero
    Complete la siguiente información para predecir la probabilidad de supervivencia.
    """)

    # Crear dos columnas para el formulario
    col1, col2 = st.columns(2)

    with col1:
        pclass = st.selectbox('Clase del Pasajero', [1, 2, 3], 
                            help="1 = Primera clase, 2 = Segunda clase, 3 = Tercera clase")
        sex = st.selectbox('Género', ['male', 'female'])
        age = st.number_input('Edad', min_value=0, max_value=100, value=30)
        sibsp = st.number_input('Número de hermanos/cónyuge a bordo', min_value=0, max_value=10, value=0)

    with col2:
        parch = st.number_input('Número de padres/hijos a bordo', min_value=0, max_value=10, value=0)
        fare = st.number_input('Tarifa del pasaje', min_value=0.0, max_value=500.0, value=32.0)
        embarked = st.selectbox('Puerto de embarque', 
                              preprocessing_info['embarked_categories'],
                              help="C = Cherbourg, Q = Queenstown, S = Southampton")

    if st.button('Realizar Predicción', type='primary'):
        # Preparar datos
        input_data = {
            'Pclass': pclass,
            'Sex': preprocessing_info['sex_mapping'][sex],
            'Age': age,
            'SibSp': sibsp,
            'Parch': parch,
            'Fare': fare
        }
        
        # Agregar variables dummy para Embarked
        for port in preprocessing_info['embarked_categories']:
            input_data[f'Embarked_{port}'] = 1 if embarked == port else 0
        
        # Procesar y escalar datos
        processed_data = process_input_data(input_data)
        scaled_data = scaler.transform(processed_data)
        
        # Realizar predicciones
        rf_prob = rf_model.predict_proba(scaled_data)[0]
        gb_prob = gb_model.predict_proba(scaled_data)[0]
        
        # Mostrar resultados
        st.write('### Resultados de la Predicción')
        
        col1, col2 = st.columns(2)
        
        def create_gauge_chart(prob, title):
            return f"""
            <div style="text-align: center;">
                <h4>{title}</h4>
                <div style="margin: 20px auto; width: 200px; height: 100px; position: relative;">
                    <div style="position: absolute; width: 100%; height: 50px; background: linear-gradient(90deg, #ff0000 0%, #ffff00 50%, #00ff00 100%); border-radius: 25px;">
                        <div style="position: absolute; left: calc({prob * 100}% - 2px); top: 0; width: 4px; height: 50px; background-color: black;"></div>
                    </div>
                    <div style="position: absolute; width: 100%; text-align: center; top: 60px;">
                        Probabilidad de supervivencia: {prob:.1%}
                    </div>
                </div>
            </div>
            """
        
        with col1:
            st.markdown(create_gauge_chart(rf_prob[1], "Random Forest"), unsafe_allow_html=True)
            
        with col2:
            st.markdown(create_gauge_chart(gb_prob[1], "Gradient Boosting"), unsafe_allow_html=True)
        
        # Interpretación
        st.write('### Interpretación')
        rf_survival = "sobreviviría" if rf_prob[1] > 0.5 else "no sobreviviría"
        gb_survival = "sobreviviría" if gb_prob[1] > 0.5 else "no sobreviviría"
        
        st.write(f"• Random Forest predice que el pasajero {rf_survival} al desastre.")
        st.write(f"• Gradient Boosting predice que el pasajero {gb_survival} al desastre.")

with tab2:
    st.write("### Información del Modelo")
    
    # Mostrar métricas de importancia de características
    st.write("#### Importancia de las Características")
    col1, col2 = st.columns(2)
    
    def plot_feature_importance(importance_dict, title):
        importance_df = pd.DataFrame(
            importance_dict.items(),
            columns=['Característica', 'Importancia']
        ).sort_values('Importancia', ascending=True)
        
        return importance_df.set_index('Característica').plot(
            kind='barh',
            title=title,
            figsize=(10, 6)
        )
    
    with col1:
        st.write("Random Forest:")
        importance_plot_rf = plot_feature_importance(
            model_info['feature_importance_rf'],
            'Importancia de Características (Random Forest)'
        )
        st.pyplot(importance_plot_rf.figure)
        
    with col2:
        st.write("Gradient Boosting:")
        importance_plot_gb = plot_feature_importance(
            model_info['feature_importance_gb'],
            'Importancia de Características (Gradient Boosting)'
        )
        st.pyplot(importance_plot_gb.figure)
    
    # Mostrar parámetros óptimos
    st.write("#### Parámetros Óptimos de los Modelos")
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("Random Forest:")
        st.json(model_info['rf_params'])
        
    with col2:
        st.write("Gradient Boosting:")
        st.json(model_info['gb_params'])

    # Agregar información adicional
    st.write("### Notas sobre el Modelo")
    st.write("""
    - Los modelos fueron entrenados usando validación cruzada y optimización de hiperparámetros con Optuna.
    - Las características fueron escaladas usando StandardScaler.
    - Los valores faltantes fueron imputados usando la mediana para variables numéricas.
    - Las variables categóricas fueron codificadas usando one-hot encoding.
    """)