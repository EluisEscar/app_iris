import streamlit as st
import joblib
import pickle
import numpy as np
import psycopg2

# Datos de conexión a Supabase
USER = "postgres.buphbbqktifykcmbnydd"
PASSWORD = "usilinoscloud"
HOST = "aws-1-us-east-1.pooler.supabase.com"
PORT = "6543"
DBNAME = "postgres"

# Configuración de la página
st.set_page_config(page_title="Predictor de Iris", page_icon="🌸")


# Función para conectarse a Supabase
def get_connection():
    return psycopg2.connect(
        user=USER,
        password=PASSWORD,
        host=HOST,
        port=PORT,
        dbname=DBNAME,
        sslmode="require"
    )


# Probar conexión
try:
    connection = get_connection()
    cursor = connection.cursor()

    cursor.execute("SELECT NOW();")
    result = cursor.fetchone()

    cursor.close()
    connection.close()

    st.success("Conectado correctamente a Supabase")
    st.write("Hora actual en Supabase:", result[0])

except Exception as e:
    st.error(f"Error de conexión con Supabase: {e}")
    result = None


# Función para cargar los modelos
@st.cache_resource
def load_models():
    try:
        model = joblib.load("components/iris_model.pkl")
        scaler = joblib.load("components/iris_scaler.pkl")

        with open("components/model_info.pkl", "rb") as f:
            model_info = pickle.load(f)

        return model, scaler, model_info

    except FileNotFoundError:
        st.error("No se encontraron los archivos del modelo en la carpeta 'components/'")
        return None, None, None


# Función para guardar predicción en Supabase
def save_prediction(
    sepal_length,
    sepal_width,
    petal_length,
    petal_width,
    predicted_species
):
    try:
        connection = get_connection()
        cursor = connection.cursor()

        query = """
            INSERT INTO iris_predictions (
                l_s,
                a_s,
                l_p,
                a_p,
                prediccion
            )
            VALUES (%s, %s, %s, %s, %s);
        """

        values = (
            sepal_length,
            sepal_width,
            petal_length,
            petal_width,
            predicted_species
        )

        cursor.execute(query, values)
        connection.commit()

        cursor.close()
        connection.close()

        return True

    except Exception as e:
        st.error(f"Error al guardar la predicción en Supabase: {e}")
        return False


# Título
st.title("🌸 Predictor de Especies de Iris")

# Cargar modelos
model, scaler, model_info = load_models()

if model is not None:
    # Inputs
    st.header("Ingresa las características de la flor:")

    sepal_length = st.number_input(
        "Longitud del Sépalo (cm)",
        min_value=0.0,
        max_value=10.0,
        value=5.0,
        step=0.1
    )

    sepal_width = st.number_input(
        "Ancho del Sépalo (cm)",
        min_value=0.0,
        max_value=10.0,
        value=3.0,
        step=0.1
    )

    petal_length = st.number_input(
        "Longitud del Pétalo (cm)",
        min_value=0.0,
        max_value=10.0,
        value=4.0,
        step=0.1
    )

    petal_width = st.number_input(
        "Ancho del Pétalo (cm)",
        min_value=0.0,
        max_value=10.0,
        value=1.0,
        step=0.1
    )

    # Botón de predicción
    if st.button("Predecir Especie"):
        # Preparar datos
        features = np.array([
            [sepal_length, sepal_width, petal_length, petal_width]
        ])

        # Estandarizar
        features_scaled = scaler.transform(features)

        # Predecir
        prediction = model.predict(features_scaled)[0]
        probabilities = model.predict_proba(features_scaled)[0]

        # Mostrar resultado
        target_names = model_info["target_names"]
        predicted_species = target_names[prediction]
        confidence = max(probabilities)

        st.success(f"Especie predicha: **{predicted_species}**")
        st.write(f"Confianza: **{confidence:.1%}**")

        # Mostrar todas las probabilidades
        st.write("Probabilidades:")

        for species, prob in zip(target_names, probabilities):
            st.write(f"- {species}: {prob:.1%}")

        # Guardar resultado en Supabase
        saved = save_prediction(
            sepal_length,
            sepal_width,
            petal_length,
            petal_width,
            predicted_species
        )

        if saved:
            st.success("Predicción guardada correctamente en Supabase.")
