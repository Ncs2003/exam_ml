import streamlit as st
import pandas as pd
import requests
from io import BytesIO

# --- Configuration page ---
st.set_page_config(
    page_title="Détecteur de Faux Billets",
    page_icon="💶",
    layout="wide"
)

# --- CSS personnalisé ---
st.markdown("""
<style>
.main-header {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 2rem;
    border-radius: 10px;
    margin-bottom: 2rem;
    text-align: center;
    color: white;
}
.metric-card {
    background: white;
    padding: 1.5rem;
    border-radius: 10px;
    box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    border-left: 4px solid #667eea;
    margin: 1rem 0;
}
.upload-section {
    background: #f8f9fa;
    padding: 2rem;
    border-radius: 10px;
    border: 2px dashed #dee2e6;
    text-align: center;
    margin: 2rem 0;
}
.results-section {
    background: white;
    padding: 1.5rem;
    border-radius: 10px;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    margin: 1rem 0;
}
</style>
""", unsafe_allow_html=True)

# --- Header ---
st.markdown("""
<div class="main-header">
    <h1>💶 Détecteur de Faux Billets</h1>
    <p>Analysez vos billets avec notre IA avancée</p>
</div>
""", unsafe_allow_html=True)

# --- Sidebar ---
st.sidebar.markdown("## ⚙️ Configuration API")
api_url = st.sidebar.text_input("URL de votre API FastAPI", "https://exam-ml-6.onrender.com/predict_file")

st.sidebar.markdown("## 🔗 Test API")
if st.sidebar.button("Tester la connexion"):
    try:
        test_response = requests.get("http://127.0.0.1:8000/docs", timeout=5)
        if test_response.status_code == 200:
            st.sidebar.success("✅ API accessible")
        else:
            st.sidebar.warning("⚠️ Problème API")
    except:
        st.sidebar.error("❌ API non accessible")

st.sidebar.markdown("## 📊 Colonnes attendues")
st.sidebar.text("diagonal, height_left, height_right, margin_low, margin_up, length")

# --- Upload CSV ---
st.markdown('<div class="upload-section">', unsafe_allow_html=True)
st.markdown("### 📁 Upload de votre fichier CSV")
uploaded_file = st.file_uploader("Choisissez un fichier CSV", type=['csv'])
st.markdown('</div>', unsafe_allow_html=True)

if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)
        # Aperçu
        st.markdown("### 👀 Aperçu des données")
        st.dataframe(df.head(10))

        if st.button("🔍 Lancer la détection"):
            with st.spinner("Analyse en cours..."):
                try:
                    uploaded_file.seek(0)  # 👈 Important sinon le fichier est vide
                    files = {"file": (uploaded_file.name, uploaded_file, "text/csv")}
                    response = requests.post(api_url, files=files, timeout=30)
                    
                    if response.status_code == 200:
                        data = response.json()
                        if "predictions" in data:
                            df['Prédiction'] = data['predictions']
                            # ⚡ Inversion des statuts pour correspondre au modèle
                            df['Statut'] = df['Prédiction'].apply(lambda x: "✅ Authentique" if x==1 else "❌ Faux billet")
                            
                            # Métriques
                            total = len(df)
                            auth = sum(df['Prédiction'])       # 1 = Authentique
                            faux = total - auth
                            taux_fraude = (faux / total) * 100 if total > 0 else 0
                            
                            col1, col2, col3, col4 = st.columns(4)
                            col1.metric("Total analysés", total)
                            col2.metric("Authentiques", auth)
                            col3.metric("Faux billets", faux)
                            col4.metric("Taux de fraude", f"{taux_fraude:.1f}%")
                            
                            # 🔹 Graphique sans Plotly
                            if total > 0:
                                chart_data = pd.DataFrame({
                                    'Statut': ['Authentiques', 'Faux billets'],
                                    'Nombre': [auth, faux]
                                })
                                st.bar_chart(data=chart_data.set_index('Statut'))
                            
                            # Tableau filtrable
                            st.markdown("### 📋 Résultats détaillés")
                            filter_option = st.selectbox("Filtrer par statut", ["Tous", "Authentiques uniquement", "Faux billets uniquement"])
                            if filter_option == "Authentiques uniquement":
                                filtered_df = df[df['Prédiction']==1]
                            elif filter_option == "Faux billets uniquement":
                                filtered_df = df[df['Prédiction']==0]
                            else:
                                filtered_df = df
                            
                            st.dataframe(filtered_df, use_container_width=True)
                            
                            # Téléchargement CSV
                            csv_buffer = BytesIO()
                            filtered_df.to_csv(csv_buffer, index=False, encoding='utf-8')
                            csv_buffer.seek(0)
                            st.download_button("📥 Télécharger les résultats", data=csv_buffer.getvalue(), file_name="predictions_billets.csv", mime="text/csv")
                            
                            # Alertes
                            if faux > 0:
                                st.warning(f"⚠️ {faux} faux billet(s) détecté(s) !")
                            else:
                                st.success("🎉 Aucun faux billet détecté !")
                        else:
                            st.error(data.get("error", "Erreur inconnue"))
                    else:
                        st.error(f"Erreur API : {response.status_code}")
                except requests.exceptions.RequestException as e:
                    st.error(f"Erreur de connexion : {e}")
    except Exception as e:
        st.error(f"Erreur lecture CSV : {e}")

else:
    st.markdown('<div class="results-section">', unsafe_allow_html=True)
    st.markdown("""
    <h3>🚀 Comment utiliser cette application ?</h3>
    <ol>
        <li>Configurez l'URL de votre API</li>
        <li>Testez la connexion</li>
        <li>Uploadez votre fichier CSV</li>
        <li>Lancez la détection</li>
        <li>Analysez les résultats et téléchargez le CSV</li>
    </ol>
    """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("💶 Détecteur de Faux Billets - Powered by Streamlit & FastAPI")
