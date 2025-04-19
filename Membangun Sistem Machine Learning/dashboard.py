import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import plotly.express as px
# from babel.numbers import format_currency
import locale
sns.set(style='dark')
import statsmodels.api as sm
import numpy as np
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import joblib

#import data
# Load dataset
ed = pd.read_csv('Data Employee Cleaned.csv')

# SIDEBAR
st.sidebar.title("HR Attrition Dashboard")
page = st.sidebar.radio("Pilih halaman", ["Overview", "Visualisasi", "Prediksi", "Rekomendasi"])

# 1. HALAMAN OVERVIEW
if page == "Overview":
    st.title("📋 HR Dashboard - Employee Attrition Overview")

    total_karyawan = len(ed)
    total_attrition = ed['Attrition'].sum()
    attrition_rate = total_attrition / total_karyawan * 100

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Karyawan", total_karyawan)
    col2.metric("Jumlah Resign", total_attrition)
    col3.metric("Attrition Rate", f"{attrition_rate:.2f}%")

    st.markdown("### 📊 Distribusi Attrition Berdasarkan Departemen")
    dept_attr = ed.groupby(['Department'])['Attrition'].mean().sort_values(ascending=False)
    fig1, ax1 = plt.subplots()
    sns.barplot(x=dept_attr.values, y=dept_attr.index, ax=ax1, palette='coolwarm')
    ax1.set_xlabel("Attrition Rate")
    ax1.set_ylabel("Department")
    st.pyplot(fig1)

    st.markdown("### 👤 Distribusi Attrition Berdasarkan Gender")
    gender_attr = ed.groupby(['Gender'])['Attrition'].mean()
    fig2, ax2 = plt.subplots()
    sns.barplot(x=gender_attr.index, y=gender_attr.values, ax=ax2)
    ax2.set_ylabel("Attrition Rate")
    st.pyplot(fig2)

    st.markdown("### 📈 Statistik Rata-rata antara Bertahan vs Resign")
    summary = ed.groupby('Attrition')[['Age', 'MonthlyIncome', 'TotalWorkingYears']].mean().rename(index={0: 'Bertahan', 1: 'Resign'})
    st.dataframe(summary.style.format("{:.2f}"))

    st.markdown("### 🧠 Insight Utama")
    st.info(
        f"""- Departemen dengan tingkat attrition tertinggi: **{dept_attr.idxmax()}** ({dept_attr.max()*100:.2f}%).
- Gender dengan attrition tertinggi: **{gender_attr.idxmax()}** ({gender_attr.max()*100:.2f}%).
- Karyawan yang keluar rata-rata lebih **muda**, **berpenghasilan lebih rendah**, dan memiliki **pengalaman kerja yang lebih sedikit** dibanding yang bertahan."""
    )

# Visualisasi Lanjutan
if page == "Visualisasi":
    st.title("📊 Visualisasi Faktor-Faktor Attrition")

    # --- 1. Korelasi Fitur terhadap Attrition ---
    st.subheader("1️⃣ Korelasi Fitur Numerik vs Attrition")
    numeric_cols = ed.select_dtypes(include=['int64', 'float64']).drop(columns=['EmployeeId']).columns
    correlations = ed[numeric_cols].corr()['Attrition'].sort_values(ascending=False)

    fig1, ax1 = plt.subplots(figsize=(10, 5))
    sns.barplot(x=correlations.values, y=correlations.index, palette='coolwarm', ax=ax1)
    ax1.set_title("Korelasi Fitur Numerik terhadap Attrition")
    st.pyplot(fig1)

    st.info(
        f"""**Insight:**
- Fitur yang memiliki korelasi positif tertinggi dengan attrition (semakin tinggi nilainya, makin tinggi risiko resign): **NumCompaniesWorked, DistanceFromHome, MonthlyRate**.
- Fitur yang berkorelasi negatif kuat (semakin tinggi nilainya, makin kecil risiko resign): **TotalWorkingYears, Age, JobLevel, StockOptionLevel**.
- Fitur-fitur ini dapat menjadi fokus utama HR dalam membuat strategi retention."""
    )

    # --- 2. Histogram Gaji vs Status Attrition ---
    st.subheader("2️⃣ Distribusi Gaji Bulanan Berdasarkan Status Attrition")
    fig2, ax2 = plt.subplots(figsize=(10, 5))
    sns.histplot(data=ed, x='MonthlyIncome', hue='Attrition', multiple='stack', bins=30, ax=ax2)
    ax2.set_title("Distribusi Gaji Bulanan: Bertahan vs Resign")
    st.pyplot(fig2)

    st.info(
        f"""**Insight:**
- Karyawan dengan gaji **lebih rendah** cenderung memiliki tingkat attrition yang lebih tinggi.
- Gaji dapat digunakan sebagai salah satu indikator retensi—HR dapat mempertimbangkan insentif atau penyesuaian gaji pada rentang rendah."""
    )

    # --- 3. Pie Chart Departemen vs Proporsi Attrition ---
    st.subheader("3️⃣ Proporsi Attrition Berdasarkan Departemen")
    dept_attr_counts = ed[ed['Attrition'] == 1]['Department'].value_counts()
    fig3, ax3 = plt.subplots()
    ax3.pie(dept_attr_counts, labels=dept_attr_counts.index, autopct='%1.1f%%', startangle=90, colors=sns.color_palette("pastel"))
    ax3.axis('equal')
    st.pyplot(fig3)

    st.info(
        f"""**Insight:**
- Departemen dengan jumlah karyawan keluar terbanyak: **{dept_attr_counts.idxmax()}**.
- Visual ini membantu HR untuk fokus pada divisi-divisi dengan turnover tinggi dan merancang strategi spesifik per departemen."""
    )

# Model Machine Learning untuk Prediksi
# Bikin Model
# Fitur dan target
features = [
    "Age", "DistanceFromHome", "MonthlyIncome", "YearsAtCompany",
    "YearsWithCurrManager", "TotalWorkingYears", "OverTime",
    "JobLevel", "JobRole", "Department"
]
target = "Attrition"

X = ed[features]
y = ed[target]

# Kolom kategorikal dan numerik
cat_cols = ["JobRole", "Department", "OverTime"] 
num_cols = ["Age", "DistanceFromHome", "MonthlyIncome", "YearsAtCompany",
            "YearsWithCurrManager", "TotalWorkingYears", "JobLevel"]

# Preprocessor
preprocessor = ColumnTransformer([
    ("num", StandardScaler(), num_cols),
    ("cat", OneHotEncoder(drop="first"), cat_cols),
], remainder='passthrough')

# Pipeline model
pipeline = Pipeline(steps=[
    ("preprocess", preprocessor),
    ("classifier", LogisticRegression(max_iter=1000, random_state=42))
])

# Split & latih
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)
pipeline.fit(X_train, y_train)

# Simpan model
joblib.dump(pipeline, "model_logreg.pkl")

# Halaman Prediksi
if page == "Prediksi":
    st.title("🤖 Prediksi Kemungkinan Karyawan Resign (Attrition)")

    st.markdown("**Masukkan data karyawan di sebelah kiri dan lihat kemungkinan attrition.**")

    # === Sidebar input ===
    st.sidebar.subheader("🧾 Input Data Karyawan")
    age = st.sidebar.slider("Umur", 18, 60, 30)
    distance = st.sidebar.slider("Jarak dari Rumah ke Kantor (km)", 1, 30, 5)
    monthly_income = st.sidebar.slider("Gaji Bulanan", 1000, 20000, 5000)
    years_at_company = st.sidebar.slider("Tahun di Perusahaan", 0, 40, 5)
    years_with_mgr = st.sidebar.slider("Tahun dengan Manager Saat Ini", 0, 20, 3)
    total_working_years = st.sidebar.slider("Total Tahun Bekerja", 0, 40, 10)
    overtime = st.sidebar.selectbox("Lembur", ["Yes", "No"])
    job_level = st.sidebar.selectbox("Level Jabatan", [1, 2, 3, 4, 5])
    job_role = st.sidebar.selectbox("Posisi", ed['JobRole'].unique())
    department = st.sidebar.selectbox("Departemen", ed['Department'].unique())

    # Buat dataframe input
    input_data = pd.DataFrame({
        "Age": [age],
        "DistanceFromHome": [distance],
        "MonthlyIncome": [monthly_income],
        "YearsAtCompany": [years_at_company],
        "YearsWithCurrManager": [years_with_mgr],
        "TotalWorkingYears": [total_working_years],
        "OverTime": [overtime],  # Biarkan string "Yes" atau "No"
        "JobLevel": [job_level],
        "JobRole": [job_role],
        "Department": [department]
    })

 # Placeholder hasil prediksi
if st.button("🔍 Prediksi Attrition"):
    try:
        # Load pipeline model (sudah include encoder + scaler)
        model = joblib.load("model_logreg.pkl")

        # Prediksi
        prob = model.predict_proba(input_data)[0][1]
        pred = model.predict(input_data)[0]

        # Interpretasi risiko
        risiko = ""
        if prob < 0.2:
            risiko = "rendah"
        elif prob < 0.5:
            risiko = "sedang"
        else:
            risiko = "tinggi"

        # Tampilkan hasil prediksi
        st.markdown("**Hasil Prediksi:**")
        st.metric("Kemungkinan Attrition", f"{round(prob*100, 2)} %")
        st.info(f"**Interpretasi:** Probabilitas menunjukkan risiko attrition **{risiko}**. HR bisa mengamati kembali faktor lembur dan tingkat jabatan.")
    
    except Exception as e:
        st.error(f"❌ Terjadi kesalahan saat memuat model atau prediksi: {e}")


elif page == "Rekomendasi":
    st.title("📌 Rekomendasi Strategis Retensi Karyawan")

    st.markdown("Analisis ini bertujuan membantu HR dalam merancang strategi retensi karyawan berdasarkan data dan prediksi model.")

    # ===================== #
    # 1. Korelasi Fitur     #
    # ===================== #
    st.subheader("🔎 Faktor-faktor yang Mempengaruhi Attrition")

    corr_matrix = ed.corr(numeric_only=True)
    attr_corr = corr_matrix['Attrition'].sort_values(ascending=False).drop('Attrition')

    fig1, ax1 = plt.subplots(figsize=(8,5))
    sns.barplot(x=attr_corr.values, y=attr_corr.index, palette='coolwarm', ax=ax1)
    ax1.set_title("Korelasi Fitur terhadap Attrition", fontsize=14)
    ax1.set_xlabel("Koefisien Korelasi")
    st.pyplot(fig1)

    st.markdown("""
    **Insight:** Fitur dengan korelasi tertinggi terhadap attrition meliputi:
    - Age (negatif)
    - TotalWorkingYears (negatif)
    - Monthly Income (negatif)
    - DistanceFromHome (positif)
    """)

    # ============================ #
    # 2. Segmentasi Risiko Tinggi #
    # ============================ #
    st.subheader("🔥 Segmentasi Karyawan Risiko Tinggi")

    if 'Attrition_Prediction' not in ed.columns:
        try:
            model = joblib.load("model_logreg.pkl")
            feature_cols = [col for col in ed.columns if col not in ['Attrition']]
            ed['Attrition_Prediction'] = model.predict(ed[feature_cols])
        except Exception as e:
            st.error(f"Gagal melakukan prediksi: {e}")

    ed_risk = ed[ed['Attrition_Prediction'] == 1]
    st.write(f"Jumlah karyawan berisiko tinggi: {ed_risk.shape[0]} orang")

    col1, col2 = st.columns(2)
    with col1:
        fig2 = px.histogram(ed_risk, x='JobRole', title='Distribusi Risiko berdasarkan Job Role', color_discrete_sequence=['#d62728'])
        st.plotly_chart(fig2, use_container_width=True)
    with col2:
        fig3 = px.pie(ed_risk, names='OverTime', title='Proporsi Lembur Karyawan Risiko Tinggi')
        st.plotly_chart(fig3, use_container_width=True)

    # 🔍 Tambahan Visualisasi Proporsi Lembur Seluruh Karyawan
    st.subheader("📊 Perbandingan Proporsi Lembur")
    col3, col4 = st.columns(2)
    with col3:
        fig4 = px.pie(ed, names='OverTime', title='Proporsi Lembur Semua Karyawan')
        st.plotly_chart(fig4, use_container_width=True)
    with col4:
        fig5 = px.histogram(ed, x='OverTime', color='Attrition_Prediction', barmode='group',
                            title='Distribusi Prediksi Attrition terhadap Status Lembur')
        st.plotly_chart(fig5, use_container_width=True)

    # ============================== #
    # 3. Filter Interaktif & Detail #
    # ============================== #
    st.subheader("📋 Filter Karyawan Risiko untuk Tindakan HR")

    dept_selected = st.selectbox("Pilih Department", options=ed_risk['Department'].unique())
    ed_filtered = ed_risk[ed_risk['Department'] == dept_selected]

    st.markdown(f"Daftar karyawan berisiko di departemen **{dept_selected}**:")

    # Gunakan index jika EmployeeNumber tidak ada
    ed_filtered = ed_filtered.reset_index(drop=False)
    if 'EmployeeNumber' not in ed_filtered.columns:
        ed_filtered.rename(columns={'index': 'EmployeeID'}, inplace=True)

    # Kolom yang ingin ditampilkan
    display_cols = ['EmployeeID' if 'EmployeeID' in ed_filtered.columns else 'EmployeeNumber', 
                    'JobRole', 'JobSatisfaction', 'OverTime', 'MonthlyIncome', 'PerformanceRating']

    # Cek kolom tersedia
    missing_cols = [col for col in display_cols if col not in ed_filtered.columns]
    if missing_cols:
        st.error(f"Kolom berikut tidak ditemukan di dataframe: {missing_cols}")
    else:
        st.dataframe(ed_filtered[display_cols])

    # =========================== #
    # 4. Rekomendasi Spesifik HR #
    # =========================== #
    st.subheader("📌 Rekomendasi Tindakan HR")

    st.markdown("""
    **Berdasarkan data dan prediksi model, HR disarankan untuk:**

    1. **Mengurangi Lembur:** Karyawan dengan status lembur konsisten memiliki risiko 2x lipat lebih tinggi.
    2. **Intervensi di Job Role Tertentu:** Role seperti *Sales Executive* dan *Laboratory Technician* menunjukkan tingkat attrition lebih tinggi.
    3. **Tinjauan Kompensasi:** Income rendah berbanding terbalik dengan tingkat retensi – perlu evaluasi struktur gaji.
    4. **Peningkatan Job Satisfaction:** Implementasi pelatihan, keterlibatan kerja, dan jalur promosi dapat membantu.
    5. **Pemantauan Personal:** Gunakan daftar filter di atas untuk melakukan evaluasi 1-on-1.

    ⚠️ *Perhatian khusus pada karyawan dengan kombinasi: low job satisfaction, overtime sering, dan pendapatan < median.*
    """)

    # ========================== #
    # 5. Optional Export Feature #
    # ========================== #
    if missing_cols == []:
        csv = ed_filtered[display_cols].to_csv(index=False).encode('utf-8')
        st.download_button("📥 Unduh Data Risiko Filtered", csv, f"Risiko_{dept_selected}.csv", "text/csv")



