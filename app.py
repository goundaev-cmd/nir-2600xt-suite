import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from sklearn.decomposition import PCA
from scipy.stats import chi2
import pickle
import io
import pandas as pd

# Настройка страницы
st.set_page_config(page_title="NIR 2600XT-1 Suite", layout="wide")

st.title("🌾 Аналитическая платформа БИК-спектрометра 2600XT-1")
st.caption("Модуль расширенной предобработки, фильтрации аномалий и экспресс-анализа")

# ==========================================
# 1. НИЗКОУРОВНЕВЫЙ ПАРСЕР ПОД ВАШ ФОРМАТ
# ==========================================
def parse_svf_binary(file_bytes):
    HEADER_SIZE = 5120
    BLOCK_SIZE = 6912
    N_POINTS = 1500
    
    total_len = len(file_bytes)
    if total_len < HEADER_SIZE + BLOCK_SIZE:
        return None, None
        
    data_bytes = file_bytes[HEADER_SIZE:]
    n_samples = len(data_bytes) // BLOCK_SIZE
    
    spectra = []
    chemistry = []
    
    for i in range(n_samples):
        block = data_bytes[i*BLOCK_SIZE : (i+1)*BLOCK_SIZE]
        
        chem_buf = np.frombuffer(block[466:470], dtype=np.float32)
        if len(chem_buf) > 0:
            chem_val = float(chem_buf[0])
        else:
            chem_val = 0.0
            
        if np.isnan(chem_val) or chem_val == 0.0:
            chem_val = 12.5
            
        spec_start = BLOCK_SIZE - (N_POINTS * 4)
        spectrum = np.frombuffer(block[spec_start:BLOCK_SIZE], dtype=np.float32).copy()
        
        if len(spectrum) == N_POINTS:
            if np.any(np.isnan(spectrum)) or np.all(spectrum == 0):
                continue
            spectra.append(spectrum)
            chemistry.append(chem_val)
            
    return np.array(spectra, dtype=np.float32), np.array(chemistry, dtype=np.float32)

# ==========================================
# 2. МАТЕМАТИКА ПРЕДОБРАБОТКИ И АНОМАЛИЙ
# ==========================================
def preprocess_spectra(spectra, wl, poly, deriv, use_snv):
    res = savgol_filter(spectra, window_length=wl, polyorder=poly, deriv=deriv, axis=1)
    if use_snv:
        mean = np.mean(res, axis=1, keepdims=True)
        std = np.std(res, axis=1, keepdims=True)
        res = (res - mean) / (std + 1e-8)
    return res

def calculate_mahalanobis(spectra, n_comp, alpha):
    pca = PCA(n_components=min(n_comp, len(spectra)-2))
    scores = pca.fit_transform(spectra)
    
    mu = np.mean(scores, axis=0)
    cov = np.cov(scores.T)
    inv_cov = np.linalg.inv(cov + np.eye(cov.shape[0]) * 1e-6)
    
    distances = []
    for score in scores:
        diff = score - mu
        md = np.dot(np.dot(diff, inv_cov), diff.T)
        distances.append(np.sqrt(md))
        
    distances = np.array(distances)
    thresh = np.sqrt(chi2.ppf(1 - alpha, df=pca.n_components_))
    
    clean_idx = np.where(distances <= thresh)[0]
    outlier_idx = np.where(distances > thresh)[0]
    
    return clean_idx, outlier_idx, distances, thresh

# ==========================================
# 3. ИНТЕРФЕЙС СТРАНИЦЫ (STREAMLIT)
# ==========================================
st.sidebar.header("🎛️ Панель управления")
uploaded_file = st.sidebar.file_uploader("Загрузить спектры (.svf, .bin)", type=["svf", "bin"])

if uploaded_file is None:
    st.sidebar.warning("Отображаются демонстрационные 32 образца ячменя.")
    np.random.seed(42)
    x = np.linspace(1100, 2500, 1500)
    base_spectrum = 0.5 * np.exp(-((x-1450)/80)**2) + 0.8 * np.exp(-((x-1940)/100)**2) + 0.3 * np.exp(-((x-2200)/120)**2)
    raw_spectra = np.array([base_spectrum + np.random.normal(0, 0.015, 1500) + np.random.uniform(-0.1, 0.1) for _ in range(32)])
    raw_spectra += 0.2 
    raw_spectra -= 0.15
    raw_chemistry = np.random.uniform(10.5, 14.5, 32)
else:
    file_bytes = uploaded_file.read()
    raw_spectra, raw_chemistry = parse_svf_binary(file_bytes)
    if raw_spectra is None or len(raw_spectra) == 0:
        st.error("Не удалось прочитать спектры из этого файла. Проверьте бинарную структуру.")
        st.stop()
if raw_spectra is not None and len(raw_spectra) > 0:
    st.sidebar.subheader("1. Настройки Савицкого-Голея")
    wl = st.sidebar.slider("Ширина окна (Window)", 5, 65, 15, step=2)
    poly = st.sidebar.slider("Степень полинома (Poly)", 1, 5, 2)
    deriv = st.sidebar.slider("Производная (Derivative)", 0, 2, 1)
    
    if poly >= wl:
        st.sidebar.error("Ошибка: Окно должно быть строго больше степени полинома!")
        wl = poly + 1
        
    use_snv = st.sidebar.checkbox("Применить SNV коррекцию", value=True)
    
    st.sidebar.subheader("2. Фильтрация аномалий")
    n_comp = st.sidebar.slider("Компонент PCA для Махаланобиса", 2, 10, 3)
    alpha = st.sidebar.slider("Уровень значимости (Alpha)", 0.001, 0.100, 0.050, step=0.005)

    # Расчеты математики
    proc_spectra = preprocess_spectra(raw_spectra, wl, poly, deriv, use_snv)
    clean_idx, outlier_idx, md_distances, thresh = calculate_mahalanobis(proc_spectra, n_comp, alpha)
    
    # Главные вкладки режимов приложения
    main_tab1, main_tab2 = st.tabs(["🏗️ Разработка и оптимизация калибровки", "🚀 Экспресс-анализ по готовой модели"])
        
    # --- РЕЖИМ РАЗРАБОТКИ КАЛИБРОВКИ ---
    with main_tab1:
        col1, col2, col3 = st.columns(3)
        col1.metric("Всего образцов", len(raw_spectra))
        col2.metric("Нормальные спектры", len(clean_idx))
        col3.metric("Обнаружено аномалий", len(outlier_idx))
            
        sub_tab1, sub_tab2 = st.tabs(["📊 Спектральный анализ", "🎯 Поиск аномалий (Mahalanobis)"])
            
        with sub_tab1:
            c1, c2 = st.columns(2)
            with c1:
                st.subheader("Исходные спектры")
                fig1, ax1 = plt.subplots(figsize=(6, 4))
                for i, spec in enumerate(raw_spectra):
                    color = 'red' if i in outlier_idx else 'gray'
                    alpha_val = 0.8 if i in outlier_idx else 0.4
                    ax1.plot(spec, color=color, alpha=alpha_val)
                st.pyplot(fig1)
                    
            with c2:
                st.subheader("После предобработки (SG + SNV)")
                fig2, ax2 = plt.subplots(figsize=(6, 4))
                for i, spec in enumerate(proc_spectra):
                    color = 'red' if i in outlier_idx else '#1f77b4'
                    alpha_val = 0.9 if i in outlier_idx else 0.4
                    ax2.plot(spec, color=color, alpha=alpha_val)
                st.pyplot(fig2)
                    
        with sub_tab2:
            st.subheader("Диаграмма расстояний Махаланобиса")
            fig3, ax3 = plt.subplots(figsize=(10, 4))
            colors = ['red' if i in outlier_idx else 'green' for i in range(len(md_distances))]
                
            ax3.bar(range(len(md_distances)), md_distances, color=colors, edgecolor='black', alpha=0.7)
            ax3.axhline(thresh, color='darkred', linestyle='--', linewidth=2, label=f'Порог: {thresh:.2f}')
            ax3.set_xlabel("Индекс образца")
            ax3.set_ylabel("Расстояние Махаланобиса")
            ax3.legend()
            st.pyplot(fig3)
                
            if len(outlier_idx) > 0:
                st.error(f"⚠️ Аномалии на индексах: {list(outlier_idx)}")
            else:
                st.success("✅ Спектральных аномалий не обнаружено.")

        # РЕГРЕССИЯ PLS
        st.write("---")
        st.header("🎯 Калибровка и PLS-регрессия")
        from sklearn.cross_decomposition import PLSRegression
        from sklearn.metrics import r2_score, mean_squared_error
            
        n_factors = st.slider("Количество латентных факторов PLS", 1, 10, 4)
            
        if st.button("🚀 Запустить PLS-калибровку"):
            X_clean = proc_spectra[clean_idx]
            y_clean = raw_chemistry[clean_idx]
            
            if len(X_clean) < n_factors + 1:
                st.error("Недостаточно образцов!")
            else:
                pls = PLSRegression(n_components=n_factors)
                pls.fit(X_clean, y_clean)
                y_pred = pls.predict(X_clean).flatten()
                
                r2 = r2_score(y_clean, y_pred)
                rmse = np.sqrt(mean_squared_error(y_clean, y_pred))
                
                cm1, cm2 = st.columns(2)
                cm1.metric("Точность R²", f"{r2:.3f}")
                cm2.metric("Ошибка RMSEC", f"{rmse:.3f}")
                
                                # --- СТАРЫЙ КОД MATPLOTLIB (ЗАМЕНИТЬ) ---
                # fig4, ax4 = plt.subplots(figsize=(6, 5))
                # ax4.scatter(y_clean, y_pred, ...)
                # ...
                # st.pyplot(fig4)
                
                # --- НОВЫЙ ИНТЕРАКТИВНЫЙ КОД PLOTLY ---
                import plotly.graph_objects as go
                
                # Формируем текст для всплывающей подсказки при наведении
                hover_text = [
                    f"Индекс: {idx}<br>Лаб. значение: {meas:.3f}<br>Предсказано: {pred:.3f}"
                    for idx, meas, pred in zip(clean_idx, y_clean, y_pred)
                ]
                
                fig4 = go.Figure()
                
                # Добавляем точки образцов
                fig4.add_trace(go.Scatter(
                    x=y_clean,
                    y=y_pred,
                    mode='markers',
                    marker=dict(size=10, color='darkgreen', line=dict(width=1, color='black')),
                    text=hover_text,
                    hoverinfo='text',
                    name='Образцы зерна'
                ))
                
                # Добавляем идеальную линию 1:1
                lims = [min(min(y_clean), min(y_pred)), max(max(y_clean), max(y_pred))]
                fig4.add_trace(go.Scatter(
                    x=lims, y=lims,
                    mode='lines',
                    line=dict(color='gray', dash='dash'),
                    name='Идеальное совпадение'
                ))
                
                # Настройка осей и внешнего вида
                fig4.update_layout(
                    title="График калибровки (Интерактивный Plotly)",
                    xaxis_title="Лабораторное значение (Измерено)",
                    yaxis_title="Предсказано моделью PLS",
                    width=700, height=500,
                    hovermode='closest'
                )
                
                # Выводим интерактивный график в Streamlit
                st.plotly_chart(fig4, use_container_width=True)


                # ЭКСПОРТ .PKL
                st.write("---")
                st.subheader("💾 Экспорт откалиброванной модели")
                model_data = {
                    "pls_model": pls, "window_length": wl,
                    "polyorder": poly, "derivative": deriv, "use_snv": use_snv
                }
                buffer = io.BytesIO()
                pickle.dump(model_data, buffer)
                buffer.seek(0)
                st.download_button(
                    label="📥 Скачать обученную модель (.pkl)",
                    data=buffer, file_name="nir_model.pkl",
                    mime="application/octet-stream"
                )

    # --- РЕЖИМ ЭКСПРЕСС-АНАЛИЗА ---
    with main_tab2:
        st.header("📋 Быстрое определение качественных показателей")
        model_file = st.file_uploader("Шаг 1: Загрузить файл модели (.pkl)", type=["pkl"])
        
        if model_file is not None:
            loaded_data = pickle.load(model_file)
            saved_pls = loaded_data["pls_model"]
            s_wl = loaded_data["window_length"]
            s_poly = loaded_data["polyorder"]
            s_deriv = loaded_data["derivative"]
            s_snv = loaded_data["use_snv"]
            
            st.success(f"✅ Модель загружена! Настройки: Окно={s_wl}, Полином={s_poly}, Производная={s_deriv}, SNV={s_snv}")
            st.write("Шаг 2: Расчет показателей для загруженных спектров...")
            
            optimized_spectra = preprocess_spectra(raw_spectra, s_wl, s_poly, s_deriv, s_snv)
            predictions = saved_pls.predict(optimized_spectra).flatten()
            
            results_df = pd.DataFrame({
                "Номер образца": range(1, len(predictions) + 1),
                "Предсказанное значение": np.round(predictions, 2)
            })
            st.subheader("📊 Результаты экспресс-анализа")
            st.dataframe(results_df, use_container_width=True)
            
            csv = results_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Экспортировать отчет в CSV",
                data=csv, file_name="nir_report.csv", mime="text/csv"
            )
