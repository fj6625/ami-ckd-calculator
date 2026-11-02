import os

# 关闭 Streamlit 使用情况收集提示
os.environ["STREAMLIT_BROWSER_GATHER_USAGE_STATS"] = "false"

import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import shap
import matplotlib.pyplot as plt
import streamlit.components.v1 as components

# 隐藏 Streamlit 默认的右上角菜单和页脚
st.set_page_config(page_title="AMI with CKD 28-Day Mortality Risk Calculator", layout="wide", menu_items={})
hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

BASE_DIR = Path(__file__).resolve().parent


@st.cache_resource
def load_model():
    model = joblib.load(BASE_DIR / 'results筛选4.1/model_RF_9features.pkl')
    bundle = joblib.load(BASE_DIR / 'results筛选4.1/preprocess_bundle_9.pkl')
    threshold = joblib.load(BASE_DIR / 'results筛选4.1/best_threshold_9.pkl')
    return model, bundle, float(threshold)


@st.cache_resource
def load_explainer(_model):
    return shap.TreeExplainer(_model)


model, bundle, threshold = load_model()
explainer = load_explainer(model)

features = list(bundle['final_features_9'])

# ------------------------------------------------------------
# 页面
# ------------------------------------------------------------
st.title("28-Day Mortality Risk Calculator for Patients with AMI and CKD")
st.markdown("Enter the indicators within 24 hours of ICU admission to calculate the predicted 28-day all-cause mortality probability and generate a SHAP-based individualized explanation.")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age (years)", min_value=18, max_value=110, value=82, step=1)
    anion_gap = st.number_input("Anion gap (mmol/L)", min_value=1.0, max_value=50.0, value=18.0, step=0.1)
    bun = st.number_input("BUN (mg/dL)", min_value=2.0, max_value=200.0, value=47.0, step=1.0)
    glucose = st.number_input("Glucose (mg/dL)", min_value=20.0, max_value=800.0, value=251.0, step=1.0)
    lactate = st.number_input("Lactate (mmol/L)", min_value=0.1, max_value=30.0, value=1.4, step=0.1)

with col2:
    ph = st.number_input("pH", min_value=6.5, max_value=8.0, value=7.36, step=0.01)
    phosphate = st.number_input("Phosphate (mg/dL)", min_value=0.5, max_value=20.0, value=3.8, step=0.1)
    resp_rate = st.number_input("Respiratory rate (/min)", min_value=5, max_value=60, value=15, step=1)
    sofa = st.number_input("First-day SOFA score", min_value=0, max_value=24, value=10, step=1)

if st.button("Calculate 28-day mortality risk and generate explanation"):
    # 1. 构建原始输入
    raw_dict = {
        'age': age, 'anion_gap': anion_gap, 'bun': bun,
        'glucose': glucose, 'lactate': lactate, 'ph': ph,
        'phosphate': phosphate, 'respiratory_rate': resp_rate,
        'sofa_first_day': sofa
    }
    raw_input = pd.DataFrame([raw_dict])[features]

    # 2. 标准化
    means = bundle['feature_means'][features]
    stds = bundle['feature_stds'][features]
    X_std = (raw_input - means) / stds

    # 3. 预测
    prob = float(model.predict_proba(X_std[features])[0, 1])

    st.subheader("Prediction Result")
    if prob >= threshold:
        st.error(f"Predicted 28-day mortality probability: **{prob:.1%}** (High risk, threshold {threshold:.3f})\n\nIntensive monitoring and intervention are recommended.")
    else:
        st.success(f"Predicted 28-day mortality probability: **{prob:.1%}** (Low risk, threshold {threshold:.3f})\n\nRoutine management pathway is recommended.")

    st.markdown("---")
    st.subheader("Individualized Risk Explanation (SHAP)")
    st.markdown("The following charts show how each indicator pushes the predicted risk for this patient up or down.")

    # 4. 计算 SHAP 值
    shap_output = explainer(X_std)

    if len(shap_output.shape) == 3:
        shap_val = shap_output[0, :, 1].values
        base_val = shap_output[0, 0, 1].base_values
    elif isinstance(shap_output, list):
        shap_val = shap_output[1][0]
        base_val = explainer.expected_value[1]
    else:
        shap_val = shap_output.values[0]
        base_val = shap_output.base_values[0]

    explanation = shap.Explanation(
        values=shap_val,
        base_values=base_val,
        data=raw_input.iloc[0].values,
        feature_names=features
    )

    # ------------------------------------------------------------
    # 图1：SHAP 力图 (Force Plot) - 放在上方全宽展示，解决重叠
    # ------------------------------------------------------------
    st.markdown("**SHAP Force Plot**")
    st.markdown("*The alternating red and blue arrows represent the pushing forces of features, which together aggregate into the model output.*")

    # 使用宽幅展示（宽度加大，解决文字挤压）
    force_plot = shap.force_plot(
        base_value=base_val,
        shap_values=shap_val,
        features=raw_input.iloc[0].values,
        feature_names=features,
        matplotlib=False,
        show=False,
        figsize=(20, 3),  # 加大 figure 宽度，给字符留足空间
        text_rotation=15  # 稍微倾斜文字，减少重叠
    )

    html_str = shap.getjs() + force_plot.html()
    # 分配 220px 的高度展示力图，滚动条设为 auto
    components.html(html_str, height=220, scrolling=True)

    st.markdown("<br>", unsafe_allow_html=True)  # 增加一点间距

    # ------------------------------------------------------------
    # 图2：SHAP 瀑布图 (Waterfall Plot) - 放在下方居中
    # ------------------------------------------------------------
    st.markdown("**SHAP Waterfall Plot**")
    st.markdown("*Starting from the baseline risk, the contribution magnitude of each feature is shown layer by layer. Red pushes the risk up, blue lowers it.*")

    # 绘制瀑布图后调整 figure 尺寸，使其缩小显示
    plt.figure(figsize=(10, 6))
    shap.plots.waterfall(explanation, max_display=10, show=False)
    fig = plt.gcf()
    # 缩小瀑布图尺寸
    fig.set_size_inches(7, 4.2)
    st.pyplot(fig, use_container_width=False)
    plt.close(fig)

    st.caption("This tool is for research assistance only and does not replace clinical decision-making.")
