import matplotlib.pyplot as plt
import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap

# 加载随机森林模型
model = joblib.load('RF-7.pkl')

# 定义特征名称（根据你的数据调整）
feature_names = ["TP",	"LYMPH%",	"LYMPH#",	"GLB",	"MCHC",	"MONO%",	"HCT"]


# Streamlit 用户界
st.title("Diffuse large B-cell lymphoma (DLBCL) Screening Model")

# 用户输入特征数据
TP = st.number_input("TP:", min_value=0.0, max_value=100.0, value=61.2)
LYMPH_percent = st.number_input("LYMPH%:", min_value=0.0, max_value=100.0, value=13.9)
LYMPH_num = st.number_input("LYMPH#:", min_value=0.0, max_value=100.0, value=0.55)
GLB = st.number_input("GLB:", min_value=0.0, max_value=100.0, value=20.7)
MCHC = st.number_input("MCHC:", min_value=0.0, max_value=1000.0, value=290.0)
MONO_percent = st.number_input("MONO%:", min_value=0.0, max_value=100.0, value=7.8)
HCT = st.number_input("HCT:", min_value=0.0, max_value=100.0, value=31.0)


# 将输入的数据转化为模型的输入格式
feature_values = [
    TP, LYMPH_percent,LYMPH_num, GLB, MCHC, MONO_percent, HCT
]
features = np.array([feature_values])

# 当点击按钮时进行预测
if st.button("Predict"):
    # 进行预测
    predicted_class = model.predict(features)[0]
    predicted_proba = model.predict_proba(features)[0]

    # 显示预测结果
    st.write(f"**Predicted Class:** {predicted_class} (0: Health, 1: DLBCL)")
    st.write(f"**Prediction Probabilities:** {predicted_proba}")

    # 根据预测结果提供建议
    probability = predicted_proba[predicted_class] * 100

    if predicted_class == 1:
        advice = (
            f"According to our model, you have a high risk of DLBCL. "
            f"The model predicts that your probability of having Severe ACP is {probability:.1f}%. "
        )
    else:
        advice = (
            f"According to our model, you have not a high risk of DLBCL. "
            f"The model predicts that your probability of not having Mild ACP is {probability:.1f}%. "
        )

    st.write(advice)

    # 计算并显示SHAP值
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(pd.DataFrame([feature_values], columns=feature_names))

    # 根据预测结果生成并显示SHAP force plot
    if predicted_class == 1:
        shap.force_plot(explainer.expected_value[1], shap_values[:, :, 1],
                        pd.DataFrame([feature_values], columns=feature_names), matplotlib=True)
    else:
        shap.force_plot(explainer.expected_value[0], shap_values[:, :, 0],
                        pd.DataFrame([feature_values], columns=feature_names), matplotlib=True)

    # 保存SHAP图并显示
    plt.savefig("shap_force_plot.png", bbox_inches='tight', dpi=1200)
    st.image("shap_force_plot.png")
