import streamlit as st
import pandas as pd
import joblib

# 页面标题
st.title("水深分类预测工具")

# 加载模型与标准化器
@st.cache_resource
def load_model():
    model = joblib.load("water_depth_classifier.pkl")
    scaler = joblib.load("water_depth_scaler.pkl")
    return model, scaler

model, scaler = load_model()

# 文件上传或手动输入
upload_option = st.radio("选择数据输入方式：", ["上传Excel文件", "手动输入"])

if upload_option == "上传Excel文件":
    uploaded_file = st.file_uploader("上传包含 AC、DEN、GR 列的 Excel 文件", type=["xlsx"])
    if uploaded_file:
        df = pd.read_excel(uploaded_file)
        if all(col in df.columns for col in ['AC', 'DEN', 'GR']):
            # 预测
            X_scaled = scaler.transform(df[['AC', 'DEN', 'GR']])
            preds = model.predict(X_scaled)
            df["预测水深"] = preds
            st.success("预测完成！")
            st.dataframe(df)
            # 下载结果
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button("下载结果为 CSV", csv, "预测结果.csv", "text/csv")
        else:
            st.error("Excel 文件必须包含 AC、DEN、GR 三列。")
else:
    ac = st.number_input("AC", value=250.0)
    den = st.number_input("DEN", value=2.60)
    gr = st.number_input("GR", value=90.0)

    if st.button("预测水深"):
        X = pd.DataFrame([[ac, den, gr]], columns=['AC', 'DEN', 'GR'])
        X_scaled = scaler.transform(X)
        pred = model.predict(X_scaled)
        st.success(f"预测水深类别为：{int(pred[0])}")
