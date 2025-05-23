# 🌊 水深分类预测工具

这是一个基于测井数据的水深分类预测工具，使用机器学习模型对地质层位中的水深类别进行智能识别。

---

## 📦 项目内容

- 使用 **Random Forest 分类模型** 对水深类别进行预测；
- 支持 **上传 Excel 文件**（包含 AC、DEN、GR 三列）或 **手动输入参数**；
- 使用 Streamlit 构建交互式网页界面；
- 支持结果导出为 CSV。

---

## 🧪 示例输入格式

上传的 Excel 文件需包含以下三列（列名必须一致）：

| AC     | DEN   | GR    |
|--------|-------|-------|
| 250.5  | 2.58  | 90.2  |
| 245.0  | 2.65  | 95.5  |

---

## 🚀 使用方法

### 1. 安装依赖

```bash
pip install -r requirements.txt
