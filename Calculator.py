from PIL import Image
import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import shap
import matplotlib.pyplot as plt
import pickle

plt.style.use('default')

st.set_page_config(
    page_title = 'Auxiliary Diagnostic System for Malignant Pulmonary Solid Nodules',
    page_icon = '🕵️‍♀️',
    layout = 'wide'
)

path='https://raw.githubusercontent.com/jzc109/LungCancer/main/dat20240527.csv'

@st.cache_data
def load_data(path):
    data = pd.read_csv(path)
    return data
data = load_data(path)
subset =data[['sex', 'age', 'RightUp', 'RightDown', 'maxlength', 'CEA', 'CYFRA21_1',
              'SCC', 'NSE', 'ProGRP', 'SmoothMargin', 'Halo', 'IrregularMargin',
              'Lobulation', 'SpinousProtuberant', 'SpiculatedMargin', 'PleuralRetraction',
              'VascularConvergence', 'BubbleLucencies', 'MultNodular', 'PolygonalShape']]
X = subset
y = data['mal']
# dashboard title
st.markdown("<h1 style='text-align: center; color: black;'>Auxiliary Diagnostic System for Malignant Pulmonary Solid Nodules</h1>", unsafe_allow_html=True)

# side-bar

def user_input_features():
    st.sidebar.header('User input parameters below ⬇️')
    age = st.sidebar.number_input(" Age(Years)", value=0, min_value=0, max_value=120, step=1)
    sex = st.sidebar.selectbox("Select gender:", ["Male", "Female"])
    RightUp = st.sidebar.selectbox("Location Right up", ('Yes', 'No'))
    RightDown = st.sidebar.selectbox("Location Right Down", ('Yes', 'No'))
    maxlength = st.sidebar.number_input("Maximum diameter of nodule", value=0.00, min_value=0.00, max_value=100.00, step=0.01)
    CEA = st.sidebar.number_input("CEA", value=0.00, min_value=0.00, max_value=5000000.00, step=0.01)
    CYFRA21_1 = st.sidebar.number_input("CYFRA21-1", value=0.00, min_value=0.00, max_value=5000000.00, step=0.01)
    SCC = st.sidebar.number_input("SCC", value=0.00, min_value=0.00, max_value=5000000.00, step=0.01)
    NSE = st.sidebar.number_input("NSE", value=0.00, min_value=0.00, max_value=5000000.00, step=0.01)
    ProGRP = st.sidebar.number_input("Pro-gastrin-releasing peptide, ProGRP", value=0.00, min_value=0.00, max_value=5000000.00, step=0.01)
    SmoothMargin = st.sidebar.selectbox("Smooth Margin", ('Yes', 'No'))
    Halo = st.sidebar.selectbox("Halo", ('Yes', 'No'))
    IrregularMargin = st.sidebar.selectbox("Irregular Margin", ('Yes', 'No'))
    Lobulation = st.sidebar.selectbox("Lobulation", ('Yes', 'No'))
    SpinousProtuberant = st.sidebar.selectbox("Spinous Protuberant", ('Yes', 'No'))
    SpiculatedMargin = st.sidebar.selectbox("Spiculated Margin", ('Yes', 'No'))
    PleuralRetraction = st.sidebar.selectbox("Pleural Retraction", ('Yes', 'No'))
    VascularConvergence = st.sidebar.selectbox("Vascular Convergence", ('Yes', 'No'))
    BubbleLucencies = st.sidebar.selectbox("Bubble Lucencies", ('Yes', 'No'))
    MultNodular = st.sidebar.selectbox("MultNodular", ('Yes', 'No'))
    PolygonalShape = st.sidebar.selectbox("Polygonal Shape", ('Yes', 'No'))

    if sex == 'Male':
            sex = 1
    else:
        sex = 0

    if RightUp == 'Yes':
            RightUp = 1
    else:
        RightUp = 0

    if RightDown == 'Yes':
        RightDown = 1
    else:
        RightDown = 0

    if SmoothMargin == 'Yes':
            SmoothMargin = 1
    else:
        SmoothMargin = 0

    if Halo == 'Yes':
            Halo = 1
    else:
        Halo = 0

    if IrregularMargin == 'Yes':
            IrregularMargin = 1
    else:
        IrregularMargin = 0

    if Lobulation == 'Yes':
            Lobulation = 1
    else:
        Lobulation = 0

    if SpinousProtuberant == 'Yes':
            SpinousProtuberant = 1
    else:
        SpinousProtuberant = 0

    if SpiculatedMargin == 'Yes':
            SpiculatedMargin = 1
    else:
        SpiculatedMargin = 0

    if PleuralRetraction == 'Yes':
            PleuralRetraction = 1
    else:
        PleuralRetraction = 0

    if VascularConvergence == 'Yes':
            VascularConvergence = 1
    else:
        VascularConvergence = 0

    if BubbleLucencies == 'Yes':
            BubbleLucencies = 1
    else:
        BubbleLucencies = 0

    if MultNodular == 'Yes':
            MultNodular = 1
    else:
        MultNodular = 0

    if PolygonalShape == 'Yes':
            PolygonalShape = 1
    else:
        PolygonalShape = 0

    output = [sex, age, RightUp, RightDown, maxlength, CEA, CYFRA21_1,
              SCC, NSE, ProGRP, SmoothMargin, Halo, IrregularMargin,
              Lobulation, SpinousProtuberant, SpiculatedMargin, PleuralRetraction,
              VascularConvergence, BubbleLucencies, MultNodular, PolygonalShape]
    return output

outputdf = user_input_features()

st.subheader('About the model')
st.write('The group of patients with lung nodules undergoing evaluation by surgeons is classified as a high-risk population for lung cancer, with a lung cancer incidence rate of 72.4%, and is specifically known as the thoracic surgery clinic cohort.'
         'Solid nodules that are newly detected in follow-up CT scans have a higher likelihood of being lung cancer compared to baseline nodules.'
         'Using laboratory measurements and imaging features, we developed a clinical prediction model to aid the diagnosis. This model used the Surper Learner to discriminate whether a nodule is benign or malignant.'
         'This model has a relatively high performace, AUC of ROC is 0.84 with 95% CI (0.74, 0.94). ' )

st.subheader('Guidelines of the Calculator ')
st.write('The calculator consists of 3 main sections.The left sidebar of the first section allows users to input relevant parameters and select model variables. The second displays the predicted probability of malignancy of the nodule. The third provides detailed model information, including global and local interpretations using SHAP providing insight into prediction generation. We hope this guide helps you effectively utilize our prediction calculator.')

image4 = Image.open('shap.png')
shapdatadf =data[['sex', 'age', 'RightUp', 'RightDown', 'maxlength', 'CEA', 'CYFRA21_1',
              'SCC', 'NSE', 'ProGRP', 'SmoothMargin', 'Halo', 'IrregularMargin',
              'Lobulation', 'SpinousProtuberant', 'SpiculatedMargin', 'PleuralRetraction',
              'VascularConvergence', 'BubbleLucencies', 'MultNodular', 'PolygonalShape']]
# shapvaluedf =pd.read_excel(r'https://github.com/jzc109/LungCancer/blob/main/shap_values.xlsx')
shapvaluedf = pd.read_excel('https://raw.githubusercontent.com/jzc109/LungCancer/main/shap_values.xlsx')
# 这里是查看SHAP值

st.subheader('Make predictions in real time')
outputdf = pd.DataFrame([outputdf], columns= shapdatadf.columns)

#st.write('User input parameters below ⬇️')
#st.write(outputdf)
# 加载模型
# with open("rf.pkl", "rb") as file:
#     rf = pickle.load(file)

# stack = joblib.load("stack.pkl")
with open("stack.pkl", "rb") as file:
    stack = pickle.load(file)

# print('stack是个啥：', rf)
p1 = stack.predict(outputdf)[0]
p2 = stack.predict_proba(outputdf)
p2 = round(p2[0][1], 4)

with open("rf.pkl", "rb") as file:
    rf = pickle.load(file)

p11 = rf.predict(outputdf)[0]
p21 = rf.predict_proba(outputdf)
p22 = round(p21[0][1], 4)


#st.write('User input parameters below ⬇️')
#st.write(outputdf)
st.write(f'Probability of malignancy: {p2}')
st.write(' ')

st.subheader("SHAP")
placeholder5 = st.empty()
with placeholder5.container():
    f1,f2 = st.columns(2)
    with f1:
        st.write('Beeswarm plot')
        st.write(' ')
        st.image(image4)
        st.write('The beeswarm plot is designed to display an information-dense summary of how the top features in a dataset impact the model’s output. Each instance the given explanation is represented by a single dot on each feature fow. The x position of the dot is determined by the SHAP value of that feature, and dots “pile up” along each feature row to show density.')
    with f2:
        st.write('Dependence plot for features')
        cf = st.selectbox("Choose a feature", (shapdatadf.columns))
        fig = px.scatter(x = shapdatadf[cf],
                         y = shapvaluedf[cf],
                         color=shapdatadf[cf],
                         color_continuous_scale= ['blue','red'],
                         labels={'x':'Original value', 'y':'shap value'})
        st.write(fig)
        st.write('The dependence plot for features using SHAP values visualizes the relationship between a specific feature and the model output. Each instance is represented by a point on the plot, where the x-axis corresponds to the feature values and the y-axis depicts the SHAP values for that feature. By observing how the SHAP values change with varying feature values, one can gain insights into the impact and directionality of the relationship between the feature and the model predictions. This type of plot provides a clear representation of how individual features influence the model output across the dataset.')


# 这里是查看SHAP和LIME图像的

# matplotlib.use('Agg')
st.set_option('deprecation.showPyplotGlobalUse', False)
placeholder6 = st.empty()

# 使用 shap.sample 函数来将数据汇总为 K 个样本
background_summary = shap.sample(X, 100)

# 可以继续使用 shapvaluedf 进行后续的 SHAP 值分析和可视化
# 示例：
# 进行 SHAP force plot 的可视化
st.write('Force plots')
plt.switch_backend('agg')
explainer = shap.KernelExplainer(stack.predict_proba, background_summary)
shap_values = explainer.shap_values(outputdf)
#explainer   = shap.TreeExplainer(rf)
#shap_values = explainer.shap_values(outputdf)

shap_values_reshaped = np.reshape(shap_values[0, :], (21, 2))

shap.force_plot(explainer.expected_value[1], shap_values[0, : ,1], outputdf.iloc[0,:], show=False, matplotlib=True)

st.pyplot(bbox_inches='tight')

st.write('The SHAP force plot can be used to visualise the SHAP value for each feature as a force that can increase (positive) or decrease (negative) the prediction relative to its baseline for the interpretation of individual patient outcome predictions.')


