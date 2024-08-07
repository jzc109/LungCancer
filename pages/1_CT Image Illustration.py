import streamlit as st  
import requests  
from PIL import Image  
from io import BytesIO  

# GitHub图片的URL  
url = 'https://raw.githubusercontent.com/jzc109/LungCancer/main/pages/CT_image.png'  

# 获取图片  
response = requests.get(url)  

# 检查请求是否成功  
if response.status_code == 200:  
    # 使用BytesIO将响应内容转换为图像  
    imageCT = Image.open(BytesIO(response.content))  
else:  
    st.error("无法加载图片。请检查URL或网络连接。")  

# 设置页面配置  
st.set_page_config(page_title="CT Image Demo")  

# 显示标题和图片  
st.markdown("# CT Image Illustration")  
st.write(' ')  
if 'imageCT' in locals():  
    st.image(imageCT)  
st.write(' ')
