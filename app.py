import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# --- 新增分析用套件 ---
import seaborn as sns
import matplotlib.pyplot as plt
import platform
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import r2_score
import os
import matplotlib.font_manager as fm

# ==========================================
# 解決 Matplotlib 中文亂碼 (全平台通用 + 雲端修正版)
# ==========================================
def set_chinese_font():
    # 1. 第一優先：尋找專案目錄下的 "font.ttf" (這是給 Streamlit Cloud 用的)
    font_path = "font.ttf" 
    if os.path.exists(font_path):
        # 動態加入字型
        fm.fontManager.addfont(font_path)
        # 設定為預設字型
        plt.rcParams['font.family'] = fm.FontProperties(fname=font_path).get_name()
        return # 成功就結束
    
    # 2. 第二優先：如果是 Windows/Mac 本機開發，嘗試系統內建字型
    system = platform.system()
    if system == 'Windows':
        plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'SimHei']
    elif system == 'Darwin': # Mac
        plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'Heiti TC']
    else:
        # Linux (若沒上傳 font.ttf，這裡通常會失敗，變成方框)
        plt.rcParams['font.sans-serif'] = ['WenQuanYi Zen Hei', 'Droid Sans Fallback']

    plt.rcParams['axes.unicode_minus'] = False # 解決負號顯示問題

# 呼叫設定函式
set_chinese_font()



# 解決 Matplotlib 中文亂碼 (Windows/Mac/Linux 通用解法)
if platform.system() == 'Windows':
    plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei']
else:
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'WenQuanYi Zen Hei', 'Heiti TC']
plt.rcParams['axes.unicode_minus'] = False

# ==========================================
# 1. 頁面基礎設定與 CSS 美化
# ==========================================
st.set_page_config(
    page_title="Pagamo 學習行為分析系統",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 注入 CSS 樣式：讓介面看起來更像專業儀表板
st.markdown("""
    <style>
    /* 全局字體 */
    .main { font-family: 'Microsoft JhengHei', sans-serif; }
    
    /* KPI 卡片樣式 */
    .kpi-card {
        background-color: #ffffff;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        text-align: center;
        border-top: 5px solid #4A90E2;
        margin-bottom: 20px;
    }
    .kpi-title { font-size: 16px; color: #666; font-weight: bold; }
    .kpi-value { font-size: 32px; color: #333; font-weight: bold; margin: 10px 0; }
    .kpi-delta { font-size: 14px; color: #28a745; font-weight: bold; }
    .kpi-delta.neg { color: #dc3545; }
    
    /* 標題樣式 */
    h1, h2, h3 { color: #2C3E50; }
    </style>
    """, unsafe_allow_html=True)

# ==========================================
# 2. 資料載入與處理 (核心修復部分)
# ==========================================
@st.cache_data
def load_and_process_data():
    filename = 'https://github.com/jimmy2000426/streamlit_implement/raw/main/final_features_full.csv'
    df = None
    
    # A. 嘗試讀取檔案
    try:
        df = pd.read_csv(filename, encoding='utf-8-sig')
    except:
        try:
            df = pd.read_csv(filename, encoding='big5')
        except:
            st.warning("⚠️ 找不到 `final_features_full.csv`，將使用模擬數據演示。")
            
    # B. 如果讀不到檔，產生模擬資料 (Demo用)
    if df is None:
        data = {
            'Feat_拖延指數_hr': np.random.uniform(0, 72, 200),
            'Feat_訂正恆毅力_min': np.random.uniform(0, 30, 200),
            'Feat_思考密度': np.random.uniform(1, 20, 200),
            'Feat_L1_訊息提取力': np.random.uniform(0.4, 1.0, 200),
            'Feat_L3_批判省思力': np.random.uniform(0.2, 0.9, 200),
            'Feat_認知落差': np.random.uniform(-0.1, 0.5, 200),
            'Feat_完成效率': np.random.uniform(0, 10, 200),
            '難易度': np.random.choice(['難', '中', '易'], 200)
        }
        df = pd.DataFrame(data)

    # C. [關鍵修復]：自動進行學生分群 (Rule-based Clustering)
    # 這裡模擬 K-Means 的結果，確保 Student_Cluster 欄位一定存在
    # 邏輯：根據特徵值貼標籤
    conditions = [
        (df['Feat_思考密度'] < 3),  # 秒殺 -> 衝動
        (df['Feat_L3_批判省思力'] > 0.8) & (df['Feat_思考密度'] > 8), # 高分且深思 -> 精熟
        (df['Feat_訂正恆毅力_min'] > 10) & (df['Feat_L3_批判省思力'] < 0.6), # 努力但低分 -> 掙扎
        (df['Feat_拖延指數_hr'] > 48) # 拖延 -> 低動機
    ]
    choices = ['衝動失誤型', '精熟深思型', '掙扎努力型', '低動機型']
    
    # 產生分群欄位，若都不符合則歸類為「一般型」
    df['Student_Cluster'] = np.select(conditions, choices, default='一般型')
    
    return df

# 載入資料
df = load_and_process_data()
# ==========================================
# 機器學習模型訓練函數 (快取起來，避免每次重跑)
# ==========================================
@st.cache_resource
def train_interaction_model(data):
    # 定義特徵
    features = [
        'Feat_拖延指數_hr', 'Feat_訂正恆毅力_min', 'Feat_思考密度', 
        'Feat_是否完成任務', 'Feat_L1_訊息提取力', 
        'Feat_難度加權表現', 'Feat_挑戰係數'
    ]
    # 確保欄位存在 (防呆)
    valid_features = [c for c in features if c in data.columns]
    target = 'Feat_L3_批判省思力'
    
    X = data[valid_features]
    y = data[target]
    
    # 切分資料
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 訓練 XGBoost
    model = XGBRegressor(n_estimators=100, learning_rate=0.05, max_depth=6, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    
    return model, valid_features, y_test, y_pred, r2
# ==========================================
# 3. 側邊欄：全域設定
# ==========================================
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2997/2997314.png", width=80)
    st.title("Pagamo 分析系統")
    st.markdown("---")
    
    # 資料篩選器 (如果 CSV 有這些欄位的話)
    if '班級' in df.columns:
        selected_class = st.multiselect("篩選班級", df['班級'].unique(), default=df['班級'].unique())
        df = df[df['班級'].isin(selected_class)]
        
    st.markdown("### 📊 分析維度")
    x_axis = st.selectbox("X 軸變數", ['Feat_L1_訊息提取力', 'Feat_思考密度', 'Feat_拖延指數_hr'], index=0)
    y_axis = st.selectbox("Y 軸變數", ['Feat_L3_批判省思力', 'Feat_認知落差', 'Feat_完成效率'], index=0)
    
    st.info("本系統應用 **自我調節學習 (SRL)** 理論與 **機器學習** 演算法開發。")

# ==========================================
# 4. 主畫面內容 (修復版：使用變數對照，確保切換無誤)
# ==========================================

# 定義分頁名稱清單 (這樣修改文字時只要改這裡，下面會自動對應)
TABS = [
    "📈 班級戰情室", 
    "🤖 AI 學生診斷", 
    "🧠 特徵解密", 
    "📊 數據儀表板"  # 我把這裡的圖示改成 📊 避免跟第一個重複
]

# 建立選單
selected_tab = st.radio(
    "", 
    TABS, 
    horizontal=True,
    label_visibility="collapsed"
)

st.markdown("---")

# --- Debug 模式 (如果不出現，請暫時取消註解下面這行來檢查) ---
# st.write(f"目前選擇的分頁是: {selected_tab}") 

# ==========================================
# 分頁邏輯 (使用 TABS[index] 來判斷，保證萬無一失)
# ==========================================

# --- 分頁 1: 班級戰情室 ---
if selected_tab == TABS[0]:
    st.markdown("## 🎓 班級學習行為戰情室")
    
    # KPI 區塊
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    with kpi1:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">平均高階思考力 (L3)</div>
            <div class="kpi-value">{df['Feat_L3_批判省思力'].mean()*100:.1f}%</div>
            <div class="kpi-delta">🎯 目標 80%</div>
        </div>
        """, unsafe_allow_html=True)
    
    
    with kpi2:
        val = df['Feat_拖延指數_hr'].mean()
        color = "neg" if val > 24 else ""
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">平均拖延時間</div>
            <div class="kpi-value">{val:.1f} hr</div>
            <div class="kpi-delta {color}">⏳ 越低越好</div>
        </div>
        """, unsafe_allow_html=True)

    with kpi3:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">平均訂正恆毅力</div>
            <div class="kpi-value">{df['Feat_訂正恆毅力_min'].mean():.1f} min</div>
            <div class="kpi-delta">💪 持續保持</div>
        </div>
        """, unsafe_allow_html=True)

    with kpi4:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">風險學生數 (認知落差大)</div>
            <div class="kpi-value">{len(df[df['Feat_認知落差']>0.3])} 人</div>
            <div class="kpi-delta neg">⚠️ 需優先關注</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    
        # 2. 圖表區
    col_chart1, col_chart2 = st.columns([2, 1])
    
    with col_chart1:
        st.subheader(f"🔍 關聯分析：{x_axis} vs {y_axis}")
        # 散佈圖
        fig_scatter = px.scatter(
            df,
            x=x_axis,
            y=y_axis,
            color="Student_Cluster",
            size="Feat_思考密度",
            hover_data=["Feat_拖延指數_hr", "Feat_認知落差"],
            color_discrete_map={
                "精熟深思型": "#2ECC71", 
                "衝動失誤型": "#E74C3C", 
                "掙扎努力型": "#F1C40F", 
                "低動機型": "#95A5A6", 
                "一般型": "#3498DB"
            },
            template="plotly_white"
        )
        fig_scatter.update_layout(height=400)
        st.plotly_chart(fig_scatter, use_container_width=True)
        
    with col_chart2:
        st.subheader("👥 學生類型分佈")
        # 甜甜圈圖
        fig_pie = px.pie(
            df, 
            names="Student_Cluster", 
            hole=0.5,
            color="Student_Cluster",
            color_discrete_map={
                "精熟深思型": "#2ECC71", 
                "衝動失誤型": "#E74C3C", 
                "掙扎努力型": "#F1C40F", 
                "低動機型": "#95A5A6", 
                "一般型": "#3498DB"
            }
        )
        fig_pie.update_layout(height=400, showlegend=False)
        fig_pie.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig_pie, use_container_width=True)
    

# --- 分頁 2: AI 學生診斷 ---
elif selected_tab == TABS[1]:
    st.markdown("## 🤖 個別學生 AI 診斷系統")
    st.markdown("輸入學生在 Pagamo 的原始行為數據，系統將自動計算特徵並提供教學建議。")
    
    c1, c2 = st.columns([1, 2])
    
    with c1:
        with st.form("ai_form"):
            st.markdown("### 📝 輸入數據")
            duration = st.number_input("首次作答耗時 (秒)", value=300, step=10)
            delay = st.slider("任務拖延時間 (小時)", 0, 72, 2)
            grit = st.slider("訂正花費時間 (分鐘)", 0, 60, 15)
            st.markdown("---")
            l1_score = st.slider("L1 訊息提取正確率", 0.0, 1.0, 0.8)
            l3_score = st.slider("L3 批判省思正確率", 0.0, 1.0, 0.6)
            
            submit = st.form_submit_button("🚀 開始診斷")
            
    with c2:
        if submit:
            with st.spinner('AI 正在分析大腦認知模型，請稍候...'):
                import time
                time.sleep(0.5) 
                
                # 計算特徵
                density = duration / 5
                gap = l1_score - l3_score
                
                # 規則判斷
                prediction = "一般型"
                color = "#3498DB"
                advice = "保持觀察，持續給予正向鼓勵。"
                
                if density < 5:
                    prediction = "衝動失誤型"
                    color = "#E74C3C"
                    advice = "⛔ **教學處方**：學生缺乏審題耐心，建議強制閱讀計時。"
                elif l3_score > 0.8 and density > 10:
                    prediction = "精熟深思型"
                    color = "#2ECC71"
                    advice = "🌟 **教學處方**：能力優異，可提供加深加廣教材。"
                elif gap > 0.3:
                    prediction = "表面學習型 (Gap大)"
                    color = "#F39C12"
                    advice = "💡 **教學處方**：能看懂字面但無法深思，需加強提問引導。"
                elif delay > 48:
                    prediction = "低動機型"
                    color = "#95A5A6"
                    advice = "🔥 **教學處方**：學習動力低落，建議先建立關係與獎勵機制。"

                # 顯示結果
                st.markdown(f"""
                <div style="background-color: {color}20; border-left: 5px solid {color}; padding: 20px; border-radius: 5px;">
                    <h3 style="color: {color}; margin:0;">🎯 診斷結果：{prediction}</h3>
                    <p style="margin-top: 10px; font-size: 18px;"><b>特徵掃描：</b>思考密度 {density:.1f} | 認知落差 {gap:.2f}</p>
                </div>
                """, unsafe_allow_html=True)
                st.info(f"### 💊 建議：{advice}")

                # 雷達圖
                categories = ['學習動機', '恆毅力', '思考品質', '基礎能力', '高階能力']
                r_val = [max(0, 5 - (delay/20)), min(5, grit/10), min(5, density/10), l1_score*5, l3_score*5]
                fig_radar = px.line_polar(r=r_val, theta=categories, line_close=True, range_r=[0,5])
                fig_radar.update_traces(fill='toself', line_color=color)
                st.plotly_chart(fig_radar, use_container_width=True)

# --- 分頁 3: 特徵解密 ---
elif selected_tab == TABS[2]:
    st.markdown("## 🧠 特徵工程：破解學習黑盒子")
    c1, c2 = st.columns(2)
    with c1:
        st.info("### 1. 認知落差 (Cognitive Gap)")
        st.latex(r"Gap = \text{L1 基礎力} - \text{L3 高階力}")
    with c2:
        st.info("### 2. 思考密度 (Thinking Density)")
        st.latex(r"Density = \frac{\text{作答總時間}}{\text{總題數}}")
    st.dataframe(df.head(10))

# --- 分頁 4: 數據儀表板 (修改版：下拉選單選取，提升效能) ---
elif selected_tab == TABS[3]:
    st.markdown("## 📊 深度學習行為分析儀表板")
    
    # 1. 建立選單 (Selectbox)
    analysis_type = st.selectbox(
        "請選擇分析視角：",
        [
            "請選擇...",
            "1. [EDA] 特徵相關性熱力圖", 
            "2. [EDA] 思考密度 vs 批判省思力", 
            "3. [EDA] 多維度行為分析 (恆毅力/效率)", 
            "4. [AI] 特徵重要性排行 (XGBoost)", 
            "5. [AI] 模型預測誤差分析"
        ]
    )
    
    st.markdown("---")

    # 2. 根據選擇顯示對應圖表 (Lazy Loading)
    
    # --- 選項 1: 熱力圖 ---
    if analysis_type == "1. [EDA] 特徵相關性熱力圖":
        st.subheader("🔥 特徵相關性熱力圖")
        st.caption("觀察哪些行為特徵之間有高度正相關(紅)或負相關(藍)")
        
        cols_to_plot = [
            'Feat_拖延指數_hr', 'Feat_訂正恆毅力_min', 'Feat_思考密度', 
            'Feat_L1_訊息提取力', 'Feat_L3_批判省思力', 'Feat_完成效率'
        ]
        valid_cols = [c for c in cols_to_plot if c in df.columns]
        
        fig = plt.figure(figsize=(10, 8))
        corr_matrix = df[valid_cols].corr()
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
        st.pyplot(fig)

    # --- 選項 2: 迴歸散佈圖 ---
    elif analysis_type == "2. [EDA] 思考密度 vs 批判省思力":
        st.subheader("📉 思考密度 vs 批判省思力")
        st.caption("驗證假設：學生思考時間越長(密度高)，答題品質越好嗎？")
        
        fig = plt.figure(figsize=(10, 6))
        sns.regplot(x='Feat_思考密度', y='Feat_L3_批判省思力', data=df, 
                    scatter_kws={'alpha':0.5, 'color':'#2980b9'}, line_kws={'color':'#e74c3c'})
        plt.xlabel('思考密度 (秒/題)')
        plt.ylabel('L3 批判省思正確率')
        plt.grid(True, linestyle='--', alpha=0.3)
        st.pyplot(fig)

    # --- 選項 3: 氣泡圖 ---
    elif analysis_type == "3. [EDA] 多維度行為分析 (恆毅力/效率)":
        st.subheader("🫧 多維度行為氣泡圖")
        st.caption("顏色代表「L3高階能力」，點大小代表「思考密度」。")
        
        fig = plt.figure(figsize=(10, 6))
        sns.scatterplot(
            x='Feat_訂正恆毅力_min', 
            y='Feat_完成效率', 
            data=df, 
            hue='Feat_L3_批判省思力', 
            palette='viridis', 
            size='Feat_思考密度', 
            sizes=(50, 300)
        )
        plt.legend(bbox_to_anchor=(1.01, 1), loc=2, borderaxespad=0.)
        plt.grid(True, alpha=0.3)
        st.pyplot(fig)

    # --- 選項 4: AI 特徵權重 (需要訓練模型) ---
    elif analysis_type == "4. [AI] 特徵重要性排行 (XGBoost)":
        st.subheader("🤖 AI 關鍵因子分析")
        
        # 只有使用者點到這裡時，才開始跑模型運算
        with st.spinner("正在啟動 AI 引擎並訓練 XGBoost 模型，請稍候..."):
            model, features, y_test, y_pred, r2 = train_interaction_model(df)
        
        st.success(f"模型訓練完成！預測準確度 R² = {r2:.2f}")
        st.caption("AI 認為「哪個行為」最能預測學生的閱讀理解成就？")
        
        fig = plt.figure(figsize=(10, 6))
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        plt.bar(range(len(features)), importances[indices], align='center', color='#2ecc71')
        plt.xticks(range(len(features)), [features[i] for i in indices], rotation=45, ha='right')
        plt.ylabel('重要性權重')
        st.pyplot(fig)

    # --- 選項 5: 預測誤差圖 (需要訓練模型) ---
    elif analysis_type == "5. [AI] 模型預測誤差分析":
        st.subheader("🎯 模型預測誤差分析")
        
        # 同樣檢查模型是否已訓練
        with st.spinner("讀取 AI 模型數據中..."):
            model, features, y_test, y_pred, r2 = train_interaction_model(df)
            
        st.caption("X軸為實際分數，Y軸為AI預測分數。點越接近虛線代表預測越準確。")
        
        fig = plt.figure(figsize=(10, 6))
        plt.scatter(y_test, y_pred, alpha=0.6, color='#8e44ad')
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
        plt.xlabel('實際值 (Actual L3)')
        plt.ylabel('預測值 (Predicted L3)')
        plt.grid(True, alpha=0.3)
        st.pyplot(fig)
        
    # --- 預設畫面 ---
    else:
        st.info("👈 請從上方選單選擇您想查看的分析圖表。")
        # 可以放一張示意圖或 logo 讓畫面不要太空
        st.markdown("""
        <div style="text-align: center; color: #aaa; padding: 50px;">
            <h3>等待指令中...</h3>
            <p>選取後將即時運算並渲染圖表</p>
        </div>

        """, unsafe_allow_html=True)
