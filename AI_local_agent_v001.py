import streamlit as st
import tempfile, os
import pandas as pd
import plotly.express as px
from openai import OpenAI
from streamlit_chat import message

# --- LANGCHAIN IMPORTS ---
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import CSVLoader
from langchain_community.embeddings import HuggingFaceEmbeddings

# --- ML IMPORTS ---
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from xgboost import XGBRegressor, XGBClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    mean_squared_error, r2_score, mean_absolute_error,
    accuracy_score, precision_score, recall_score, f1_score
)
import numpy as np

# --- SURVIVAL ANALYSIS IMPORTS ---
try:
    from sksurv.ensemble import RandomSurvivalForest
    from sksurv.metrics import concordance_index_censored
    HAS_SURVIVAL = True
except ImportError:
    HAS_SURVIVAL = False

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Interactive Data Science Agent", layout="wide")
st.title("⚡ Interactive Data Science Agent")

# ---------------- UTILS & CACHING ----------------
@st.cache_resource
def load_embedding_model():
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

def ask_gpt(prompt):
    # Ensure OPENAI_API_KEY is set in environment variables
    client = OpenAI()
    res = client.chat.completions.create(
        model="gpt-4o",
        temperature=0.2,
        messages=[
            {"role": "system", "content": "You are a data assistant. Answer based on the context provided."},
            {"role": "user", "content": prompt}
        ]
    )
    return res.choices[0].message.content

# ---------------- SESSION STATE INIT ----------------
st.session_state.setdefault("history", [])
st.session_state.setdefault("index", None)
st.session_state.setdefault("dfs", {})
st.session_state.setdefault("processed_files", [])

# ---------------- SIDEBAR (SMART CACHING) ----------------
with st.sidebar:
    st.header("Data Import")
    files = st.file_uploader("Upload CSVs", type=["csv"], accept_multiple_files=True)
    
    if files:
        current_file_names = [f.name for f in files]
        
        if st.session_state["processed_files"] != current_file_names:
            with st.spinner("Processing & Indexing files... (One-time setup)"):
                docs, dfs = [], {}
                embed_model = load_embedding_model()
                
                for f in files:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tf:
                        tf.write(f.getvalue())
                        tf_path = tf.name
                    
                    loader = CSVLoader(tf_path)
                    docs.extend(loader.load())
                    os.remove(tf_path)
                    
                    f.seek(0)
                    dfs[f.name] = pd.read_csv(f, low_memory=False)
                
                splits = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100).split_documents(docs)
                st.session_state.index = FAISS.from_documents(splits, embed_model)
                st.session_state.dfs = dfs
                st.session_state["processed_files"] = current_file_names
                
            st.success(f"Indexed {len(files)} files.")
        else:
            st.info(f"✅ Active Data: {len(st.session_state.dfs)} files ready.")

# ---------------- MAIN UI ----------------
t1, t2, t3 = st.tabs(["🤖 AI Chat", "📈 Plot", "🧠 ML"])

# ================= AI CHAT TAB =================
with t1:
    q = st.chat_input("Ask about your data...")
    if q and st.session_state.index:
        docs = st.session_state.index.similarity_search(q, k=3)
        ctx = "\n".join([d.page_content for d in docs])
        ans = ask_gpt(f"Context:\n{ctx}\n\nQuestion: {q}")
        st.session_state.history.append((q, ans))

    for q_txt, a_txt in st.session_state.history:
        message(q_txt, is_user=True)
        message(a_txt, is_user=False)

# ================= PLOT TAB =================
with t2:
    if not st.session_state.dfs:
        st.info("Upload data first.")
        st.stop()

    fname = st.selectbox("File to Plot", list(st.session_state.dfs.keys()))
    df = st.session_state.dfs[fname]

    c1, c2, c3, c4 = st.columns(4)
    x = c1.selectbox("X-Axis", df.columns)
    y = c2.selectbox("Y-Axis", ["None (Count)"] + list(df.columns))
    color = c3.selectbox("Color", [None] + list(df.columns))
    ctype = c4.selectbox("Type", ["Bar", "Scatter", "Line", "Box", "Histogram"])

    fig = None
    if ctype == "Bar":
        if y == "None (Count)":
            df_grouped = df.groupby([x, color] if color else [x]).size().reset_index(name="Count")
            y_col = "Count"
        else:
            df_grouped = df.groupby([x, color] if color else [x])[y].sum().reset_index()
            y_col = y
        df_grouped[x] = df_grouped[x].astype(str)
        fig = px.bar(df_grouped, x=x, y=y_col, color=color, barmode="group" if color else "relative")
        
    elif ctype in ["Scatter", "Line", "Box"]:
        plot_df = df.sample(5000) if len(df) > 5000 else df
        y_final = None if y == "None (Count)" else y
        
        if ctype == "Scatter":
            fig = px.scatter(plot_df, x=x, y=y_final, color=color)
        elif ctype == "Line":
            fig = px.line(plot_df, x=x, y=y_final, color=color)
        elif ctype == "Box":
            fig = px.box(plot_df, x=x, y=y_final, color=color)
    
    elif ctype == "Histogram":
        fig = px.histogram(df, x=x, color=color)

    if fig:
        st.plotly_chart(fig, use_container_width=True)

# ================= ML TAB =================
with t3:
    if not st.session_state.dfs:
        st.info("Upload data first.")
        st.stop()
        
    df_ml = st.session_state.dfs[fname]
    
    st.markdown("### Machine Learning Configuration")
    ml_task = st.radio("Task Type", ["Regression", "Classification", "Survival Analysis"], horizontal=True)

    # --- SURVIVAL ANALYSIS ---
    if ml_task == "Survival Analysis":
        if not HAS_SURVIVAL:
            st.error("Missing library. Run: `pip install scikit-survival`")
            st.stop()
            
        c1, c2 = st.columns(2)
        time_col = c1.selectbox("Time Column (Duration)", df_ml.columns)
        event_col = c2.selectbox("Event Column (1=Event, 0=Censored)", df_ml.columns)
        
        drop_cols = [time_col, event_col]
        valid_predictors = [c for c in df_ml.columns if c not in drop_cols]
        options = ["All"] + valid_predictors
        selected_options = st.multiselect("Predictors (X)", options, default=["All"])
        X_cols = valid_predictors if "All" in selected_options else selected_options
        
        if st.button("Train Survival Forest") and X_cols:
            X = df_ml[X_cols]
            
            # --- 1. SMART CLEANING (Prevents NaN Crashes) ---
            # Remove columns with >40% missing values
            missing_threshold = 0.4
            X = X.dropna(thresh=int((1 - missing_threshold) * len(X)), axis=1)
            
            # Remove ID-like columns (high cardinality strings)
            for col in X.select_dtypes(include=['object', 'category']).columns:
                if X[col].nunique() > 100 and X[col].nunique() > 0.5 * len(X):
                    X = X.drop(columns=[col])

            # Imputation for remaining numbers
            num_cols = X.select_dtypes(include=np.number).columns
            if len(num_cols) > 0:
                X[num_cols] = SimpleImputer(strategy='mean').fit_transform(X[num_cols])
            
            # One-Hot Encoding + Final Fill
            X = pd.get_dummies(X, drop_first=True).fillna(0)
            
            st.info(f"Using {X.shape[1]} features after cleaning (Dropped sparse/ID columns).")
            # ------------------------------------------------

            try:
                y = np.zeros(len(df_ml), dtype={'names':('event', 'time'), 'formats':('?', '<f8')})
                y['event'] = df_ml[event_col].astype(bool)
                y['time'] = df_ml[time_col].astype(float)
                
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                
                with st.spinner("Training Random Survival Forest..."):
                    rsf = RandomSurvivalForest(n_estimators=100, min_samples_split=10, min_samples_leaf=15, n_jobs=-1, random_state=42)
                    rsf.fit(X_train, y_train)
                    c_index = concordance_index_censored(y_test['event'], y_test['time'], rsf.predict(X_test))[0]
                
                st.success("Training Complete!")
                st.metric("C-Index (Test Score)", f"{c_index:.3f}", help="1.0 = Perfect, 0.5 = Random")
            except Exception as e:
                st.error(f"Error preparing survival data: {e}")

    # --- STANDARD ML BLOCK ---
    else:
        c1, c2 = st.columns(2)
        target = c1.selectbox("Target Column (y)", df_ml.columns)
        model_choice = c2.selectbox("Model", ["Random Forest", "XGBoost"])
        
        valid_predictors = [c for c in df_ml.columns if c != target]
        options = ["All"] + valid_predictors
        selected_options = st.multiselect("Predictors (X)", options, default=["All"])
        X_cols = valid_predictors if "All" in selected_options else selected_options
        
        if st.button(f"Train {model_choice}") and X_cols:
            X = df_ml[X_cols]
            y = df_ml[target]
            
            # --- 1. SMART CLEANING ---
            missing_threshold = 0.4
            X = X.dropna(thresh=int((1 - missing_threshold) * len(X)), axis=1)
            
            for col in X.select_dtypes(include=['object', 'category']).columns:
                if X[col].nunique() > 100 and X[col].nunique() > 0.5 * len(X):
                    X = X.drop(columns=[col])

            num_cols = X.select_dtypes(include=np.number).columns
            if len(num_cols) > 0:
                X[num_cols] = SimpleImputer(strategy='mean').fit_transform(X[num_cols])
            
            X = pd.get_dummies(X, drop_first=True).fillna(0)
            
            st.info(f"Using {X.shape[1]} features after cleaning.")
            # -------------------------
            
            # 2. ROBUST TARGET HANDLING
            if ml_task == "Regression":
                if not pd.api.types.is_numeric_dtype(y):
                    st.warning(f"⚠️ Target '{target}' is categorical. Converting classes to float numbers for regression.")
                    y = pd.factorize(y)[0]
                y = y.astype(float)
                if pd.isna(y).any():
                     y = pd.Series(y).fillna(pd.Series(y).mean())

            elif ml_task == "Classification":
                if y.dtype == 'object' or y.dtype.name == 'category':
                    y = pd.factorize(y)[0]
                if pd.Series(y).isnull().any():
                     y = pd.Series(y).fillna(pd.Series(y).mode()[0])

            # 3. TRAIN/TEST SPLIT
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # 4. MODEL INIT
            if ml_task == "Regression":
                if model_choice == "Random Forest":
                    model = RandomForestRegressor(n_estimators=100, random_state=42)
                else:
                    model = XGBRegressor(n_estimators=100, random_state=42)
            else: # Classification
                if model_choice == "Random Forest":
                    model = RandomForestClassifier(n_estimators=100, random_state=42)
                else:
                    model = XGBClassifier(n_estimators=100, random_state=42, use_label_encoder=False, eval_metric='logloss')

            # 5. TRAINING
            with st.spinner(f"Training {model_choice}..."):
                model.fit(X_train, y_train)
                preds = model.predict(X_test)
            
            # 6. METRICS
            st.subheader("📊 Model Performance")
            if ml_task == "Regression":
                r2 = r2_score(y_test, preds)
                rmse = np.sqrt(mean_squared_error(y_test, preds))
                mae = mean_absolute_error(y_test, preds)
                m1, m2, m3 = st.columns(3)
                m1.metric("R² Score", f"{r2:.3f}")
                m2.metric("RMSE", f"{rmse:.3f}")
                m3.metric("MAE", f"{mae:.3f}")
            else:
                acc = accuracy_score(y_test, preds)
                prec = precision_score(y_test, preds, average='weighted', zero_division=0)
                rec = recall_score(y_test, preds, average='weighted', zero_division=0)
                f1 = f1_score(y_test, preds, average='weighted', zero_division=0)
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Accuracy", f"{acc:.3f}")
                m2.metric("Precision", f"{prec:.3f}")
                m3.metric("Recall", f"{rec:.3f}")
                m4.metric("F1 Score", f"{f1:.3f}")
            
            # 7. FEATURE IMPORTANCE PLOT
            st.subheader("🔍 Feature Importance")
            if hasattr(model, "feature_importances_"):
                imp = model.feature_importances_
                feat_df = pd.DataFrame({"Feature": X.columns, "Importance": imp})
                feat_df = feat_df.sort_values(by="Importance", ascending=True).tail(15)
                
                fig_imp = px.bar(
                    feat_df, 
                    x="Importance", 
                    y="Feature", 
                    orientation='h',
                    title=f"Top 15 Features ({model_choice})",
                    color="Importance",
                    color_continuous_scale="Viridis"
                )
                st.plotly_chart(fig_imp, use_container_width=True)
            else:
                st.info("This model type does not support feature importance extraction directly.")