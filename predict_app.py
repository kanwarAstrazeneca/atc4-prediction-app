import streamlit as st
import pandas as pd
import joblib
from catboost import CatBoostClassifier
from st_aggrid import AgGrid, GridOptionsBuilder, JsCode

# --- Load Model & Encoder ---
@st.cache_resource
def load_model_and_encoder(model_path='catboost_model_v12.cbm', encoder_path='label_encoder_v12.pkl'):
    model = CatBoostClassifier()
    model.load_model(model_path)
    label_encoder = joblib.load(encoder_path)
    return model, label_encoder

# --- Prediction Function ---
def predict_atc4(model, label_encoder, input_dict):
    input_df = pd.DataFrame([input_dict])
    pred_encoded = model.predict(input_df)
    pred_label = label_encoder.inverse_transform(pred_encoded.reshape(1))[0]
    return pred_label

# --- Load Reference Data ---
@st.cache_data
def load_reference_data():
    df = pd.read_csv("merged_filtered_top100_atcs.csv")
    return df.head(1000)  # Adjust as needed

# --- Page Configuration ---
st.set_page_config(page_title="ATC4 Recode Prediction App", layout="wide")
st.title("🔮 ATC4 Recode Prediction App")

# --- Session State Init ---
for field in ["product", "corporation", "molecules", "atc4"]:
    if field not in st.session_state:
        st.session_state[field] = ""

# --- Paste Box for Copied Row ---
pasted_row = st.text_input("📋 Paste copied row here (optional)", key="pasted_row")

if pasted_row:
    parts = [p.strip() for p in pasted_row.split("\t")]
    if len(parts) == 4:
        st.session_state.product = parts[0]
        st.session_state.corporation = parts[1]
        st.session_state.molecules = parts[2]
        st.session_state.atc4 = parts[3]

# --- Input Form ---
col1, col2, col3, col4 = st.columns(4)
product = col1.text_input("Product", key="product")
corporation = col2.text_input("Corporation", key="corporation")
molecules = col3.text_input("Molecules", key="molecules")
atc4 = col4.text_input("ATC4", key="atc4")

# --- Predict Button ---
if st.button("🎯 Predict ATC4 Recode"):
    if all([product, corporation, molecules, atc4]):
        try:
            model, label_encoder = load_model_and_encoder()
            input_dict = {
                "product": product,
                "corporation": corporation,
                "molecules": molecules,
                "atc4": atc4,
            }
            prediction = predict_atc4(model, label_encoder, input_dict)
            st.success(f"🔮 Predicted ATC4 Recode: `{prediction}`")
        except Exception as e:
            st.error(f"❌ Prediction failed: {e}")
    else:
        st.warning("Please fill in all 4 fields.")

# --- Divider ---
st.markdown("---")
st.subheader("📘 Reference Table: Top 100 ATC4 Classes")

# --- Load and Modify Reference Data ---
df = load_reference_data()
df.insert(0, 'Copy', "")

# --- JS Copy Code for Entire Row ---
copy_js = JsCode("""
class CopyRowRenderer {
    init(params) {
        const data = params.data;
        const value = `${data.product}\\t${data.corporation}\\t${data.molecules}\\t${data.atc4}`;
        this.eGui = document.createElement('button');
        this.eGui.innerHTML = '📋';
        this.eGui.style.cursor = 'pointer';
        this.eGui.title = 'Copy row to input';
        this.eGui.addEventListener('click', () => {
            navigator.clipboard.writeText(value).then(() => {
                this.eGui.innerHTML = '✅';
                setTimeout(() => { this.eGui.innerHTML = '📋'; }, 1000);
            });
        });
    }
    getGui() {
        return this.eGui;
    }
}
""")

# --- AG Grid Settings ---
gb = GridOptionsBuilder.from_dataframe(df)
gb.configure_column("Copy", cellRenderer=copy_js, width=70)
gb.configure_pagination(paginationAutoPageSize=True)
gb.configure_default_column(groupable=True, filter=True, editable=False)

AgGrid(
    df,
    gridOptions=gb.build(),
    allow_unsafe_jscode=True,
    fit_columns_on_grid_load=True,
    height=500
)
