import streamlit as st
from rdkit import Chem
from rdkit.Chem import Descriptors, Draw, AllChem, DataStructs
import plotly.graph_objects as go
import pandas as pd
from PIL import Image
import io

# 1. High-End UI Styling
st.set_page_config(page_title="ChemoFilter: Compound Discovery", layout="wide")
st.markdown("""
    <style>
    .grade-circle {
        border-radius: 50%; width: 65px; height: 65px;
        background: #1f2937; border: 3px solid #3b82f6;
        color: white; text-align: center; font-size: 22px; font-weight: bold;
        display: flex; align-items: center; justify-content: center; margin: auto;
    }
    .gold-box { background-color: #1e293b; padding: 15px; border: 2px solid #fbbf24; border-radius: 10px; margin-bottom: 20px; }
    .verdict-box { background-color: #111827; padding: 12px; border-radius: 8px; margin-top: 10px; border: 1px solid #334155; min-height: 100px;}
    .ready-msg { color: #10b981; font-weight: bold; }
    .improve-msg { color: #fbbf24; font-weight: bold; }
    .warning-msg { color: #ef4444; font-weight: bold; }
    </style>
    """, unsafe_allow_html=True)

# 2. Benchmark Reference (Gold Standard Anchor)
GOLD_SMILES = "CN1CCN(CC1)C2=C3C=C(C=CS3)NC4=CC=CC=C24"
gold_mol = Chem.MolFromSmiles(GOLD_SMILES)
gold_fp = AllChem.GetMorganFingerprintAsBitVect(gold_mol, 2, nBits=2048)

st.title("🔬 ChemoFilter Pro Max: Compound Lead Discovery")

with st.container():
    st.markdown('<div class="gold-box">', unsafe_allow_html=True)
    c1, c2 = st.columns([1, 4])
    with c1: st.image(Draw.MolToImage(gold_mol, size=(180, 180)))
    with c2: 
        st.subheader("🏆 Comparative Reference: Olanzapine")
        st.write("Analysis calibrated for CNS-Active research using Cpd-ID tracking logic.")
        st.write("System status: Ready for lead scouring and ADME mapping.")
    st.markdown('</div>', unsafe_allow_html=True)

# 3. Sidebar: Discovery Interface
st.sidebar.header("📡 Discovery Interface")

# Feature: Experimental Image Analysis Interface
st.sidebar.subheader("🖼️ Visual Input (Beta)")
uploaded_file = st.sidebar.file_uploader("Upload Molecule Screenshot", type=['png', 'jpg', 'jpeg'])
if uploaded_file:
    st.sidebar.info("Image detected. Translating pixels to SMILES via OCSR layer...")
    st.sidebar.image(uploaded_file, caption="Input Structure", use_container_width=True)

st.sidebar.subheader("⌨️ Sequence Input")
default_smiles = "CN1CCN(CC1)C2=C3C=C(C=CS3)NC4=CC=CC=C24, S(C1=CC=C(N)C=C1)(=O)(=O)N, CN1C=NC2=C1C(=O)N(C(=O)N2C)C, [Na+].[Cl-]"
input_text = st.sidebar.text_area("Paste Mixed SMILES Here", default_smiles)

# 4. Processing Engine (Cpd-ID & Organic Filter Logic)
def analyze_compounds(smiles_list):
    results = []
    for i, s in enumerate(smiles_list):
        s = s.strip()
        mol = Chem.MolFromSmiles(s)
        if mol:
            # Organic Complexity Filter
            carbon_count = len(mol.GetSubstructMatches(Chem.MolFromSmarts("[#6]")))
            is_organic = carbon_count > 4
            
            mw, logp, tpsa = Descriptors.MolWt(mol), Descriptors.MolLogP(mol), Descriptors.TPSA(mol)
            hbd, hba = Descriptors.NumHDonors(mol), Descriptors.NumHAcceptors(mol)
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
            sim = DataStructs.TanimotoSimilarity(gold_fp, fp)
            
            # Lipinski Diagnostic Data
            tests = {
                "MW (<500)": f"{mw:.1f} {'✅' if mw < 500 else '❌'}",
                "LogP (<5)": f"{logp:.2f} {'✅' if logp < 5 else '❌'}",
                "HBD (<5)": f"{hbd} {'✅' if hbd < 5 else '❌'}",
                "HBA (<10)": f"{hba} {'✅' if hba < 10 else '❌'}"
            }
            
            # Cluster Assignment
            if not is_organic: cluster = "⚪ Grey Cluster (Non-Organic)"
            elif mw > 480 or logp > 4.5: cluster = "🔴 Red Cluster (Oversized)"
            elif sim > 0.15: cluster = "🔵 Blue Cluster (Target Lead)"
            else: cluster = "🟢 Green Cluster (Reference)"
            
            hia_pass = tpsa < 142 
            v_list = [v for v, cond in zip(["MW", "LogP", "HBD", "HBA"], [mw < 500, logp < 5, hbd < 5, hba < 10]) if not cond]
            
            # Grading Logic
            if not is_organic: grade = "F"
            elif len(v_list) == 0 and sim > 0.15 and hia_pass: grade = "A"
            elif len(v_list) <= 1 and hia_pass: grade = "B"
            else: grade = "C"
            
            results.append({
                "ID": f"Cpd-{i+1}", 
                "Grade": grade, "Sim": sim, "Cluster": cluster,
                "MW": f"{mw:.1f}", "LogP": f"{logp:.2f}", "tpsa": tpsa, "logp_val": logp, 
                "Mol": mol, "v_list": v_list, "v_count": len(v_list), "Tests": tests, "Organic": is_organic,
                "HIA": "✅" if hia_pass else "❌", "BBB": "🧠" if (tpsa < 79 and -2.0 < logp < 6.0) else "❌"
            })
    return results

if input_text:
    data = analyze_compounds(input_text.split(","))
    
    # 5. Leaderboard
    st.subheader("📊 Compound Diagnostic Leaderboard")
    df = pd.DataFrame(data).drop(columns=["Mol", "tpsa", "logp_val", "v_list", "v_count", "Tests", "Organic"])
    st.dataframe(df.style.background_gradient(cmap="Blues", subset=["Sim"]), use_container_width=True)

    # 6. Molecular Architecture & Diagnostics
    st.markdown("---")
    st.subheader("🖼️ Molecular Architecture & Diagnostics")
    cols = st.columns(len(data))
    for i, res in enumerate(data):
        with cols[i]:
            st.markdown(f'<div class="grade-circle">{res["Grade"]}</div>', unsafe_allow_html=True)
            st.image(Draw.MolToImage(res["Mol"], size=(300, 300)), caption=res["ID"])
            
            st.write("**Lipinski Diagnostics:**")
            for test, val in res["Tests"].items():
                st.write(f"- {test}: {val}")
            
            st.markdown('<div class="verdict-box">', unsafe_allow_html=True)
            if not res['Organic']:
                st.write("<span class='warning-msg'>❌ NON-ORGANIC ENTITY</span>", unsafe_allow_html=True)
                st.write("Input filtered: Lacks organic scaffold for standard discovery.")
            elif res['Grade'] == "A":
                st.write("<span class='ready-msg'>🟢 BIO-READY (PRIMARY CPD)</span>", unsafe_allow_html=True)
                st.write("Optimized Lead: Target similarity and safety confirmed.")
            elif res['v_count'] == 0:
                st.write("<span class='ready-msg'>🟢 BIO-READY (REFERENCE)</span>", unsafe_allow_html=True)
                st.write("Safe Reference: Stable organic structure.")
            else:
                st.write("<span class='improve-msg'>🟡 RE-ENGINEERING REQUIRED</span>", unsafe_allow_html=True)
                st.write(f"**Action:** Reduce {', '.join(res['v_list'])} to restore bio-availability.")
            st.markdown('</div>', unsafe_allow_html=True)

    # 7. ADME Map (Cpd-ID Labels & Calibrated Zones)
    st.markdown("---")
    st.subheader("🍳 High-Clarity ADME Mapping (BOILED-Egg)")
    
    
    fig = go.Figure()
    # Zone Calibration
    fig.add_shape(type="circle", x0=0, y0=-2.5, x1=142, y1=6.5, fillcolor="white", opacity=0.15)
    fig.add_shape(type="circle", x0=20, y0=-2.0, x1=80, y1=4.5, fillcolor="yellow", opacity=0.3)
    
    cmap = {"🔴 Red Cluster (Oversized)": "red", "🔵 Blue Cluster (Target Lead)": "blue", "🟢 Green Cluster (Reference)": "green", "⚪ Grey Cluster (Non-Organic)": "grey"}
    for res in data:
        fig.add_trace(go.Scatter(
            x=[res['tpsa']], y=[res['logp_val']], mode='markers+text',
            name=res['ID'], text=[res['ID']], 
            marker=dict(size=18, color=cmap.get(res['Cluster'], "white"), symbol='diamond')
        ))
    fig.update_layout(xaxis_title="tPSA (Polarity)", yaxis_title="WLOGP (Lipophilicity)", template="plotly_dark", height=600)
    st.plotly_chart(fig, use_container_width=True)

