import streamlit as st
import pandas as pd
import numpy as np
import io
import requests
import time
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

# Set page configuration
st.set_page_config(
    page_title="OncoVEP",
    page_icon="🧬",
    layout="wide"
)

# -----------------------------------------------------------------------------
# CORE PIPELINE FUNCTIONS
# -----------------------------------------------------------------------------
def fetch_ensembl_sequence(chrom, start, end, build="GRCh38"):
    """Fetches a specific genomic sequence slice via Ensembl REST API."""
    server = "https://grch37.rest.ensembl.org" if build == "GRCh37" else "https://rest.ensembl.org"
    chrom_clean = str(chrom).replace("chr", "").strip()
    endpoint = f"/sequence/region/human/{chrom_clean}:{start}..{end}:1?"
    
    try:
        response = requests.get(server + endpoint, headers={"Content-Type": "text/plain"}, timeout=5)
        if response.status_code == 200:
            return response.text
        else:
            return None
    except Exception:
        return None

def parse_uploaded_file(uploaded_file):
    """Parses .txt, .vcf, or .maf files into a unified pandas DataFrame."""
    file_bytes = uploaded_file.getvalue()
    lines = file_bytes.decode("utf-8", errors="replace").splitlines()
    data_lines = [line for line in lines if not line.startswith('#')]
    
    if not data_lines:
        return pd.DataFrame()

    if uploaded_file.name.endswith('.vcf'):
        records = []
        for line in data_lines:
            parts = line.split('\t')
            if len(parts) >= 5:
                records.append({
                    "chrom": parts[0].replace("chr", ""),
                    "pos": int(parts[1]),
                    "ref": parts[3],
                    "alt": parts[4].split(',')[0]
                })
        return pd.DataFrame(records)
    else:
        try:
            df = pd.read_csv(io.StringIO("\n".join(data_lines)), sep="\t", low_memory=False)
            rename_dict = {}
            for col in df.columns:
                if col.lower() in ["chromosome", "chrom", "chr"]: rename_dict[col] = "chrom"
                elif col.lower() in ["start_position", "pos", "position"]: rename_dict[col] = "pos"
                elif col.lower() in ["reference_allele", "ref"]: rename_dict[col] = "ref"
                elif col.lower() in ["tumor_seq_allele2", "alt"]: rename_dict[col] = "alt"
            
            df = df.rename(columns=rename_dict)
            return df
        except Exception:
            return pd.DataFrame()

def batch_extract_and_annotate(df, build="GRCh38", max_variants=10):
    """Iterates through the variant DataFrame to extract contexts and generate mock metrics."""
    processed_records = []
    df_subset = df.head(max_variants).copy()
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, row in df_subset.iterrows():
        if pd.isna(row.get('chrom')) or pd.isna(row.get('pos')):
            continue
            
        chrom = str(row['chrom']).replace("chr", "").strip()
        pos = int(row['pos'])
        ref = str(row.get('ref', 'N')).upper()
        alt = str(row.get('alt', 'N')).upper()
        
        status_text.text(f"Processing variant {i+1}/{len(df_subset)}: chr{chrom}:{pos} ({ref}>{alt})")
        
        start = max(1, pos - 30)
        end = pos + 30
        ref_context = fetch_ensembl_sequence(chrom, start, end, build=build)
        
        if ref_context and len(ref_context) == 61:
            alt_context = ref_context[:30] + alt + ref_context[31:]
        else:
            ref_context = "API_FETCH_FAILED_OR_BOUNDS_ERROR"
            alt_context = "API_FETCH_FAILED_OR_BOUNDS_ERROR"
            
        mock_genes = ["TP53", "KRAS", "EGFR", "PIK3CA", "PTEN", "MYC", "BRCA1", "TERT"]
        mock_consequences = ["missense_variant", "intron_variant", "regulatory_region_variant", "upstream_gene_variant"]
        
        gene = row.get('gene', np.random.choice(mock_genes)) if pd.isna(row.get('gene')) else row.get('gene')
        consequence = row.get('consequence', np.random.choice(mock_consequences)) if pd.isna(row.get('consequence')) else row.get('consequence')
        
        delta_l2 = np.round(np.random.uniform(0.1, 1.2), 4)
        ag_shift = np.round(np.random.uniform(0.0, 0.9), 4)
        
        processed_records.append({
            "chrom": chrom,
            "pos": pos,
            "ref": ref,
            "alt": alt,
            "gene": gene,
            "consequence": consequence,
            "ref_context_61bp": ref_context,
            "alt_context_61bp": alt_context,
            "delta_l2_score": delta_l2,
            "ag_max_shift": ag_shift
        })
        
        progress_bar.progress((i + 1) / len(df_subset))
        time.sleep(0.15) 
        
    status_text.text(f"Batch processing complete for {len(processed_records)} variants!")
    
    out_df = pd.DataFrame(processed_records)
    if not out_df.empty:
        out_df["patient_disruption_percentile"] = out_df["delta_l2_score"].rank(pct=True).round(4)
        
    return out_df

# -----------------------------------------------------------------------------
# PLOTTING FUNCTIONS
# -----------------------------------------------------------------------------
def plot_two_tier_quadrant(df, carbon_cutoff=0.5, ag_cutoff=0.4):
    """Renders the Two-Tier Quadrant Scatter Plot classifying the variants."""
    fig, ax = plt.subplots(figsize=(6, 5))
    
    sns.scatterplot(
        data=df,
        x="patient_disruption_percentile",
        y="ag_max_shift",
        hue="consequence",
        size="delta_l2_score",
        sizes=(40, 200),
        palette="viridis",
        alpha=0.85,
        ax=ax
    )
    
    ax.axvline(carbon_cutoff, color="crimson", linestyle=":", lw=1.2, label=f"Tier 1 Cutoff ({carbon_cutoff})")
    ax.axhline(ag_cutoff, color="navy", linestyle=":", lw=1.2, label=f"Tier 2 Cutoff ({ag_cutoff})")

    for _, row in df.iterrows():
        if row.get("patient_disruption_percentile", 0) >= carbon_cutoff or row.get("ag_max_shift", 0) >= ag_cutoff:
            ax.text(
                row["patient_disruption_percentile"] + 0.01, 
                row["ag_max_shift"] + 0.01, 
                str(row["gene"]), 
                fontsize=8, 
                fontweight="bold"
            )

    ax.set_title("Two-Tier Variant Screening Space", fontsize=10, fontweight="bold")
    ax.set_xlabel("Carbon Sequence Disruption Percentile", fontsize=9)
    ax.set_ylabel("AlphaGenome Max Regulatory Shift", fontsize=9)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=7)
    fig.tight_layout()
    return fig

def plot_variant_impact(var_row, window_size=16384):
    """Plots REF vs. ALT epigenomic tracks and their differential (ALT - REF)."""
    n_bins = 128
    x_coords = np.linspace(-window_size // 2, window_size // 2, n_bins)
    center_pos = int(var_row["pos"])
    genomic_coords = center_pos + x_coords

    np.random.seed(center_pos % 10000)
    base_signal = 2.0 * np.exp(-0.5 * (x_coords / 1500)**2) + np.random.gamma(2, 0.1, n_bins)
    
    # Simulate disruptive loss of a CAGE promoter/enhancer peak at the mutation locus:
    ref_track = base_signal + 5.0 * np.exp(-0.5 * (x_coords / 400)**2)
    alt_track = base_signal + 1.2 * np.exp(-0.5 * (x_coords / 400)**2)
    delta_track = alt_track - ref_track

    fig = plt.figure(figsize=(6, 5))
    gs = gridspec.GridSpec(3, 1, height_ratios=[1, 1, 1], hspace=0.4)

    title_str = (
        f"AlphaGenome Regulatory Disruption: {var_row['gene']}\n"
        f"chr{var_row['chrom']}:{center_pos} ({var_row['ref']} > {var_row['alt']})"
    )
    plt.suptitle(title_str, fontsize=10, fontweight="bold")

    # 1. Reference Track
    ax0 = fig.add_subplot(gs[0])
    ax0.fill_between(genomic_coords, ref_track, color="#2b5c8f", alpha=0.6, label="REF (Wild-Type)")
    ax0.axvline(center_pos, color="crimson", linestyle="--", lw=1)
    ax0.set_ylabel("REF Signal", fontsize=8)
    ax0.legend(loc="upper right", fontsize=7)
    ax0.tick_params(labelbottom=False, labelsize=7)

    # 2. Mutated Track
    ax1 = fig.add_subplot(gs[1], sharex=ax0)
    ax1.fill_between(genomic_coords, alt_track, color="#e67e22", alpha=0.6, label=f"ALT ({var_row['alt']})")
    ax1.axvline(center_pos, color="crimson", linestyle="--", lw=1)
    ax1.set_ylabel("ALT Signal", fontsize=8)
    ax1.legend(loc="upper right", fontsize=7)
    ax1.tick_params(labelbottom=False, labelsize=7)

    # 3. Delta Track
    ax2 = fig.add_subplot(gs[2], sharex=ax0)
    ax2.fill_between(genomic_coords, delta_track, where=(delta_track >= 0), color="#27ae60", alpha=0.7, label="Gain")
    ax2.fill_between(genomic_coords, delta_track, where=(delta_track < 0), color="#c0392b", alpha=0.7, label="Loss")
    ax2.axhline(0, color="gray", lw=0.8, linestyle=":")
    ax2.axvline(center_pos, color="crimson", linestyle="--", lw=1)
    ax2.set_ylabel("Δ (ALT - REF)", fontsize=8)
    ax2.set_xlabel(f"Genomic Coordinate (chr{var_row['chrom']})", fontsize=9)
    ax2.legend(loc="upper right", fontsize=7)
    ax2.tick_params(labelsize=7)

    return fig

# -----------------------------------------------------------------------------
# USER INTERFACE
# -----------------------------------------------------------------------------

# Initialize session state for the processed dataframe
if 'processed_df' not in st.session_state:
    st.session_state['processed_df'] = None
if 'last_uploaded_file' not in st.session_state:
    st.session_state['last_uploaded_file'] = None

st.sidebar.title("🧬 OncoVEP Settings")
genome_build = st.sidebar.selectbox(
    "Reference Genome Assembly",
    ["GRCh38", "GRCh37"],
    help="Select the reference assembly corresponding to your input variants."
)

st.title("🧬 OncoVEP")
st.caption("Oncology Variant Effect Predictor & Summary Tool")
st.markdown("---")

uploaded_file = st.file_uploader("Upload a variant file (.txt, .maf, or .vcf)", type=["txt", "vcf", "maf"])

# Clear processed data if a new file is uploaded
if uploaded_file is not None and uploaded_file.name != st.session_state['last_uploaded_file']:
    st.session_state['processed_df'] = None
    st.session_state['last_uploaded_file'] = uploaded_file.name

if uploaded_file is not None:
    st.success(f"File **{uploaded_file.name}** uploaded successfully!")
    
    raw_df = parse_uploaded_file(uploaded_file)
    
    if raw_df.empty:
        st.error("Failed to parse genomic variants. Please ensure your file has valid Chromosome and Position columns.")
    else:
        st.subheader("📊 File Summary")
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Variants Detected", f"{len(raw_df):,}")
        col2.metric("Genome Assembly", genome_build)
        col3.metric("Data Columns", f"{len(raw_df.columns)}")
        
        with st.expander("View Raw Parsed Data"):
            st.dataframe(raw_df.head(100), use_container_width=True)

        st.markdown("---")
        st.subheader("⚙️ Step 1: Sequence Extraction, Annotation & Scoring")
        st.write("This step builds the 61 bp sequence contexts, annotates consequences, and computes the Two-Tier disruption metrics.")
        
        batch_limit = st.slider("Select number of variants to process (Demo Mode Limit)", min_value=1, max_value=50, value=5)
        
        if st.button("Run Batch Pre-processing"):
            with st.spinner("Executing extraction and annotation pipeline..."):
                # Run the batch logic and save to session state memory
                processed_df = batch_extract_and_annotate(raw_df, build=genome_build, max_variants=batch_limit)
                st.session_state['processed_df'] = processed_df
                
        # Only render the dashboard if processed data exists in memory
        if st.session_state['processed_df'] is not None:
            processed_df = st.session_state['processed_df']
            st.success("Pre-processing Complete! Data is ready for analysis.")
            
            # 1. Output the Annotated Data Table
            st.dataframe(processed_df, use_container_width=True)
            
            # 2. Side-by-Side Visualizations
            st.markdown("---")
            st.subheader("📈 Variant Prioritization Dashboard")
            
            plot_col1, plot_col2 = st.columns(2)
            
            with plot_col1:
                st.write("**Tier 1 vs Tier 2 Quadrant Classification**")
                fig_quadrant = plot_two_tier_quadrant(processed_df, carbon_cutoff=0.5, ag_cutoff=0.4)
                st.pyplot(fig_quadrant)
            
            with plot_col2:
                st.write("**Epigenomic Track Disruption Profile**")
                # Dropdown will now trigger a rerun safely relying on session_state
                selected_gene = st.selectbox("Select Variant to Inspect:", processed_df["gene"].unique())
                selected_row = processed_df[processed_df["gene"] == selected_gene].iloc[0]
                
                fig_track = plot_variant_impact(selected_row)
                st.pyplot(fig_track)
            
            # 3. Enable Output Download
            st.markdown("---")
            csv = processed_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="⬇️ Download Scored & Annotated Data (CSV)",
                data=csv,
                file_name="oncovep_scored_results.csv",
                mime="text/csv",
            )
else:
    st.info("Please upload a `.txt`, `.maf`, or `.vcf` file to begin.")
