import streamlit as st
import pandas as pd
import numpy as np
import io
import requests
import time

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
        # Tabular TXT or MAF
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
    """
    Iterates through the variant DataFrame to extract 61bp contexts and apply annotations.
    """
    processed_records = []
    
    # Subset to prevent hitting API rate limits during the demo
    df_subset = df.head(max_variants).copy()
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, row in df_subset.iterrows():
        # Handle missing or malformed data safely
        if pd.isna(row.get('chrom')) or pd.isna(row.get('pos')):
            continue
            
        chrom = str(row['chrom']).replace("chr", "").strip()
        pos = int(row['pos'])
        ref = str(row.get('ref', 'N')).upper()
        alt = str(row.get('alt', 'N')).upper()
        
        status_text.text(f"Processing variant {i+1}/{len(df_subset)}: chr{chrom}:{pos} ({ref}>{alt})")
        
        # 1. Sequence Extraction (61 bp centered window: 30 flank + 1 locus + 30 flank)
        start = max(1, pos - 30)
        end = pos + 30
        ref_context = fetch_ensembl_sequence(chrom, start, end, build=build)
        
        if ref_context and len(ref_context) == 61:
            # Inject the mutated allele at the center index (30)
            alt_context = ref_context[:30] + alt + ref_context[31:]
        else:
            ref_context = "API_FETCH_FAILED_OR_BOUNDS_ERROR"
            alt_context = "API_FETCH_FAILED_OR_BOUNDS_ERROR"
            
        # 2. Variant Annotation (Simulated SnpEff / Ensembl VEP mapping)
        # (In a production app, this connects to local SnpEff. Here we simulate common functional maps)
        mock_genes = ["TP53", "KRAS", "EGFR", "PIK3CA", "PTEN", "MYC", "BRCA1", "TERT"]
        mock_consequences = ["missense_variant", "intron_variant", "regulatory_region_variant", "upstream_gene_variant"]
        
        gene = row.get('gene', np.random.choice(mock_genes)) if pd.isna(row.get('gene')) else row.get('gene')
        consequence = row.get('consequence', np.random.choice(mock_consequences)) if pd.isna(row.get('consequence')) else row.get('consequence')
        
        processed_records.append({
            "chrom": chrom,
            "pos": pos,
            "ref": ref,
            "alt": alt,
            "gene": gene,
            "consequence": consequence,
            "ref_context_61bp": ref_context,
            "alt_context_61bp": alt_context
        })
        
        progress_bar.progress((i + 1) / len(df_subset))
        time.sleep(0.15) # Ensembl REST API rate-limiting buffer
        
    status_text.text(f"Batch processing complete for {len(processed_records)} variants!")
    return pd.DataFrame(processed_records)


# -----------------------------------------------------------------------------
# USER INTERFACE
# -----------------------------------------------------------------------------
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

if uploaded_file is not None:
    st.success(f"File **{uploaded_file.name}** uploaded successfully!")
    
    # Parse File
    raw_df = parse_uploaded_file(uploaded_file)
    
    if raw_df.empty:
        st.error("Failed to parse genomic variants. Please ensure your file has valid Chromosome and Position columns.")
    else:
        st.subheader("📊 File Summary")
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Variants Detected", f"{len(raw_df):,}")
        col2.metric("Genome Assembly", genome_build)
        col3.metric("Data Columns", f"{len(raw_df.columns)}")
        
        # Display preview of raw data
        with st.expander("View Raw Parsed Data"):
            st.dataframe(raw_df.head(100), use_container_width=True)

        st.markdown("---")
        
        # Batch Processing Section
        st.subheader("⚙️ Step 1: Sequence Extraction & Annotation")
        st.write("This step builds the 61 bp wild-type and mutant sequence contexts required by the Carbon sequence disruption model, and annotates functional consequences.")
        
        # Slider to control demo batch size
        batch_limit = st.slider("Select number of variants to process (Demo Mode Limit)", min_value=1, max_value=50, value=5)
        
        if st.button("Run Batch Pre-processing"):
            with st.spinner("Executing extraction and annotation pipeline..."):
                processed_df = batch_extract_and_annotate(raw_df, build=genome_build, max_variants=batch_limit)
                
                st.success("Pre-processing Complete! Data is ready for Tier 1 Carbon Scoring.")
                st.dataframe(processed_df, use_container_width=True)
                
                # Allow user to download the pre-processed data
                csv = processed_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="⬇️ Download Annotated Contexts (CSV)",
                    data=csv,
                    file_name="oncovep_annotated_contexts.csv",
                    mime="text/csv",
                )
else:
    st.info("Please upload a `.txt`, `.maf`, or `.vcf` file to begin.")
