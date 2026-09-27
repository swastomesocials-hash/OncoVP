import streamlit as st
import pandas as pd
import io
import requests

# Set page configuration
st.set_page_config(
    page_title="OncoVEP",
    page_icon="🧬",
    layout="wide"
)

def fetch_ensembl_sequence(chrom, start, end, build="GRCh38"):
    """
    Fetches a specific genomic sequence slice via Ensembl REST API on the fly.
    No need to store or download 3GB FASTA files!
    """
    # Map build to Ensembl REST API endpoint
    if build == "GRCh37":
        server = "https://grch37.rest.ensembl.org"
    else:
        server = "https://rest.ensembl.org"
    
    # Format chromosome string (strip 'chr' if present)
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

def analyze_file(uploaded_file):
    """Processes uploaded VCF or TXT file and returns basic stats."""
    file_bytes = uploaded_file.getvalue()
    file_size_kb = len(file_bytes) / 1024.0
    
    lines = file_bytes.decode("utf-8", errors="replace").splitlines()
    total_lines = len(lines)
    data_lines = [line for line in lines if not line.startswith('#')]
    
    if uploaded_file.name.endswith('.vcf'):
        comment_lines = total_lines - len(data_lines)
        return {
            "type": "VCF",
            "size_kb": file_size_kb,
            "total_lines": total_lines,
            "data_count": len(data_lines),
            "extra_info": f"Contains {comment_lines} header lines and {len(data_lines)} variant records."
        }
    else:
        if not data_lines:
            return {
                "type": "TXT",
                "size_kb": file_size_kb,
                "total_lines": total_lines,
                "data_count": 0,
                "extra_info": "File contains no data rows."
            }
        try:
            df = pd.read_csv(io.StringIO("\n".join(data_lines)), sep="\t", low_memory=False)
            return {
                "type": "TXT/Tabular",
                "size_kb": file_size_kb,
                "total_lines": total_lines,
                "data_count": len(df),
                "extra_info": f"Parsed {len(df)} data records across {len(df.columns)} columns."
            }
        except Exception:
            return {
                "type": "TXT (Plain)",
                "size_kb": file_size_kb,
                "total_lines": total_lines,
                "data_count": len(data_lines),
                "extra_info": f"Processed {len(data_lines)} non-comment lines."
            }

# --- Sidebar Controls ---
st.sidebar.title("🧬 OncoVEP Settings")
genome_build = st.sidebar.selectbox(
    "Select Reference Genome Assembly",
    ["GRCh38", "GRCh37"],
    help="Select the reference assembly corresponding to your input variants."
)

# --- App UI ---
st.title("🧬 OncoVEP")
st.caption("Oncology Variant Effect Predictor & Summary Tool")

st.markdown("---")

# File Upload Section
uploaded_file = st.file_uploader(
    "Upload a variant file (.txt or .vcf)", 
    type=["txt", "vcf"]
)

if uploaded_file is not None:
    st.success(f"File **{uploaded_file.name}** uploaded successfully!")
    
    # Calculate stats
    stats = analyze_file(uploaded_file)
    
    # Display Summary
    st.subheader("📊 Summary Statistics")
    col1, col2, col3 = st.columns(3)
    col1.metric("File Size", f"{stats['size_kb']:.2f} KB")
    col2.metric("Total Variants", f"{stats['data_count']:,}")
    col3.metric("Genome Assembly", genome_build)
    
    st.info(f"**Format:** {stats['type']} | **Details:** {stats['extra_info']}")
    
    # Ensembl REST API Demo Box
    st.subheader("🔍 Sequence Context Fetcher (Live API Test)")
    st.write("Demonstration of on-demand sequence retrieval without storing full genome files:")
    
    test_col1, test_col2, test_col3 = st.columns(3)
    test_chrom = test_col1.text_input("Chromosome", "17")
    test_pos = test_col2.number_input("Variant Position", value=7577120)
    flank = test_col3.number_input("Flanking Window (bp)", value=30)
    
    if st.button("Fetch Sequence Context"):
        with st.spinner(f"Querying Ensembl {genome_build} API..."):
            start = max(1, test_pos - flank)
            end = test_pos + flank
            seq = fetch_ensembl_sequence(test_chrom, start, end, build=genome_build)
            if seq:
                st.code(f">{genome_build}_chr{test_chrom}:{start}-{end}\n{seq}", language="text")
            else:
                st.error("Could not fetch sequence. Check coordinates or assembly choice.")
else:
    st.info("Please upload a `.txt` or `.vcf` file to view summary statistics.")
