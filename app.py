import streamlit as st
import pandas as pd
import io

# Set page configuration
st.set_page_config(
    page_title="OncoVEP",
    page_icon="🧬",
    layout="centered"
)

def analyze_file(uploaded_file):
    """Processes uploaded VCF or TXT file and returns basic stats."""
    file_bytes = uploaded_file.getvalue()
    file_size_kb = len(file_bytes) / 1024.0

    # Read text lines to get overall line counts and ignore comments/headers
    lines = file_bytes.decode("utf-8", errors="replace").splitlines()
    total_lines = len(lines)

    # Check if file is a VCF file
    if uploaded_file.name.endswith('.vcf'):
        data_lines = [line for line in lines if not line.startswith('#')]
        comment_lines = total_lines - len(data_lines)
        variant_count = len(data_lines)
        return {
            "type": "VCF",
            "size_kb": file_size_kb,
            "total_lines": total_lines,
            "data_count": variant_count,
            "extra_info": f"Contains {comment_lines} header/comment lines and {variant_count} variant records."
        }

    # Otherwise treat as tab/comma delimited TXT / MAF file
    else:
        # Filter out comments starting with '#'
        data_lines = [line for line in lines if not line.startswith('#')]
        if not data_lines:
            return {
                "type": "TXT",
                "size_kb": file_size_kb,
                "total_lines": total_lines,
                "data_count": 0,
                "extra_info": "File contains no data rows."
            }

        # Try reading tabular data
        try:
            df = pd.read_csv(io.StringIO("\n".join(data_lines)), sep="\t")
            record_count = len(df)
            col_count = len(df.columns)
            return {
                "type": "TXT/Tabular",
                "size_kb": file_size_kb,
                "total_lines": total_lines,
                "data_count": record_count,
                "extra_info": f"Parsed {record_count} data records across {col_count} columns."
            }
        except Exception:
            # Fallback for plain text files
            return {
                "type": "TXT (Plain)",
                "size_kb": file_size_kb,
                "total_lines": total_lines,
                "data_count": len(data_lines),
                "extra_info": f"Processed {len(data_lines)} non-comment lines."
            }

# App UI
st.title("🧬 OncoVEP")
st.write("Oncology Variant Effect Predictor & Summary Tool")

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

    # Display 2–3 line summary stats
    st.subheader("📊 File Summary Stats")
    st.markdown(
        f"""
        - **File Size:** {stats['size_kb']:.2f} KB | **File Format:** {stats['type']}
        - **Total Lines:** {stats['total_lines']:,} lines | **Total Variants/Records:** {stats['data_count']:,}
        - **Details:** {stats['extra_info']}
        """
    )
else:
    st.info("Please upload a `.txt` or `.vcf` file to view summary statistics.")

