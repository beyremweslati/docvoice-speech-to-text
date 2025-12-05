import streamlit as st
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from peft import PeftModel, PeftConfig
import time
import librosa
import io
from fpdf import FPDF

st.set_page_config(
    page_title="DocVoice: Medical Reporter",
    page_icon="🩺",
    layout="centered"
)

@st.cache_resource
def load_model():
    
    lora_dir = "./whisper-docvoice-lora/final_adapter" 
    print(f"Loading adapter from: {lora_dir}")
    
    model_id = "openai/whisper-medium.en"


    base_model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True
    )

    processor = AutoProcessor.from_pretrained(model_id)

    model = PeftModel.from_pretrained(base_model, lora_dir)
    model = model.merge_and_unload()  

    model.to(device)

    return model, processor

def transcribe(audio_np):
    SAMPLE_RATE = 16000

    print("Transcribing, please wait.")
    inputs = processor(
        audio_np,
        sampling_rate=SAMPLE_RATE,
        return_tensors="pt"
    ).to(device)

    inputs["input_features"] = inputs["input_features"].to(dtype=torch_dtype)

    with torch.no_grad():
        predicted_ids = model.generate(
            inputs["input_features"],
            max_length=448
        )

    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
    return transcription

def create_pdf(patient_name, date, full_text):
    pdf = FPDF()
    pdf.add_page()
    
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(200, 10, txt="Medical Consultation Report", ln=True, align='C')
    pdf.cell(200, 10, txt="", ln=True, align='C')
    pdf.ln(10)
    
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, txt=f"Patient: {patient_name}", ln=True)
    pdf.cell(0, 10, txt=f"Date: {date}", ln=True)
    pdf.line(10, 35, 200, 35)
    pdf.ln(10)
    
    pdf.set_font("Arial", '', 11)
    pdf.multi_cell(0, 7, txt=full_text.strip())
            
    return pdf.output(dest='S').encode('latin-1', 'replace')

st.title("DocVoice: Speech to Text")
st.markdown("### Automated Medical Reporting")
st.markdown("---")

# Load model with a spinner
with st.spinner("Loading AI Model..."):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    model, processor = load_model()
# --- Input Section ---
col1, col2 = st.columns(2)
with col1:
    patient_name = st.text_input("Patient Name", placeholder="e.g., John Doe")
with col2:
    report_date = st.date_input("Date of Visit")

st.markdown("### Dictation")

audio_value = st.audio_input("Record your observation")

if audio_value is not None:
    st.audio(audio_value)
    
    if st.button("Generate Report", type="primary"):
        if not patient_name:
            st.warning("Please enter a patient name first.")
        else:
            with st.spinner("Transcribing..."):
                audio_bytes = audio_value.read()
                audio_np, _ = librosa.load(io.BytesIO(audio_bytes), sr=16000)

                result = transcribe(audio_np)
            pdf_bytes = create_pdf(patient_name, report_date, result)
            st.success("Transcription Complete!")
            final_report = f"""
**MEDICAL REPORT**
-----------------
**Patient:** {patient_name} \n
**Date:** {report_date}

**Observations:**
{result}
            """
            
            st.markdown("### Final Report Preview")
            st.info(final_report)
            
            # Download Button
            st.download_button(
            label="Download PDF",
            data=pdf_bytes,
            file_name=f"{patient_name}_Report.pdf",
            mime="application/pdf"
        )