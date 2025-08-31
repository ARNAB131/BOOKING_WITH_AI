# app.py
import streamlit as st
import pandas as pd
import io
import os
import re
import math
import fitz  # PyMuPDF for PDF uploads
from fpdf import FPDF
from datetime import datetime, timedelta, date
from ai_booking import recommend_doctors, symptom_specialization_map, generate_slots
import streamlit.components.v1 as components

# ------------------------------------------------------------------------------------
# Setup
# ------------------------------------------------------------------------------------
os.makedirs("data", exist_ok=True)
st.set_page_config(page_title="Doctigo – AI Doctor & Bed/Cabin Booking", page_icon="🩺", layout="centered")

# ------------------------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------------------------
def _normalize_text(s: str) -> str:
    if not isinstance(s, str):
        return ""
    s = s.casefold()
    s = re.sub(r"[^a-z0-9\s]", " ", s)  # keep alnum + space
    s = re.sub(r"\s+", " ", s).strip()
    return s

def load_hospitals():
    path = "hospitals.csv"
    if not os.path.exists(path):
        st.error("❌ hospitals.csv not found in project root.")
        return pd.DataFrame()
    df = pd.read_csv(path)
    needed = {"Hospital", "Address", "Latitude", "Longitude"}
    if not needed.issubset(set(df.columns)):
        st.error(f"❌ hospitals.csv missing columns. Required: {needed}")
        return pd.DataFrame()
    return df

def load_doctors_file():
    """
    Load doctors from 'doctor.csv' or 'doctors.csv' (whichever exists).
    Required columns: Doctor Name, Specialization, Chamber, Visiting Time
    Optional: Hospital
    """
    path = None
    if os.path.exists("doctor.csv"):
        path = "doctor.csv"
    elif os.path.exists("doctors.csv"):
        path = "doctors.csv"
    else:
        st.error("❌ Neither doctor.csv nor doctors.csv found in project root.")
        return pd.DataFrame()

    df = pd.read_csv(path)
    needed = {"Doctor Name", "Specialization", "Chamber", "Visiting Time"}
    if not needed.issubset(set(df.columns)):
        st.error(f"❌ {path} is missing required columns. Required: {needed}")
        return pd.DataFrame()
    return df

def filter_doctors_by_hospital(doctor_df: pd.DataFrame, hospital_name: str, hospital_address: str = "") -> pd.DataFrame:
    """
    Priority:
      1) If doctor_df has 'Hospital', do exact casefold equality match.
      2) Else, fuzzy/substring: 'Chamber' contains hospital name OR tokens from address.
    """
    if doctor_df.empty or not hospital_name:
        return doctor_df

    # 1) Exact hospital column match
    if "Hospital" in doctor_df.columns:
        mask = doctor_df["Hospital"].astype(str).str.casefold() == hospital_name.casefold()
        exact = doctor_df[mask]
        if not exact.empty:
            return exact

    # 2) Fallback: chamber text match
    hn = _normalize_text(hospital_name)
    ha = _normalize_text(hospital_address)
    address_tokens = {t for t in ha.split(" ") if len(t) >= 4}

    def _ok(chamber_val: str) -> bool:
        ch = _normalize_text(chamber_val)
        if not ch:
            return False
        if hn and hn in ch:
            return True
        if address_tokens and any(tok in ch for tok in address_tokens):
            return True
        return False

    return doctor_df[doctor_df["Chamber"].astype(str).apply(_ok)]

def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(p1)*math.cos(p2)*math.sin(dl/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    return R * c

# --- Original appointment PDF (kept) ---
def generate_pdf_receipt(patient_name, doctor, chamber, slot, symptoms):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", "B", 18)
    pdf.cell(0, 10, "Doctigo", ln=True, align='C')
    pdf.set_font("Arial", "B", 12)
    pdf.ln(5)
    pdf.cell(0, 10, "Patient Appointment Receipt", ln=True, align='C')
    pdf.set_font("Arial", "", 11)
    pdf.cell(0, 5, "----------------------------------------", ln=True, align='C')
    pdf.ln(5)
    pdf.set_font("Arial", "", 12)
    pdf.cell(0, 8, f"Patient Name: {patient_name}", ln=True)
    pdf.cell(0, 8, f"Doctor: Dr. {doctor}", ln=True)
    pdf.cell(0, 8, f"Chamber: {chamber}", ln=True)
    pdf.cell(0, 8, f"Slot: {slot}", ln=True)
    pdf.multi_cell(0, 8, f"Symptoms: {symptoms}")
    pdf.cell(0, 8, f"Issued On: {datetime.now().strftime('%d %B %Y, %I:%M %p')}", ln=True)
    pdf.ln(10)
    pdf.set_font("Courier", "B", 11)
    pdf.set_text_color(150)
    pdf.cell(0, 10, "--- Tear Here ---", ln=True, align='C')
    pdf.set_text_color(0)
    pdf.ln(5)
    pdf.set_font("Arial", "", 12)
    pdf.cell(0, 8, "Hospital Copy", ln=True)
    pdf.cell(0, 8, f"Patient Name: {patient_name}", ln=True)
    pdf.cell(0, 8, f"Doctor: Dr. {doctor}", ln=True)
    pdf.cell(0, 8, f"Chamber: {chamber}", ln=True)
    pdf.cell(0, 8, f"Slot: {slot}", ln=True)
    pdf.multi_cell(0, 8, f"Symptoms: {symptoms}")
    pdf.cell(0, 8, f"Issued On: {datetime.now().strftime('%d %B %Y, %I:%M %p')}", ln=True)
    pdf.ln(10)
    pdf.set_font("Arial", "I", 9)
    pdf.set_text_color(120)
    pdf.cell(0, 10, "This receipt is auto-generated by Doctigo AI Booking System.", ln=True, align="C")
    pdf_bytes = pdf.output(dest="S").encode("latin1")
    return io.BytesIO(pdf_bytes)

# --- NEW: combined PDF for appointment + bed/cabin ---
def generate_full_pdf(hospital_name, patient, appointment, bed_choice):
    """
    hospital_name: str
    patient: dict -> name, attendant_name, attendant_phone, patient_phone, patient_age,
                     patient_email, patient_address, checkin_date, checkout_mode, checkout_date
    appointment: dict or None -> doctor, chamber, slot, symptoms
    bed_choice: dict or None -> tier, unit_id, price, features(list)
    """
    pdf = FPDF()
    pdf.add_page()

    # Header
    pdf.set_font("Arial", "B", 18)
    pdf.cell(0, 10, hospital_name or "Doctigo", ln=True, align='C')

    pdf.set_font("Arial", "B", 12)
    pdf.ln(3)
    pdf.cell(0, 8, "Booking Summary", ln=True, align='C')
    pdf.set_font("Arial", "", 11)
    pdf.cell(0, 5, "----------------------------------------", ln=True, align='C')

    # Patient Details
    pdf.ln(4)
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 8, "Patient Details", ln=True)
    pdf.set_font("Arial", "", 11)

    checkout_str = "To be decided (post-discharge)" if patient.get("checkout_mode") == "unknown" else patient.get("checkout_date", "")
    pdf.multi_cell(0, 7, f"""
Patient Name: {patient.get('name','')}
Attendant Name: {patient.get('attendant_name','')}
Attendant Phone: {patient.get('attendant_phone','')}
Patient Phone: {patient.get('patient_phone','')}
Patient Age: {patient.get('patient_age','')}
Patient Email: {patient.get('patient_email','')}
Patient Address: {patient.get('patient_address','')}
Check-in: {patient.get('checkin_date','')}
Check-out: {checkout_str}
Issued On: {datetime.now().strftime('%d %B %Y, %I:%M %p')}
""".strip())

    # Appointment
    if appointment:
        pdf.ln(3)
        pdf.set_font("Arial", "B", 12)
        pdf.cell(0, 8, "Doctor Appointment", ln=True)
        pdf.set_font("Arial", "", 11)
        pdf.multi_cell(0, 7, f"""
Doctor: Dr. {appointment.get('doctor','')}
Chamber: {appointment.get('chamber','')}
Slot: {appointment.get('slot','')}
Symptoms: {appointment.get('symptoms','')}
""".strip())

    # Bed/Cabin
    if bed_choice:
        pdf.ln(3)
        pdf.set_font("Arial", "B", 12)
        pdf.cell(0, 8, "Bed/Cabin Booking", ln=True)
        pdf.set_font("Arial", "", 11)
        features_text = ", ".join(bed_choice.get('features', []))
        pdf.multi_cell(0, 7, f"""
Type: {bed_choice.get('tier','')}
Unit ID: {bed_choice.get('unit_id','Any')}
Price per night: ₹{bed_choice.get('price','')}
Features: {features_text}
""".strip())

    pdf.ln(6)
    pdf.set_font("Arial", "I", 9)
    pdf.set_text_color(120)
    pdf.cell(0, 8, "This receipt is auto-generated by Doctigo AI System.", ln=True, align="C")
    pdf.set_text_color(0)

    pdf_bytes = pdf.output(dest="S").encode("latin1")
    return io.BytesIO(pdf_bytes)

# ------------------------------------------------------------------------------------
# Session state
# ------------------------------------------------------------------------------------
if "flow_step" not in st.session_state:
    st.session_state.flow_step = "home"  # home -> emergency -> hospital -> doctors -> beds -> details -> summary
for key, default in [
    ("selected_hospital", None),
    ("appointment", None),
    ("bed_choice", None),
    ("patient_details", None),
    ("user_location", {"lat": None, "lon": None}),
    ("recommendations", []),
    ("doctor_message", ""),
    ("booked", False),
    ("booked_doctor", ""),
    ("slot", ""),
    ("symptoms_used", []),
]:
    if key not in st.session_state:
        st.session_state[key] = default

# ------------------------------------------------------------------------------------
# Bulk Appointment Upload (kept)
# ------------------------------------------------------------------------------------
st.markdown("##  Bulk Appointment Upload")
uploaded_file = st.file_uploader("Upload patient list (CSV or PDF)", type=["csv", "pdf"])

if uploaded_file:
    try:
        if uploaded_file.name.endswith(".csv"):
            df_bulk = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith(".pdf"):
            text = ""
            doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
            for page in doc:
                text += page.get_text()
            rows = [line.strip() for line in text.split('\n') if 'Name:' in line and 'Symptoms:' in line]
            data = []
            for row in rows:
                parts = row.split(',')
                name = parts[0].split('Name:')[1].strip()
                symptoms = parts[1].split('Symptoms:')[1].strip()
                data.append({"Patient Name": name, "Symptoms": symptoms})
            df_bulk = pd.DataFrame(data)
        else:
            st.warning("Unsupported file format")
            df_bulk = pd.DataFrame()

        if not df_bulk.empty:
            st.success(f"✅ Loaded {len(df_bulk)} patients. Starting auto booking...")
            try:
                for _, row in df_bulk.iterrows():
                    name = row["Patient Name"]
                    symptoms = [s.strip() for s in row["Symptoms"].split(',') if s.strip()]
                    msg, recs = recommend_doctors(symptoms)
                    if recs:
                        selected = recs[0]
                        slot = selected['Slots'][0]
                        full_slot = f"{slot} on {datetime.today().strftime('%d %B %Y')}"
                        appt = pd.DataFrame([{
                            "Patient Name": name,
                            "Symptoms": '; '.join(symptoms),
                            "Doctor": selected['Doctor'],
                            "Slot": full_slot,
                            "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        }])
                        if os.path.exists("appointments.csv"):
                            appt.to_csv("appointments.csv", mode="a", header=False, index=False)
                        else:
                            appt.to_csv("appointments.csv", mode="w", header=True, index=False)
                st.success(f"✅ Successfully booked appointments for {len(df_bulk)} patients.")
            except Exception as e_inner:
                st.error(f"Error during auto-booking loop: {e_inner}")
    except Exception as e:
        st.error(f"Error processing bulk upload: {e}")

# ------------------------------------------------------------------------------------
# HOME
# ------------------------------------------------------------------------------------
st.title("🤖 Doctigo – AI Doctor & Bed/Cabin Booking")
col1, col2 = st.columns([1, 1])
with col1:
    if st.button("🔍 Normal Booking"):
        st.session_state.flow_step = "doctors"
with col2:
    if st.button("🚨 Emergency"):
        st.session_state.flow_step = "emergency"

# ------------------------------------------------------------------------------------
# EMERGENCY → Nearest Hospitals (Idea 1)
# ------------------------------------------------------------------------------------
if st.session_state.flow_step == "emergency":
    st.subheader("🚑 Nearest Hospitals")
    st.caption("Tap **Detect My Location** to rank nearby hospitals by distance.")

    # Inputs for location (populated by JS)
    lat_val = st.text_input("Latitude", value=st.session_state.user_location["lat"] or "", key="lat_input")
    lon_val = st.text_input("Longitude", value=st.session_state.user_location["lon"] or "", key="lon_input")

    components.html("""
        <button onclick="getLoc()" style="padding:8px 14px;">📍 Detect My Location</button>
        <p id="locout" style="margin-top:8px;"></p>
        <script>
        function getLoc(){
          if(navigator.geolocation){
            navigator.geolocation.getCurrentPosition(function(pos){
              const lat = pos.coords.latitude.toFixed(6);
              const lon = pos.coords.longitude.toFixed(6);
              document.getElementById('locout').innerText = "✓ Location captured: " + lat + ", " + lon;
              const inputs = window.parent.document.querySelectorAll('input[type="text"]');
              for (let i=0;i<inputs.length;i++){
                const lbl = inputs[i].previousSibling && inputs[i].previousSibling.textContent ? inputs[i].previousSibling.textContent : "";
                if (lbl.includes("Latitude")) {
                  inputs[i].value = lat; inputs[i].dispatchEvent(new Event('input', { bubbles: true }));
                }
                if (lbl.includes("Longitude")) {
                  inputs[i].value = lon; inputs[i].dispatchEvent(new Event('input', { bubbles: true }));
                }
              }
            }, function(err){
              document.getElementById('locout').innerText = "❌ " + err.message;
            });
          } else {
            document.getElementById('locout').innerText = "❌ Geolocation not supported in this browser.";
          }
        }
        </script>
    """, height=120)

    hospitals_df = load_hospitals()

    # Store to session if we have lat/lon
    try:
        st.session_state.user_location["lat"] = float(lat_val) if lat_val else None
        st.session_state.user_location["lon"] = float(lon_val) if lon_val else None
    except:
        pass

    if not hospitals_df.empty and st.session_state.user_location["lat"] and st.session_state.user_location["lon"]:
        lat0 = st.session_state.user_location["lat"]
        lon0 = st.session_state.user_location["lon"]

        hospitals_df["Distance_km"] = hospitals_df.apply(
            lambda r: haversine_km(lat0, lon0, float(r["Latitude"]), float(r["Longitude"])), axis=1
        )
        hospitals_df = hospitals_df.sort_values("Distance_km").reset_index(drop=True)

        st.markdown("### 🏥 Nearby options")
        for i, row in hospitals_df.head(5).iterrows():
            label = "Most Nearest" if i == 0 else ("Next Nearest" if i == 1 else f"Option {i+1}")
            st.markdown(f"**{label}: {row['Hospital']}**  \n{row['Address']}  \nDistance: {row['Distance_km']:.2f} km")
            if st.button(f"Select {row['Hospital']}", key=f"pick_hosp_{i}"):
                st.session_state.selected_hospital = row['Hospital']
                st.session_state.flow_step = "hospital"
    else:
        st.info("Enter Latitude/Longitude (or use Detect My Location).")

# ------------------------------------------------------------------------------------
# HOSPITAL/DOCTORS PAGE (Idea 2)
# ------------------------------------------------------------------------------------
if st.session_state.flow_step in ("hospital", "doctors"):
    doctor_df = load_doctors_file()
    if doctor_df.empty:
        st.stop()

    hospitals_df = load_hospitals()
    chosen_hospital = st.session_state.selected_hospital

    # Hospital header (if chosen)
    if st.session_state.flow_step == "hospital" and chosen_hospital:
        st.subheader(f"🏥 {chosen_hospital}")
    else:
        st.subheader("🧑‍⚕️ Find Doctors")

    # Resolve selected hospital address to improve matching
    hospital_address = ""
    if chosen_hospital and not hospitals_df.empty:
        row = hospitals_df[hospitals_df["Hospital"].astype(str).str.casefold() == chosen_hospital.casefold()]
        if not row.empty and "Address" in row.columns:
            hospital_address = str(row.iloc[0]["Address"])

    # Patient name
    patient_name = st.text_input("Enter your name:", key="patient_name")

    # Voice-to-text (kept)
    st.markdown("### 🎙️ Voice Input for Patient Name")
    components.html(
        """
        <button onclick="startDictation()" style="padding: 8px 16px; font-size: 16px; background-color: #4CAF50; color: white; border: none; border-radius: 5px;">🎤 Start Voice Input</button>
        <p id="output" style="margin-top: 10px; font-weight: bold;"></p>
        <script>
        function startDictation() {
            if (!('webkitSpeechRecognition' in window)) {
                document.getElementById("output").innerText = "❌ Speech recognition not supported in this browser.";
                return;
            }
            const recognition = new webkitSpeechRecognition();
            recognition.lang = "en-US";
            recognition.continuous = false;
            recognition.interimResults = false;
            recognition.onstart = function() {
                document.getElementById("output").innerText = "🎙️ Listening...";
            };
            recognition.onresult = function(event) {
                const transcript = event.results[0][0].transcript;
                document.getElementById("output").innerText = "✅ You said: " + transcript;
                const streamlitInput = window.parent.document.querySelectorAll('input[type="text"]')[0];
                streamlitInput.value = transcript;
                streamlitInput.dispatchEvent(new Event('input', { bubbles: true }));
            };
            recognition.onerror = function(event) {
                document.getElementById("output").innerText = "❌ Error: " + event.error;
            };
            recognition.start();
        }
        </script>
        """,
        height=180,
    )

    # Symptoms input
    symptom_options = list(symptom_specialization_map.keys())
    symptoms_selected = st.multiselect("Select your symptom(s):", options=symptom_options)
    symptoms_typed = st.text_input("Or type your symptoms (comma-separated):", key="typed_symptoms")
    all_symptoms = list(set(symptoms_selected + [s.strip() for s in symptoms_typed.split(',') if s.strip()]))

    # Filter doctors to chosen hospital (by Hospital column or Chamber text)
    filtered_doctors = doctor_df.copy()
    if chosen_hospital:
        filtered_doctors = filter_doctors_by_hospital(doctor_df, chosen_hospital, hospital_address)
        if filtered_doctors.empty:
            st.info("No exact matches by Hospital/Chamber; showing all doctors.")
            filtered_doctors = doctor_df

    # Direct booking UI (kept, uses filtered doctors)
    st.markdown("---")
    st.subheader("📋 Book a Doctor")
    doctor_names = filtered_doctors['Doctor Name'].unique().tolist()
    selected_doctor = st.selectbox("Choose a doctor (optional):", ["None"] + doctor_names)

    direct_slot = ""
    selected_date = None
    if selected_doctor != "None":
        selected_info = filtered_doctors[filtered_doctors['Doctor Name'] == selected_doctor].iloc[0]
        st.markdown(f"**Specialization:** {selected_info['Specialization']}")
        st.markdown(f"**Chamber:** {selected_info['Chamber']}")
        slots = generate_slots(selected_info['Visiting Time'])
        direct_slot = st.selectbox("Select a slot:", slots, key="direct_slot")
        today = datetime.today()
        selected_date = st.date_input("Choose appointment date:", min_value=today)
        if st.button(f"📅 Book Appointment with Dr. {selected_doctor}", key="direct_book"):
            st.session_state.appointment = {
                "doctor": selected_doctor,
                "chamber": selected_info["Chamber"],
                "slot": f"{direct_slot} on {selected_date.strftime('%d %B %Y')}",
                "symptoms": "; ".join(all_symptoms or ["Direct Booking"])
            }
            st.success(f"✅ Appointment Booked with Dr. {selected_doctor} at {st.session_state.appointment['slot']}")

    # AI recommendations (kept) + hospital-bound filter
    if st.button("🔍 Find Doctors (AI)") and all_symptoms:
        message, recommendations = recommend_doctors(all_symptoms)
        if chosen_hospital and recommendations:
            recs_df = pd.DataFrame(recommendations)
            if not recs_df.empty and "Chamber" in recs_df.columns:
                mask = recs_df["Chamber"].astype(str).str.contains(chosen_hospital, case=False, na=False)
                recommendations = recs_df[mask].to_dict("records") if mask.any() else recommendations
        st.session_state.recommendations = recommendations
        st.session_state.doctor_message = message

    st.subheader(st.session_state.doctor_message)
    if st.session_state.recommendations:
        for idx, doc in enumerate(st.session_state.recommendations):
            with st.expander(f"{idx+1}. Dr. {doc['Doctor']} - {doc['Specialization']}"):
                st.markdown(f"**Chamber:** {doc['Chamber']}")
                slot = st.selectbox(f"Select a slot for Dr. {doc['Doctor']}", options=doc['Slots'], key=f"slot_{idx}")
                appt_date = st.date_input(f"Choose date for Dr. {doc['Doctor']}", key=f"date_{idx}", min_value=datetime.today())
                if st.button(f"Book Appointment with Dr. {doc['Doctor']}", key=f"book_{idx}"):
                    st.session_state.appointment = {
                        "doctor": doc["Doctor"],
                        "chamber": doc["Chamber"],
                        "slot": f"{slot} on {appt_date.strftime('%d %B %Y')}",
                        "symptoms": "; ".join(all_symptoms)
                    }
                    st.success(f"✅ Appointment Booked with Dr. {doc['Doctor']} at {slot} on {appt_date.strftime('%d %B %Y')}")

    # Book Beds/Cabins
    st.markdown("---")
    if st.button("🛏️ Book Beds/Cabins"):
        if not st.session_state.selected_hospital and chosen_hospital:
            st.session_state.selected_hospital = chosen_hospital
        st.session_state.flow_step = "beds"

# ------------------------------------------------------------------------------------
# BEDS/CABINS (Idea 3)
# ------------------------------------------------------------------------------------
if st.session_state.flow_step == "beds":
    st.subheader(f"🛏️ Beds & Cabins – {st.session_state.selected_hospital or 'Hospital'}")
    st.caption("Choose your option. Click the info to view inclusions.")

    tiers = [
        {"tier": "General Bed", "price": 100, "features": ["1 bed", "1 chair", "bed table"], "ids": [f"G-{i}" for i in range(1, 41)]},
        {"tier": "General Cabin", "price": 1000, "features": ["2 beds", "attached washroom", "bed table", "chair", "food x3 times"], "ids": [f"C-{i}" for i in range(1, 21)]},
        {"tier": "VIP Cabin", "price": 4000, "features": ["premium bed x2", "sofa", "Air Conditioning", "attached washroom", "TV", "fridge", "bed table x2", "coffee table", "2 chairs"], "ids": [f"V-{i}" for i in range(1, 11)]},
    ]

    # Select tier
    tier_names = [t["tier"] for t in tiers]
    pick_tier = st.radio("Select type", options=tier_names)  # no horizontal=True for 1.26 compat
    tier_obj = next(t for t in tiers if t["tier"] == pick_tier)

    with st.expander(f"ℹ️ What is included in {pick_tier}?"):
        st.markdown(f"**₹{tier_obj['price']} per night**")
        st.markdown("- " + "\n- ".join(tier_obj["features"]))

    st.markdown("#### Choose a specific unit (optional)")
    unit_id = st.selectbox("Pick a bed/cabin ID (or leave as 'Any available')", options=["Any available"] + tier_obj["ids"])

    if st.button("Next ➜"):
        st.session_state.bed_choice = {
            "tier": tier_obj["tier"],
            "unit_id": None if unit_id == "Any available" else unit_id,
            "price": tier_obj["price"],
            "features": tier_obj["features"],
        }
        st.session_state.flow_step = "details"

# ------------------------------------------------------------------------------------
# PATIENT DETAILS (Idea 3)
# ------------------------------------------------------------------------------------
if st.session_state.flow_step == "details":
    st.subheader("👤 Patient & Stay Details")
    with st.form("patient_form"):
        name = st.text_input("Patient's Name", value="")
        attendant_name = st.text_input("Attendant's Name", value="")
        attendant_phone = st.text_input("Attendant's Phone Number", value="")
        patient_phone = st.text_input("Patient's Phone Number", value="")
        patient_age = st.number_input("Patient's Age", min_value=0, max_value=120, value=30, step=1)
        patient_email = st.text_input("Patient's Email Address", value="")
        patient_address = st.text_area("Patient's Address", value="")
        checkin_date = st.date_input("Date of Check-in", value=date.today())

        st.markdown("**Date of Check-out**")
        unknown = st.checkbox("Discharge date will be provided later (no fixed check-out yet)", value=True)
        checkout_date = None
        if not unknown:
            checkout_date = st.date_input("Select your check-out date", min_value=checkin_date)

        submitted = st.form_submit_button("Done")
        if submitted:
            st.session_state.patient_details = {
                "name": name,
                "attendant_name": attendant_name,
                "attendant_phone": attendant_phone,
                "patient_phone": patient_phone,
                "patient_age": patient_age,
                "patient_email": patient_email,
                "patient_address": patient_address,
                "checkin_date": checkin_date.strftime("%d %B %Y"),
                "checkout_mode": "unknown" if unknown else "date",
                "checkout_date": checkout_date.strftime("%d %B %Y") if checkout_date else "",
            }
            st.session_state.flow_step = "summary"

# ------------------------------------------------------------------------------------
# SUMMARY + COMBINED PDF (Idea 3)
# ------------------------------------------------------------------------------------
if st.session_state.flow_step == "summary":
    hospital_header = st.session_state.selected_hospital or "Doctigo Partner Hospital"
    st.success("✅ Details captured. Your combined summary is ready.")

    # Simple framed summary
    st.markdown("---")
    st.markdown(f"### 🏥 {hospital_header}")

    if st.session_state.appointment:
        ap = st.session_state.appointment
        st.markdown(f"**Doctor:** Dr. {ap['doctor']}  \n**Chamber:** {ap['chamber']}  \n**Slot:** {ap['slot']}  \n**Symptoms:** {ap['symptoms']}")
    else:
        st.info("No doctor appointment selected.")

    if st.session_state.bed_choice:
        bc = st.session_state.bed_choice
        st.markdown(f"**Bed/Cabin:** {bc['tier']} (₹{bc['price']}/night)  \n**Unit ID:** {bc.get('unit_id','Any')}  \n**Features:** {', '.join(bc['features'])}")
    else:
        st.info("No bed/cabin selected.")

    pdets = st.session_state.patient_details or {}
    check_out_text = "To be decided" if pdets.get('checkout_mode') == 'unknown' else pdets.get('checkout_date','')
    st.markdown(
        f"**Patient:** {pdets.get('name','')}  \n"
        f"**Attendant:** {pdets.get('attendant_name','')} ({pdets.get('attendant_phone','')})  \n"
        f"**Patient Phone:** {pdets.get('patient_phone','')}  \n"
        f"**Age:** {pdets.get('patient_age','')}  \n"
        f"**Email:** {pdets.get('patient_email','')}  \n"
        f"**Address:** {pdets.get('patient_address','')}  \n"
        f"**Check-in:** {pdets.get('checkin_date','')}  \n"
        f"**Check-out:** {check_out_text}"
    )

    # Download combined PDF
    pdf_buf = generate_full_pdf(
        hospital_name=hospital_header,
        patient=st.session_state.patient_details or {},
        appointment=st.session_state.appointment,
        bed_choice=st.session_state.bed_choice
    )
    st.download_button("⬇️ Download Combined PDF", data=pdf_buf, file_name="doctigo_booking_summary.pdf", mime="application/pdf")

    st.markdown("---")
    colA, colB = st.columns(2)
    with colA:
        if st.button("🏠 Home"):
            st.session_state.flow_step = "home"
            st.session_state.appointment = None
            st.session_state.bed_choice = None
            st.session_state.patient_details = None
            st.session_state.selected_hospital = None
    with colB:
        if st.button("🔁 Book Another Bed/Cabin"):
            st.session_state.flow_step = "beds"

# ------------------------------------------------------------------------------------
# ORIGINAL quick confirmation (kept) – generates appointment CSV/PDF on old flow
# ------------------------------------------------------------------------------------
if st.session_state.get("booked", False):
    st.success(f"✅ Appointment Booked with Dr. {st.session_state.booked_doctor} at {st.session_state.slot}")

    # Save appointment
    # NOTE: This block uses the older variables; kept for compatibility only.
    doctor_df_for_save = load_doctors_file()
    chamber_val = ""
    if not doctor_df_for_save.empty:
        m = doctor_df_for_save[doctor_df_for_save["Doctor Name"] == st.session_state.booked_doctor]
        if not m.empty:
            chamber_val = m.iloc[0]["Chamber"]

    appointment_df = pd.DataFrame([{
        "Patient Name": st.session_state.get("patient_name", ""),
        "Symptoms": '; '.join(st.session_state.symptoms_used),
        "Doctor": st.session_state.booked_doctor,
        "Chamber": chamber_val,
        "Slot": st.session_state.slot,
        "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }])

    if os.path.exists("appointments.csv"):
        appointment_df.to_csv("appointments.csv", mode="a", header=False, index=False)
    else:
        appointment_df.to_csv("appointments.csv", mode="w", header=True, index=False)

    # CSV Download
    csv_file = io.StringIO()
    csv_file.write("Doctor,Chamber,Slot,Symptoms\n")
    csv_file.write(f"{st.session_state.booked_doctor},{chamber_val},{st.session_state.slot},{'; '.join(st.session_state.symptoms_used)}\n")
    st.download_button("⬇️ Download CSV", data=csv_file.getvalue(), file_name="appointment.csv", mime="text/csv")

    # PDF Download (old format)
    pdf_buffer = generate_pdf_receipt(
        st.session_state.get("patient_name", ""),
        st.session_state.booked_doctor,
        chamber_val,
        st.session_state.slot,
        '; '.join(st.session_state.symptoms_used)
    )
    st.download_button("⬇️ Download PDF", data=pdf_buffer, file_name="appointment.pdf", mime="application/pdf")

# ------------------------------------------------------------------------------------
# Admin Dashboard (kept)
# ------------------------------------------------------------------------------------
ADMIN_PASSWORD = "admin123"  # change as needed
st.markdown("---")
st.header("🔒 Admin Dashboard Login")
admin_access = False

with st.expander("Login as Admin"):
    entered_password = st.text_input("Enter admin password", type="password", key="admin_pass_input")
    if st.button("Login", key="admin_login_btn"):
        if entered_password == ADMIN_PASSWORD:
            st.success("✅ Access granted.")
            st.session_state.admin_logged_in = True
        else:
            st.error("❌ Incorrect password")

if st.session_state.get("admin_logged_in", False):
    admin_access = True

if admin_access:
    st.header("📊 Admin Dashboard - All Appointments")

    uploaded_past_file = st.file_uploader("Upload Past Appointments CSV (Optional)", type=["csv"])
    if uploaded_past_file is not None:
        with open("data/appointments_past.csv", "wb") as f:
            f.write(uploaded_past_file.getbuffer())
        st.success("✅ Past appointment file saved successfully!")

    dfs = []

    if os.path.exists("data/appointments_past.csv"):
        past_df = pd.read_csv("data/appointments_past.csv")
        if "Slot" not in past_df.columns and "Visiting Time" in past_df.columns and "Appointment Date" in past_df.columns:
            past_df["Slot"] = past_df["Visiting Time"] + " on " + past_df["Appointment Date"]
        if "Doctor Name" in past_df.columns:
            past_df.rename(columns={"Doctor Name": "Doctor"}, inplace=True)
        dfs.append(past_df)

    if os.path.exists("appointments.csv"):
        current_df = pd.read_csv("appointments.csv")
        dfs.append(current_df)

    if dfs:
        df = pd.concat(dfs, ignore_index=True)

        total = len(df)
        st.subheader(f"📌 Appointments Done Till Now: **{total}**")

        doctor_filter = st.selectbox("Select Doctor:", ["All"] + sorted(df["Doctor"].unique().tolist()), key="admin_doctor_filter")
        filtered_df = df if doctor_filter == "All" else df[df["Doctor"] == doctor_filter]
        st.dataframe(filtered_df)

        patient_query = st.text_input("Search by Patient Name:", key="admin_patient_search")
        if patient_query:
            mask = filtered_df['Patient Name'].str.contains(patient_query, case=False)
            history = filtered_df[mask]
            st.dataframe(history)

        today = datetime.today()
        filtered_df['Parsed Date'] = pd.to_datetime(filtered_df['Slot'].str.extract(r'on (.+)$')[0], errors='coerce')
        upcoming = filtered_df[filtered_df['Parsed Date'].between(today, today + timedelta(days=3))]
        if not upcoming.empty:
            st.markdown("### ⏰ Upcoming Appointment Reminders")
            for _, row in upcoming.iterrows():
                if pd.notnull(row['Parsed Date']):
                    st.info(f"📅 {row['Parsed Date'].strftime('%d %b %Y')} - {row['Patient Name']} with Dr. {row['Doctor']}")

        st.download_button("⬇️ Export All Appointments CSV", filtered_df.to_csv(index=False).encode('utf-8'),
                           "appointments_admin.csv", mime="text/csv")

        st.markdown("### 📅 Calendar View")
        grouped = filtered_df.groupby(['Parsed Date', 'Doctor']).size().reset_index(name='Appointments')
        for date_, group in grouped.groupby('Parsed Date'):
            if pd.notnull(date_):
                st.markdown(f"#### {date_.strftime('%d %B %Y')}")
                for _, row in group.iterrows():
                    st.write(f"👨‍⚕️ {row['Doctor']} - {row['Appointments']} appointment(s)")
    else:
        st.info("No appointments to show yet.")
else:
    st.warning("🔐 Admin access required to view dashboard.")

st.markdown("---")
st.caption("Built with ❤️ by Doctigo AI Booking System")
