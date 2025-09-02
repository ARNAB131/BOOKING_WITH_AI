# app.py
import streamlit as st
import pandas as pd
import io
import os
import re
import math
import csv
import json
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
# Constants / Paths
# ------------------------------------------------------------------------------------
APPOINTMENTS_PATH = "appointments.csv"
INVENTORY_PATH = "beds_inventory.csv"
WAITLIST_PATH = "waitlist.csv"

# ------------------------------------------------------------------------------------
# Helpers (general)
# ------------------------------------------------------------------------------------
def _normalize_text(s: str) -> str:
    if not isinstance(s, str):
        return ""
    s = s.casefold()
    s = re.sub(r"[^a-z0-9\s]", " ", s)  # keep alnum + space
    s = re.sub(r"\s+", " ", s).strip()
    return s

def pdfsafe(s) -> str:
    """Make text safe for FPDF core fonts (latin-1)."""
    if s is None:
        return ""
    s = str(s)
    s = s.replace("₹", "Rs ")
    s = s.replace("–", "-").replace("—", "-")
    s = s.replace("•", "*")
    s = s.replace("“", '"').replace("”", '"').replace("’", "'").replace("‘", "'")
    return s.encode("latin-1", "replace").decode("latin-1")

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

# --- Appointment helpers (slot uniqueness + serial + distance/ETA) -------------------
def ensure_appointments_file():
    """Create appointments.csv with headers if missing."""
    if not os.path.exists(APPOINTMENTS_PATH):
        with open(APPOINTMENTS_PATH, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(["Patient Name","Symptoms","Doctor","Chamber","Slot","Timestamp","Date","SlotTime","Serial","DistanceKm","ETAmin"])

def parse_slot(slot_str: str):
    """
    'HH:MM ... on DD Month YYYY' -> ('HH:MM ...', 'DD Month YYYY')
    """
    if not isinstance(slot_str, str):
        return "", ""
    m = re.match(r"^(.*?)\s+on\s+(.*)$", slot_str.strip())
    if not m:
        return slot_str.strip(), ""
    return m.group(1).strip(), m.group(2).strip()

def load_appointments_df():
    ensure_appointments_file()
    try:
        df = pd.read_csv(APPOINTMENTS_PATH)
    except Exception:
        return pd.DataFrame(columns=["Patient Name","Symptoms","Doctor","Chamber","Slot","Timestamp","Date","SlotTime","Serial","DistanceKm","ETAmin"])
    # backfill Date/SlotTime if missing
    if "Date" not in df.columns or "SlotTime" not in df.columns:
        df["Date"] = ""
        df["SlotTime"] = ""
        for i, row in df.iterrows():
            stime, sdate = parse_slot(str(row.get("Slot","")))
            df.at[i, "Date"] = sdate
            df.at[i, "SlotTime"] = stime
    return df

def booked_slot_times_for(doctor_name: str, day: date) -> set:
    df = load_appointments_df()
    if df.empty:
        return set()
    day_str = day.strftime("%d %B %Y")
    sub = df[(df["Doctor"].astype(str) == str(doctor_name)) & (df["Date"] == day_str)]
    return set(sub["SlotTime"].astype(str).tolist())

def detect_hospital_for_chamber(chamber_val: str, hospitals_df: pd.DataFrame) -> str | None:
    if hospitals_df.empty or not chamber_val:
        return None
    ch = chamber_val.casefold()
    for _, r in hospitals_df.iterrows():
        h = str(r["Hospital"])
        if h and h.casefold() in ch:
            return h
    return None

def get_hospital_coords(hospital_name: str, hospitals_df: pd.DataFrame):
    if not hospital_name or hospitals_df.empty:
        return None
    m = hospitals_df[hospitals_df["Hospital"].astype(str).str.casefold() == hospital_name.casefold()]
    if m.empty:
        return None
    row = m.iloc[0]
    try:
        return float(row["Latitude"]), float(row["Longitude"])
    except Exception:
        return None

def travel_eta_minutes(distance_km: float) -> int:
    # Rule-of-thumb requested
    if distance_km <= 5:
        return 15
    if distance_km <= 10:
        return 20
    if distance_km <= 20:
        return 30
    return 45  # fallback for >20 km

def compute_distance_and_eta(user_loc: dict, hospital_name: str, hospitals_df: pd.DataFrame):
    if not user_loc or user_loc.get("lat") is None or user_loc.get("lon") is None:
        return None, None
    coords = get_hospital_coords(hospital_name, hospitals_df)
    if not coords:
        return None, None
    d = haversine_km(user_loc["lat"], user_loc["lon"], coords[0], coords[1])
    return d, travel_eta_minutes(d)

def compute_serial(slot_time: str, visiting_time_str: str) -> int:
    """Serial = position of slot_time in the day's generated slots order."""
    try:
        all_slots = generate_slots(visiting_time_str)
    except Exception:
        all_slots = []
    try:
        return all_slots.index(slot_time) + 1
    except ValueError:
        # if slot_time isn't in the known list, fallback to 1
        return 1

def save_appointment_row(patient_name, symptoms, doctor, chamber, slot_full, visiting_time_str, distance_km=None, eta_min=None):
    """
    Persist appointment ONLY if the (doctor, date, slot_time) is still free.
    Returns (ok: bool, msg: str, serial: int | None)
    """
    ensure_appointments_file()
    slot_time, date_str = parse_slot(slot_full)
    # guard
    if not doctor or not slot_time or not date_str:
        return False, "Invalid slot or doctor.", None

    # uniqueness check
    df = load_appointments_df()
    dup = df[(df["Doctor"].astype(str) == str(doctor)) & (df["Date"] == date_str) & (df["SlotTime"] == slot_time)]
    if not dup.empty:
        return False, "That time frame is already booked for this doctor and date. Please choose another slot.", None

    serial = compute_serial(slot_time, visiting_time_str)
    new_row = {
        "Patient Name": patient_name or "",
        "Symptoms": symptoms or "",
        "Doctor": doctor,
        "Chamber": chamber or "",
        "Slot": slot_full,
        "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Date": date_str,
        "SlotTime": slot_time,
        "Serial": serial,
        "DistanceKm": round(distance_km, 2) if isinstance(distance_km, (int, float)) else "",
        "ETAmin": int(eta_min) if isinstance(eta_min, (int, float)) else "",
    }
    # append
    with open(APPOINTMENTS_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(new_row.keys()))
        # write header only if file is empty
        if os.stat(APPOINTMENTS_PATH).st_size == 0:
            writer.writeheader()
        writer.writerow(new_row)

    return True, "✅ Appointment booked and saved.", serial

# --- Original appointment PDF (kept, now unicode-safe) ---
def generate_pdf_receipt(patient_name, doctor, chamber, slot, symptoms):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", "B", 18)
    pdf.cell(0, 10, pdfsafe("Doctigo"), ln=True, align='C')
    pdf.set_font("Arial", "B", 12)
    pdf.ln(5)
    pdf.cell(0, 10, pdfsafe("Patient Appointment Receipt"), ln=True, align='C')
    pdf.set_font("Arial", "", 11)
    pdf.cell(0, 5, pdfsafe("----------------------------------------"), ln=True, align='C')
    pdf.ln(5)
    pdf.set_font("Arial", "", 12)
    pdf.cell(0, 8, pdfsafe(f"Patient Name: {patient_name}"), ln=True)
    pdf.cell(0, 8, pdfsafe(f"Doctor: Dr. {doctor}"), ln=True)
    pdf.cell(0, 8, pdfsafe(f"Chamber: {chamber}"), ln=True)
    pdf.cell(0, 8, pdfsafe(f"Slot: {slot}"), ln=True)
    pdf.multi_cell(0, 8, pdfsafe(f"Symptoms: {symptoms}"))
    pdf.cell(0, 8, pdfsafe(f"Issued On: {datetime.now().strftime('%d %B %Y, %I:%M %p')}"), ln=True)
    pdf.ln(10)
    pdf.set_font("Courier", "B", 11)
    pdf.set_text_color(150)
    pdf.cell(0, 10, pdfsafe("--- Tear Here ---"), ln=True, align='C')
    pdf.set_text_color(0)
    pdf.ln(5)
    pdf.set_font("Arial", "", 12)
    pdf.cell(0, 8, pdfsafe("Hospital Copy"), ln=True)
    pdf.cell(0, 8, pdfsafe(f"Patient Name: {patient_name}"), ln=True)
    pdf.cell(0, 8, pdfsafe(f"Doctor: Dr. {doctor}"), ln=True)
    pdf.cell(0, 8, pdfsafe(f"Chamber: {chamber}"), ln=True)
    pdf.cell(0, 8, pdfsafe(f"Slot: {slot}"), ln=True)
    pdf.multi_cell(0, 8, pdfsafe(f"Symptoms: {symptoms}"))
    pdf.cell(0, 8, pdfsafe(f"Issued On: {datetime.now().strftime('%d %B %Y, %I:%M %p')}"), ln=True)
    pdf.ln(10)
    pdf.set_font("Arial", "I", 9)
    pdf.set_text_color(120)
    pdf.cell(0, 10, pdfsafe("This receipt is auto-generated by Doctigo AI Booking System."), ln=True, align="C")
    pdf.set_text_color(0)
    pdf_bytes = pdf.output(dest="S").encode("latin-1")
    return io.BytesIO(pdf_bytes)

# --- NEW: combined PDF for appointment + bed/cabin (unicode-safe) ---
def generate_full_pdf(hospital_name, patient, appointment, bed_choice):
    """
    hospital_name: str
    patient: dict -> name, attendant_name, attendant_phone, patient_phone, patient_age,
                     patient_email, patient_address, checkin_date, checkout_mode, checkout_date
    appointment: dict or None -> doctor, chamber, slot, symptoms, serial, distance_km, eta_min
    bed_choice: dict or None -> tier, unit_id, price, features(list)
    """
    pdf = FPDF()
    pdf.add_page()

    # Header
    pdf.set_font("Arial", "B", 18)
    pdf.cell(0, 10, pdfsafe(hospital_name or "Doctigo"), ln=True, align='C')

    pdf.set_font("Arial", "B", 12)
    pdf.ln(3)
    pdf.cell(0, 8, pdfsafe("Booking Summary"), ln=True, align='C')
    pdf.set_font("Arial", "", 11)
    pdf.cell(0, 5, pdfsafe("----------------------------------------"), ln=True, align='C')

    # Patient Details
    pdf.ln(4)
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 8, pdfsafe("Patient Details"), ln=True)
    pdf.set_font("Arial", "", 11)

    checkout_str = "To be decided (post-discharge)" if patient.get("checkout_mode") == "unknown" else patient.get("checkout_date", "")
    details = f"""
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
""".strip()
    pdf.multi_cell(0, 7, pdfsafe(details))

    # Appointment
    if appointment:
        pdf.ln(3)
        pdf.set_font("Arial", "B", 12)
        pdf.cell(0, 8, pdfsafe("Doctor Appointment"), ln=True)
        pdf.set_font("Arial", "", 11)
        serial_text = f"\nSerial #: {appointment.get('serial','')}" if appointment.get("serial") else ""
        dist_text = ""
        if appointment.get("distance_km") is not None and appointment.get("eta_min") is not None:
            dist_text = f"\nDistance: {appointment['distance_km']:.2f} km | ETA: {appointment['eta_min']} min"
        ap_text = f"""
Doctor: Dr. {appointment.get('doctor','')}
Chamber: {appointment.get('chamber','')}
Slot: {appointment.get('slot','')}{serial_text}{dist_text}
Symptoms: {appointment.get('symptoms','')}
""".strip()
        pdf.multi_cell(0, 7, pdfsafe(ap_text))

    # Bed/Cabin
    if bed_choice:
        pdf.ln(3)
        pdf.set_font("Arial", "B", 12)
        pdf.cell(0, 8, pdfsafe("Bed/Cabin Booking"), ln=True)
        pdf.set_font("Arial", "", 11)
        features_text = ", ".join(bed_choice.get('features', []))
        bed_text = f"""
Type: {bed_choice.get('tier','')}
Unit ID: {bed_choice.get('unit_id','Any')}
Price per night: Rs {bed_choice.get('price','')}
Features: {features_text}
""".strip()
        pdf.multi_cell(0, 7, pdfsafe(bed_text))

    pdf.ln(6)
    pdf.set_font("Arial", "I", 9)
    pdf.set_text_color(120)
    pdf.cell(0, 8, pdfsafe("This receipt is auto-generated by Doctigo AI System."), ln=True, align="C")
    pdf.set_text_color(0)

    pdf_bytes = pdf.output(dest="S").encode("latin-1")
    return io.BytesIO(pdf_bytes)

# ------------------------------------------------------------------------------------
# Helpers (bed inventory with per-date availability)
# ------------------------------------------------------------------------------------
def _tier_ids():
    return {
        "General Bed":   [f"G-{i}" for i in range(1, 41)],
        "General Cabin": [f"C-{i}" for i in range(1, 21)],
        "VIP Cabin":     [f"V-{i}" for i in range(1, 11)],
    }

def ensure_inventory():
    if not os.path.exists(INVENTORY_PATH):
        with open(INVENTORY_PATH, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(["Hospital", "Tier", "UnitID", "Date", "Status"])  # Status: available|booked

def _ensure_rows_for(hospital: str, tier: str, day: date):
    """Make sure rows exist for this hospital/tier/day with Status=available."""
    ensure_inventory()
    df = pd.read_csv(INVENTORY_PATH)
    ids = _tier_ids()[tier]
    day_str = day.strftime("%Y-%m-%d")

    mask = (df["Hospital"].astype(str) == hospital) & (df["Tier"] == tier) & (df["Date"] == day_str)
    existing = set(df.loc[mask, "UnitID"].astype(str).tolist())
    missing = [uid for uid in ids if uid not in existing]

    if missing:
        add = pd.DataFrame({
            "Hospital": [hospital]*len(missing),
            "Tier": [tier]*len(missing),
            "UnitID": missing,
            "Date": [day_str]*len(missing),
            "Status": ["available"]*len(missing),
        })
        out = pd.concat([df, add], ignore_index=True)
        out.to_csv(INVENTORY_PATH, index=False)

def get_inventory(hospital: str, tier: str, day: date) -> pd.DataFrame:
    ensure_inventory()
    _ensure_rows_for(hospital, tier, day)
    df = pd.read_csv(INVENTORY_PATH)
    return df[(df["Hospital"].astype(str) == hospital) & (df["Tier"] == tier) & (df["Date"] == day.strftime("%Y-%m-%d"))].copy()

def mark_booked(hospital: str, tier: str, unit_id: str, day: date):
    ensure_inventory()
    _ensure_rows_for(hospital, tier, day)
    df = pd.read_csv(INVENTORY_PATH)
    mask = (
        (df["Hospital"].astype(str) == hospital)
        & (df["Tier"] == tier)
        & (df["UnitID"] == unit_id)
        & (df["Date"] == day.strftime("%Y-%m-%d"))
    )
    if mask.any():
        df.loc[mask, "Status"] = "booked"
        df.to_csv(INVENTORY_PATH, index=False)

def mark_booked_range(hospital: str, tier: str, unit_id: str, start_day: date, end_day: date | None):
    """Book unit for each day in [start_day, end_day]. If end_day is None, only start_day."""
    last = end_day or start_day
    cur = start_day
    while cur <= last:
        mark_booked(hospital, tier, unit_id, cur)
        cur += timedelta(days=1)

def reset_inventory_for_date(hospital: str, tier: str, day: date):
    ensure_inventory()
    _ensure_rows_for(hospital, tier, day)
    df = pd.read_csv(INVENTORY_PATH)
    mask = (df["Hospital"].astype(str) == hospital) & (df["Tier"] == tier) & (df["Date"] == day.strftime("%Y-%m-%d"))
    df.loc[mask, "Status"] = "available"
    df.to_csv(INVENTORY_PATH, index=False)

# ------------------------------------------------------------------------------------
# --- IDEA 5: Waitlist helpers (CSV-backed) ------------------------------------------
# ------------------------------------------------------------------------------------
def ensure_waitlist():
    if not os.path.exists(WAITLIST_PATH):
        with open(WAITLIST_PATH, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([
                "Timestamp", "Hospital", "Tier", "Date", "Status", "Patient", "Phone", "AssignedUnit"
            ])  # Status: waiting | assigned

def load_waitlist() -> pd.DataFrame:
    ensure_waitlist()
    try:
        return pd.read_csv(WAITLIST_PATH)
    except Exception:
        return pd.DataFrame(columns=["Timestamp","Hospital","Tier","Date","Status","Patient","Phone","AssignedUnit"])

def save_waitlist(df: pd.DataFrame):
    df.to_csv(WAITLIST_PATH, index=False)

def add_to_waitlist(hospital: str, tier: str, day: date, patient: str = "", phone: str = ""):
    ensure_waitlist()
    df = load_waitlist()
    new_row = {
        "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Hospital": hospital,
        "Tier": tier,
        "Date": day.strftime("%Y-%m-%d"),
        "Status": "waiting",
        "Patient": patient or "",
        "Phone": phone or "",
        "AssignedUnit": ""
    }
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    save_waitlist(df)

def auto_match_waitlist():
    """
    Try to fulfill each waiting request by assigning the first available unit for that date/tier/hospital.
    Returns a tuple (#assigned, list of (idx, assigned_unit)).
    """
    df = load_waitlist()
    assigned = 0
    changes = []

    if df.empty:
        return assigned, changes

    for idx, row in df[df["Status"] == "waiting"].sort_values("Timestamp").iterrows():
        hosp = str(row["Hospital"])
        tier = str(row["Tier"])
        try:
            day = datetime.strptime(str(row["Date"]), "%Y-%m-%d").date()
        except Exception:
            continue

        inv = get_inventory(hosp, tier, day)
        avail = inv[inv["Status"] != "booked"]["UnitID"].astype(str).tolist()
        if not avail:
            continue

        unit_id = sorted(avail, key=lambda x: (len(x), x))[0]
        mark_booked(hosp, tier, unit_id, day)

        df.at[idx, "Status"] = "assigned"
        df.at[idx, "AssignedUnit"] = unit_id
        assigned += 1
        changes.append((idx, unit_id))

    if assigned > 0:
        save_waitlist(df)

    return assigned, changes

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
    ("seat_selected", ""),
    ("beds_avail_day", date.today()),
    ("selected_beds_day", None),
]:
    if key not in st.session_state:
        st.session_state[key] = default

# ------------------------------------------------------------------------------------
# Bulk Appointment Upload (now respects slot-uniqueness)
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
                doctors_df = load_doctors_file()
                for _, row in df_bulk.iterrows():
                    name = row.get("Patient Name","")
                    symptoms_list = [s.strip() for s in str(row.get("Symptoms","")).split(',') if s.strip()]
                    msg, recs = recommend_doctors(symptoms_list)
                    if not recs:
                        continue
                    selected = recs[0]
                    doctor_name = selected['Doctor']
                    chamber_val = selected.get('Chamber', '')
                    visiting_time = selected.get('Visiting Time', '')
                    today_date = datetime.today().date()

                    # compute available slot for today
                    all_slots = selected.get('Slots') or generate_slots(visiting_time)
                    booked = booked_slot_times_for(doctor_name, today_date)
                    avail = [s for s in all_slots if s not in booked]
                    if not avail:
                        # skip if nothing free today (simple policy)
                        continue
                    slot_time = avail[0]
                    full_slot = f"{slot_time} on {today_date.strftime('%d %B %Y')}"

                    # Distance/ETA optional (needs hospital + user location)
                    hospitals_df = load_hospitals()
                    hosp_name = detect_hospital_for_chamber(chamber_val, hospitals_df) or st.session_state.get("selected_hospital")
                    dist_km, eta_min = compute_distance_and_eta(st.session_state.get("user_location"), hosp_name, hospitals_df)

                    ok, msg, serial = save_appointment_row(
                        patient_name=name,
                        symptoms='; '.join(symptoms_list),
                        doctor=doctor_name,
                        chamber=chamber_val,
                        slot_full=full_slot,
                        visiting_time_str=visiting_time,
                        distance_km=dist_km,
                        eta_min=eta_min
                    )
                st.success("✅ Auto-booking finished (unique per time frame per day).")
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
# EMERGENCY → Nearest Hospitals + quick actions
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

        # Quick actions
        try:
            nearest = hospitals_df.iloc[0]
            dest_lat = float(nearest["Latitude"])
            dest_lon = float(nearest["Longitude"])
            maps_url = f"https://www.google.com/maps/dir/?api=1&origin={lat0},{lon0}&destination={dest_lat},{dest_lon}"
            share_my_loc = f"https://maps.google.com/?q={lat0},{lon0}"
            st.markdown("---")
            st.markdown("### 🚀 Quick Actions")
            st.markdown(f"- 🧭 **Navigation:** [Open Google Maps to {nearest['Hospital']}]({maps_url})")
            st.markdown(f"- 📍 **Share my current location:** {share_my_loc}")

            copy_html = """
            <button id="copy" style="margin-top:6px;padding:6px 10px;border:1px solid #2a9d8f;border-radius:8px;background:#fff;color:#2a9d8f;cursor:pointer;">Copy share link</button>
            <script>
              const link = "__LINK__";
              const btn = document.getElementById('copy');
              if (btn && navigator.clipboard) {
                btn.addEventListener('click', function(){
                  navigator.clipboard.writeText(link).then(function(){
                    btn.textContent = "Copied!";
                    setTimeout(function(){ btn.textContent = "Copy share link"; }, 1200);
                  });
                });
              }
            </script>
            """
            components.html(copy_html.replace("__LINK__", share_my_loc), height=50)

            st.markdown("- ☎️ **Call Ambulance:** [Tap to call 108](tel:108)")
        except Exception:
            pass

    else:
        st.info("Enter Latitude/Longitude (or use Detect My Location).")

# ------------------------------------------------------------------------------------
# HOSPITAL/DOCTORS PAGE (with slot-uniqueness + Serial/Distance/ETA)
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
                return.
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

    # --- Direct booking UI (NOW: pick date first -> available slots only) ---
    st.markdown("---")
    st.subheader("📋 Book a Doctor")
    doctor_names = filtered_doctors['Doctor Name'].unique().tolist()
    selected_doctor = st.selectbox("Choose a doctor (optional):", ["None"] + doctor_names)

    direct_slot = ""
    selected_date = None
    serial_for_ui = None
    if selected_doctor != "None":
        selected_info = filtered_doctors[filtered_doctors['Doctor Name'] == selected_doctor].iloc[0]
        st.markdown(f"**Specialization:** {selected_info['Specialization']}")
        st.markdown(f"**Chamber:** {selected_info['Chamber']}")
        visiting_time_str = selected_info['Visiting Time']

        today = datetime.today()
        selected_date = st.date_input("Choose appointment date:", min_value=today)
        # compute available slots for that doctor/date
        all_slots = generate_slots(visiting_time_str)
        already = booked_slot_times_for(selected_doctor, selected_date.date())
        available_slots = [s for s in all_slots if s not in already]
        if not available_slots:
            st.warning("All time frames are booked for this doctor on the selected date.")
        else:
            direct_slot = st.selectbox("Select a slot (available only):", available_slots, key="direct_slot")

            # Show Serial vs Distance vs Time
            if direct_slot:
                serial_for_ui = compute_serial(direct_slot, visiting_time_str)
                # find hospital for distance
                hospitals_df_local = load_hospitals()
                hospital_for_distance = chosen_hospital or detect_hospital_for_chamber(selected_info["Chamber"], hospitals_df_local)
                dist_km, eta_min = compute_distance_and_eta(st.session_state.get("user_location"), hospital_for_distance, hospitals_df_local)
                if dist_km is not None:
                    st.info(f"**Serial #{serial_for_ui}**  |  **Distance:** {dist_km:.2f} km  |  **ETA:** {eta_min} min (≤5km→15m, ≤10km→20m, ≤20km→30m)")
                else:
                    st.info(f"**Serial #{serial_for_ui}**  |  **ETA rule:** ≤5km→15m, ≤10km→20m, ≤20km→30m (enable location to compute distance)")

            if st.button(f"📅 Book Appointment with Dr. {selected_doctor}", key="direct_book"):
                if not direct_slot:
                    st.error("Please select an available slot.")
                else:
                    # Re-check & persist to CSV (enforce uniqueness)
                    slot_full = f"{direct_slot} on {selected_date.strftime('%d %B %Y')}"
                    hospitals_df_local = load_hospitals()
                    hospital_for_distance = chosen_hospital or detect_hospital_for_chamber(selected_info["Chamber"], hospitals_df_local)
                    dist_km, eta_min = compute_distance_and_eta(st.session_state.get("user_location"), hospital_for_distance, hospitals_df_local)
                    ok, msg, serial_num = save_appointment_row(
                        patient_name=st.session_state.get("patient_name",""),
                        symptoms='; '.join(all_symptoms or ["Direct Booking"]),
                        doctor=selected_doctor,
                        chamber=selected_info["Chamber"],
                        slot_full=slot_full,
                        visiting_time_str=visiting_time_str,
                        distance_km=dist_km,
                        eta_min=eta_min
                    )
                    if ok:
                        st.session_state.appointment = {
                            "doctor": selected_doctor,
                            "chamber": selected_info["Chamber"],
                            "slot": slot_full,
                            "symptoms": "; ".join(all_symptoms or ["Direct Booking"]),
                            "serial": serial_num,
                            "distance_km": dist_km,
                            "eta_min": eta_min
                        }
                        st.success(f"✅ Appointment booked. Serial #{serial_num}.")
                    else:
                        st.error(msg)

    # --- AI recommendations (respect slot uniqueness; date first) ---
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
        hospitals_df_global = load_hospitals()
        for idx, doc in enumerate(st.session_state.recommendations):
            with st.expander(f"{idx+1}. Dr. {doc['Doctor']} - {doc['Specialization']}"):
                st.markdown(f"**Chamber:** {doc['Chamber']}")
                # date first
                appt_date = st.date_input(f"Choose date for Dr. {doc['Doctor']}", key=f"date_{idx}", min_value=datetime.today())
                # available slots only
                visiting_time_str = doc.get("Visiting Time", "")
                all_slots = doc.get("Slots") or generate_slots(visiting_time_str)
                already = booked_slot_times_for(doc["Doctor"], appt_date.date())
                avail = [s for s in all_slots if s not in already]
                if not avail:
                    st.warning("All time frames are booked for this doctor on the selected date.")
                    continue
                slot = st.selectbox(f"Select a slot for Dr. {doc['Doctor']}", options=avail, key=f"slot_{idx}")

                # Serial vs distance vs time
                serial_num = compute_serial(slot, visiting_time_str)
                hospital_for_distance = chosen_hospital or detect_hospital_for_chamber(doc["Chamber"], hospitals_df_global)
                dist_km, eta_min = compute_distance_and_eta(st.session_state.get("user_location"), hospital_for_distance, hospitals_df_global)
                if dist_km is not None:
                    st.info(f"**Serial #{serial_num}**  |  **Distance:** {dist_km:.2f} km  |  **ETA:** {eta_min} min (≤5km→15m, ≤10km→20m, ≤20km→30m)")
                else:
                    st.info(f"**Serial #{serial_num}**  |  **ETA rule:** ≤5km→15m, ≤10km→20m, ≤20km→30m (enable location to compute distance)")

                if st.button(f"Book Appointment with Dr. {doc['Doctor']}", key=f"book_{idx}"):
                    slot_full = f"{slot} on {appt_date.strftime('%d %B %Y')}"
                    ok, msg, serial_saved = save_appointment_row(
                        patient_name=st.session_state.get("patient_name",""),
                        symptoms='; '.join(all_symptoms),
                        doctor=doc["Doctor"],
                        chamber=doc["Chamber"],
                        slot_full=slot_full,
                        visiting_time_str=visiting_time_str,
                        distance_km=dist_km,
                        eta_min=eta_min
                    )
                    if ok:
                        st.session_state.appointment = {
                            "doctor": doc["Doctor"],
                            "chamber": doc["Chamber"],
                            "slot": slot_full,
                            "symptoms": "; ".join(all_symptoms),
                            "serial": serial_saved,
                            "distance_km": dist_km,
                            "eta_min": eta_min
                        }
                        st.success(f"✅ Appointment booked. Serial #{serial_saved}.")
                    else:
                        st.error(msg)

    # Book Beds/Cabins
    st.markdown("---")
    if st.button("🛏️ Book Beds/Cabins"):
        if not st.session_state.selected_hospital and chosen_hospital:
            st.session_state.selected_hospital = chosen_hospital
        st.session_state.flow_step = "beds"

# ------------------------------------------------------------------------------------
# BEDS/CABINS — Checkbox grid (no typing)
# ------------------------------------------------------------------------------------
if st.session_state.flow_step == "beds":
    hospital_name_for_beds = st.session_state.selected_hospital or "Doctigo Partner Hospital"
    st.subheader(f"🛏️ Beds & Cabins – {hospital_name_for_beds}")
    st.caption("Pick a date, choose a type, then tick **one** box to select your unit. (▭ Sold = unavailable)")

    beds_day = st.date_input(
        "Availability date (usually check-in):",
        value=st.session_state.get("beds_avail_day", date.today()),
        key="beds_avail_day"
    )
    day_str = beds_day.strftime("%Y-%m-%d")

    tiers = [
        {"tier": "General Bed",   "price": 100,  "features": ["1 bed","1 chair","bed table"], "ids": [f"G-{i}" for i in range(1,41)], "cols": 10},
        {"tier": "General Cabin", "price": 1000, "features": ["2 beds","attached washroom","bed table","chair","food x3 times"], "ids": [f"C-{i}" for i in range(1,21)], "cols": 5},
        {"tier": "VIP Cabin",     "price": 4000, "features": ["premium bed x2","sofa","Air Conditioning","attached washroom","TV","fridge","bed table x2","coffee table","2 chairs"], "ids": [f"V-{i}" for i in range(1,11)], "cols": 5},
    ]
    tier_names = [t["tier"] for t in tiers]
    pick_tier = st.radio("Select type", options=tier_names)
    tier_obj = next(t for t in tiers if t["tier"] == pick_tier)

    inv = get_inventory(hospital_name_for_beds, tier_obj["tier"], beds_day)
    sold = set(inv[inv["Status"] == "booked"]["UnitID"].astype(str).tolist())

    with st.expander(f"ℹ️ What is included in {pick_tier}?"):
        st.markdown(f"**₹{tier_obj['price']} per night**")
        st.markdown("- " + "\n- ".join(tier_obj["features"]))

    # Checkbox grid
    ids = tier_obj["ids"]
    cols = tier_obj["cols"]

    if st.session_state.get("seat_selected") not in ids:
        st.session_state.seat_selected = ""

    key_prefix = f"unit_{pick_tier}_{day_str}_"

    def _select_unit(sel_uid, id_list, prefix):
        st.session_state.seat_selected = sel_uid
        for u in id_list:
            st.session_state[f"{prefix}{u}"] = (u == sel_uid)

    total = len(ids)
    rows = math.ceil(total / cols)
    for r in range(rows):
        row_ids = ids[r*cols:(r+1)*cols]
        row_cols = st.columns(len(row_ids))
        for c, uid in enumerate(row_ids):
            disabled = uid in sold
            label = f"{uid}" + (" ▭" if disabled else "")
            cb_key = f"{key_prefix}{uid}"
            row_cols[c].checkbox(
                label,
                key=cb_key,
                value=(st.session_state.get("seat_selected") == uid and not disabled),
                disabled=disabled,
                on_change=_select_unit,
                args=(uid, ids, key_prefix),
                help="Tick to select this unit" if not disabled else "Already booked"
            )

    available_units = [uid for uid in ids if uid not in sold]

    st.markdown("#### Your selection")
    selected_unit = st.session_state.get("seat_selected", "")
    if selected_unit:
        st.success(f"Selected: **{selected_unit}** for **{beds_day.strftime('%d %b %Y')}**")
    else:
        st.info("No unit selected yet. Tick any available box above.")

    if st.button("Next ➜", disabled=not bool(selected_unit)):
        st.session_state.bed_choice = {
            "tier": tier_obj["tier"],
            "unit_id": selected_unit,
            "price": tier_obj["price"],
            "features": tier_obj["features"],
        }
        st.session_state.selected_beds_day = beds_day
        st.session_state.flow_step = "details"

    st.markdown("---")
    st.markdown("### 🛎️ Waitlist (if fully booked)")
    if len(available_units) == 0:
        st.warning("All units in this type are currently **sold out** for the selected date.")
        with st.form("waitlist_form_blocked"):
            wl_name = st.text_input("Patient Name (optional)", value=st.session_state.get("patient_name",""))
            wl_phone = st.text_input("Phone (optional)")
            submitted = st.form_submit_button("🔔 Join Waitlist for this date & type")
            if submitted:
                add_to_waitlist(hospital_name_for_beds, pick_tier, beds_day, wl_name, wl_phone)
                st.success("You're on the waitlist. We'll try to auto-assign a unit when one is freed.")
    else:
        with st.expander("Prefer to wait for a different unit? (Optional)"):
            with st.form("waitlist_form_optional"):
                wl_name2 = st.text_input("Patient Name (optional)", value=st.session_state.get("patient_name",""))
                wl_phone2 = st.text_input("Phone (optional)")
                submitted2 = st.form_submit_button("🔔 Join Waitlist anyway")
                if submitted2:
                    add_to_waitlist(hospital_name_for_beds, pick_tier, beds_day, wl_name2, wl_phone2)
                    st.success("Added to waitlist. If additional units become available, we'll auto-assign in Admin > Waitlist.")

# ------------------------------------------------------------------------------------
# PATIENT DETAILS
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
# SUMMARY + COMBINED PDF
# ------------------------------------------------------------------------------------
if st.session_state.flow_step == "summary":
    hospital_header = st.session_state.selected_hospital or "Doctigo Partner Hospital"
    st.success("✅ Details captured. Your combined summary is ready.")

    st.markdown("---")
    st.markdown(f"### 🏥 {hospital_header}")

    if st.session_state.appointment:
        ap = st.session_state.appointment
        extra = ""
        if ap.get("serial"):
            extra += f"  \n**Serial #:** {ap['serial']}"
        if ap.get("distance_km") is not None and ap.get("eta_min") is not None:
            extra += f"  \n**Distance/ETA:** {ap['distance_km']:.2f} km / {ap['eta_min']} min"
        st.markdown(f"**Doctor:** Dr. {ap['doctor']}  \n**Chamber:** {ap['chamber']}  \n**Slot:** {ap['slot']}  \n**Symptoms:** {ap.get('symptoms','')}{extra}")
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

    pdf_buf = generate_full_pdf(
        hospital_name=hospital_header,
        patient=st.session_state.patient_details or {},
        appointment=st.session_state.appointment,
        bed_choice=st.session_state.bed_choice
    )
    st.download_button("⬇️ Download Combined PDF", data=pdf_buf, file_name="doctigo_booking_summary.pdf", mime="application/pdf")

    # Reserve bed/cabin (per date or range)
    if st.session_state.bed_choice and st.session_state.bed_choice.get("unit_id"):
        start_day = datetime.strptime(st.session_state.patient_details.get("checkin_date"), "%d %B %Y").date()
        if st.session_state.patient_details.get("checkout_mode") == "date" and st.session_state.patient_details.get("checkout_date"):
            end_day = datetime.strptime(st.session_state.patient_details["checkout_date"], "%d %B %Y").date()
        else:
            end_day = start_day

        if st.button("✅ Confirm & Reserve Bed/Cabin"):
            mark_booked_range(
                hospital=st.session_state.selected_hospital or "Doctigo Partner Hospital",
                tier=st.session_state.bed_choice["tier"],
                unit_id=st.session_state.bed_choice["unit_id"],
                start_day=start_day,
                end_day=end_day
            )
            st.success("Bed/Cabin reserved. Inventory updated for the selected date range.")

    st.markdown("---")
    colA, colB = st.columns(2)
    with colA:
        if st.button("🏠 Home"):
            st.session_state.flow_step = "home"
            st.session_state.appointment = None
            st.session_state.bed_choice = None
            st.session_state.patient_details = None
            st.session_state.selected_hospital = None
            st.session_state.seat_selected = ""
    with colB:
        if st.button("🔁 Book Another Bed/Cabin"):
            st.session_state.flow_step = "beds"

# ------------------------------------------------------------------------------------
# ORIGINAL quick confirmation (legacy path)
# ------------------------------------------------------------------------------------
if st.session_state.get("booked", False):
    st.success(f"✅ Appointment Booked with Dr. {st.session_state.booked_doctor} at {st.session_state.slot}")

    # Save appointment (legacy) — this path may double-book; prefer the new flows above.
    doctor_df_for_save = load_doctors_file()
    chamber_val = ""
    visiting_time_guess = ""
    if not doctor_df_for_save.empty:
        m = doctor_df_for_save[doctor_df_for_save["Doctor Name"] == st.session_state.booked_doctor]
        if not m.empty:
            chamber_val = m.iloc[0]["Chamber"]
            visiting_time_guess = m.iloc[0]["Visiting Time"]

    # Attempt to save via the safe helper
    ok, msg, serial_num = save_appointment_row(
        patient_name=st.session_state.get("patient_name", ""),
        symptoms='; '.join(st.session_state.symptoms_used),
        doctor=st.session_state.booked_doctor,
        chamber=chamber_val,
        slot_full=st.session_state.slot,
        visiting_time_str=visiting_time_guess
    )
    if not ok:
        st.warning(msg)

    # CSV quick download (legacy)
    csv_file = io.StringIO()
    csv_file.write("Doctor,Chamber,Slot,Symptoms\n")
    csv_file.write(f"{st.session_state.booked_doctor},{chamber_val},{st.session_state.slot},{'; '.join(st.session_state.symptoms_used)}\n")
    st.download_button("⬇️ Download CSV", data=csv_file.getvalue(), file_name="appointment.csv", mime="text/csv")

    # PDF quick download (legacy)
    pdf_buffer = generate_pdf_receipt(
        st.session_state.get("patient_name", ""),
        st.session_state.booked_doctor,
        chamber_val,
        st.session_state.slot,
        '; '.join(st.session_state.symptoms_used)
    )
    st.download_button("⬇️ Download PDF", data=pdf_buffer, file_name="appointment.pdf", mime="application/pdf")

# ------------------------------------------------------------------------------------
# Admin Dashboard + Bed Inventory Tools + Waitlist tools
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

    if os.path.exists(APPOINTMENTS_PATH):
        current_df = pd.read_csv(APPOINTMENTS_PATH)
        dfs.append(current_df)

    if dfs:
        df = pd.concat(dfs, ignore_index=True)

        total = len(df)
        st.subheader(f"📌 Appointments Done Till Now: **{total}**")

        doctor_filter = st.selectbox("Select Doctor:", ["All"] + sorted(df["Doctor"].dropna().unique().tolist()), key="admin_doctor_filter")
        filtered_df = df if doctor_filter == "All" else df[df["Doctor"] == doctor_filter]
        st.dataframe(filtered_df)

        patient_query = st.text_input("Search by Patient Name:", key="admin_patient_search")
        if patient_query:
            mask = filtered_df['Patient Name'].astype(str).str.contains(patient_query, case=False, na=False)
            history = filtered_df[mask]
            st.dataframe(history)

        today = datetime.today()
        if "Date" in filtered_df.columns:
            try:
                parsed_dates = pd.to_datetime(filtered_df["Date"], errors="coerce")
            except Exception:
                parsed_dates = pd.Series([pd.NaT]*len(filtered_df))
            filtered_df = filtered_df.assign(ParsedDate=parsed_dates)
            upcoming = filtered_df[filtered_df['ParsedDate'].between(today, today + timedelta(days=3), inclusive="both")]
        else:
            # fallback for legacy rows
            filtered_df['Parsed Date'] = pd.to_datetime(filtered_df['Slot'].str.extract(r'on (.+)$')[0], errors='coerce')
            upcoming = filtered_df[filtered_df['Parsed Date'].between(today, today + timedelta(days=3))]

        if not upcoming.empty:
            st.markdown("### ⏰ Upcoming Appointment Reminders")
            date_col = "ParsedDate" if "ParsedDate" in upcoming.columns else "Parsed Date"
            for _, row in upcoming.iterrows():
                if pd.notnull(row[date_col]):
                    st.info(f"📅 {row[date_col].strftime('%d %b %Y')} - {row.get('Patient Name','(Name)')} with Dr. {row['Doctor']} (Serial #{row.get('Serial','-')})")

        st.download_button("⬇️ Export All Appointments CSV", filtered_df.to_csv(index=False).encode('utf-8'),
                           "appointments_admin.csv", mime="text/csv")

        st.markdown("### 📅 Calendar View")
        if "ParsedDate" in filtered_df.columns:
            grouped = filtered_df.dropna(subset=["ParsedDate"]).groupby(['ParsedDate', 'Doctor']).size().reset_index(name='Appointments')
            for date_, group in grouped.groupby('ParsedDate'):
                st.markdown(f"#### {date_.strftime('%d %B %Y')}")
                for _, row in group.iterrows():
                    st.write(f"👨‍⚕️ {row['Doctor']} - {row['Appointments']} appointment(s)")
        else:
            st.info("Calendar view available once date parsing completes.")

        # ---------- Bed Inventory Tools ----------
        st.markdown("### 🧰 Bed Inventory Tools")
        hospitals_df = load_hospitals()
        h_options = hospitals_df["Hospital"].tolist() if not hospitals_df.empty else []
        sel_hosp = st.selectbox("Hospital", options=h_options) if h_options else st.text_input("Hospital (type)")
        sel_tier = st.selectbox("Tier", options=["General Bed", "General Cabin", "VIP Cabin"])
        sel_day = st.date_input("Date to reset", value=date.today())

        if st.button("♻️ Reset availability for that date"):
            reset_inventory_for_date(sel_hosp, sel_tier, sel_day)
            st.success("Availability reset to ALL AVAILABLE for that date/tier.")

        # ---------- Waitlist Admin ----------
        st.markdown("### 📬 Waitlist")
        wl_df = load_waitlist()
        if wl_df.empty:
            st.info("No waitlist entries yet.")
        else:
            st.dataframe(wl_df, use_container_width=True)

            if st.button("⚙️ Run Auto-Match Now"):
                assigned_count, changes = auto_match_waitlist()
                if assigned_count > 0:
                    st.success(f"Assigned {assigned_count} waitlist request(s): " + ", ".join([f"#{i}→{u}" for i, u in changes]))
                else:
                    st.info("No matches found right now (no available units for pending waitlist entries).")
            wl_df = load_waitlist()  # refresh after potential changes
            st.download_button("⬇️ Export Waitlist CSV", wl_df.to_csv(index=False).encode("utf-8"), "waitlist.csv", mime="text/csv")
    else:
        st.info("No appointments to show yet.")
else:
    st.warning("🔐 Admin access required to view dashboard.")

st.markdown("---")
st.caption("Built with ❤️ by Doctigo AI Booking System")
