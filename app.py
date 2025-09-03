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
    # Fix common mojibake + keep symbols printable
    s = (s.replace("₹", "Rs ")
           .replace("–", "-").replace("—", "-")
           .replace("•", "*")
           .replace("“", '"').replace("”", '"')
           .replace("’", "'").replace("‘", "'"))
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
    if "Hospital" in doctor_df.columns:
        mask = doctor_df["Hospital"].astype(str).str.casefold() == hospital_name.casefold()
        exact = doctor_df[mask]
        if not exact.empty:
            return exact
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
            csv.writer(f).writerow([
                "Patient Name","Symptoms","Doctor","Chamber","Slot","Timestamp","Date","SlotTime",
                "Serial","DistanceKm","ETAmin","UserLat","UserLon","Hospital","HospitalLat","HospitalLon","AvgSpeedKmph"
            ])

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
        return pd.DataFrame(columns=[
            "Patient Name","Symptoms","Doctor","Chamber","Slot","Timestamp","Date","SlotTime",
            "Serial","DistanceKm","ETAmin","UserLat","UserLon","Hospital","HospitalLat","HospitalLon","AvgSpeedKmph"
        ])
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

# NEW: speed-aware ETA (with fallback buckets)
def _bucket_eta(distance_km: float) -> int:
    if distance_km <= 5:
        return 15
    if distance_km <= 10:
        return 20
    if distance_km <= 20:
        return 30
    return 45

def travel_eta_minutes(distance_km: float, speed_kmph: float | None = None) -> int:
    """
    If speed is provided, use time = distance / speed.
    Otherwise fall back to the previous bucketed estimate.
    Always return a rounded integer minutes with sensible lower bound.
    """
    if distance_km is None or distance_km <= 0:
        return 5
    if speed_kmph and speed_kmph > 0:
        minutes = (distance_km / float(speed_kmph)) * 60.0
        return max(5, int(round(minutes)))
    return _bucket_eta(distance_km)

def compute_distance_and_eta(user_loc: dict, hospital_name: str, hospitals_df: pd.DataFrame, speed_kmph: float | None = None):
    if not user_loc or user_loc.get("lat") is None or user_loc.get("lon") is None:
        return None, None, None, None
    coords = get_hospital_coords(hospital_name, hospitals_df)
    if not coords:
        return None, None, None, None
    d = haversine_km(user_loc["lat"], user_loc["lon"], coords[0], coords[1])
    eta = travel_eta_minutes(d, speed_kmph)
    return d, eta, coords[0], coords[1]

def compute_serial(slot_time: str, visiting_time_str: str) -> int:
    try:
        all_slots = generate_slots(visiting_time_str)
    except Exception:
        all_slots = []
    try:
        return all_slots.index(slot_time) + 1
    except ValueError:
        return 1

def save_appointment_row(patient_name, symptoms, doctor, chamber, slot_full, visiting_time_str,
                         distance_km=None, eta_min=None, user_loc=None, hospital=None,
                         hosp_lat=None, hosp_lon=None, avg_speed=None):
    ensure_appointments_file()
    slot_time, date_str = parse_slot(slot_full)
    if not doctor or not slot_time or not date_str:
        return False, "Invalid slot or doctor.", None
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
        "UserLat": (user_loc or {}).get("lat", ""),
        "UserLon": (user_loc or {}).get("lon", ""),
        "Hospital": hospital or "",
        "HospitalLat": hosp_lat if hosp_lat is not None else "",
        "HospitalLon": hosp_lon if hosp_lon is not None else "",
        "AvgSpeedKmph": int(avg_speed) if isinstance(avg_speed, (int, float)) else "",
    }
    with open(APPOINTMENTS_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(new_row.keys()))
        if os.stat(APPOINTMENTS_PATH).st_size == 0:
            writer.writeheader()
        writer.writerow(new_row)
    return True, "✅ Appointment booked and saved.", serial

# --- PDF helpers --------------------------------------------------------------------
# NEW: small utility to format coordinates
def _fmt_coords(lat, lon):
    try:
        if lat is None or lon is None:
            return ""
        return f"{lat:.6f}, {lon:.6f}"
    except Exception:
        return ""

def generate_pdf_receipt(patient_name, doctor, chamber, slot, symptoms,
                         phone="", email="", distance_km=None, eta_min=None,
                         user_lat=None, user_lon=None, hospital_name=None,
                         hosp_lat=None, hosp_lon=None, avg_speed=None):
    """
    (Kept for compatibility if you use it elsewhere.)
    Now prints Phone, Email, and travel info too.
    """
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", "B", 18)
    pdf.cell(0, 10, pdfsafe(hospital_name or "Doctigo"), ln=True, align='C')
    pdf.set_font("Arial", "B", 12)
    pdf.ln(5)
    pdf.cell(0, 10, pdfsafe("Patient Appointment Receipt"), ln=True, align='C')
    pdf.set_font("Arial", "", 11)
    pdf.cell(0, 5, pdfsafe("----------------------------------------"), ln=True, align='C')
    pdf.ln(5)
    pdf.set_font("Arial", "", 12)
    pdf.cell(0, 8, pdfsafe(f"Patient Name: {patient_name}"), ln=True)
    pdf.cell(0, 8, pdfsafe(f"Phone: {phone}"), ln=True)     # NEW
    pdf.cell(0, 8, pdfsafe(f"Email: {email}"), ln=True)     # NEW
    pdf.cell(0, 8, pdfsafe(f"Doctor: Dr. {doctor}"), ln=True)
    pdf.cell(0, 8, pdfsafe(f"Chamber: {chamber}"), ln=True)
    pdf.cell(0, 8, pdfsafe(f"Slot: {slot}"), ln=True)
    if distance_km is not None and eta_min is not None:
        pdf.cell(0, 8, pdfsafe(f"Distance/ETA: {distance_km:.2f} km / ~{eta_min} min (avg {avg_speed or '—'} km/h)"), ln=True)
        pdf.cell(0, 8, pdfsafe(f"From (you): {_fmt_coords(user_lat, user_lon)}"), ln=True)
        pdf.cell(0, 8, pdfsafe(f"To (hospital): {_fmt_coords(hosp_lat, hosp_lon)}"), ln=True)
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
    pdf.cell(0, 8, pdfsafe(f"Phone: {phone}"), ln=True)     # NEW
    pdf.cell(0, 8, pdfsafe(f"Email: {email}"), ln=True)     # NEW
    pdf.cell(0, 8, pdfsafe(f"Doctor: Dr. {doctor}"), ln=True)
    pdf.cell(0, 8, pdfsafe(f"Chamber: {chamber}"), ln=True)
    pdf.cell(0, 8, pdfsafe(f"Slot: {slot}"), ln=True)
    if distance_km is not None and eta_min is not None:
        pdf.cell(0, 8, pdfsafe(f"Distance/ETA: {distance_km:.2f} km / ~{eta_min} min (avg {avg_speed or '—'} km/h)"), ln=True)
        pdf.cell(0, 8, pdfsafe(f"From (you): {_fmt_coords(user_lat, user_lon)}"), ln=True)
        pdf.cell(0, 8, pdfsafe(f"To (hospital): {_fmt_coords(hosp_lat, hosp_lon)}"), ln=True)
    pdf.multi_cell(0, 8, pdfsafe(f"Symptoms: {symptoms}"))
    pdf.cell(0, 8, pdfsafe(f"Issued On: {datetime.now().strftime('%d %B %Y, %I:%M %p')}"), ln=True)
    pdf.ln(10)
    pdf.set_font("Arial", "I", 9)
    pdf.set_text_color(120)
    pdf.cell(0, 10, pdfsafe("This receipt is auto-generated by Doctigo AI Booking System."), ln=True, align="C")
    pdf.set_text_color(0)
    pdf_bytes = pdf.output(dest="S").encode("latin-1")
    return io.BytesIO(pdf_bytes)

def generate_full_pdf(hospital_name, patient, appointment, bed_choice):
    """
    Combined PDF for appointment + bed/cabin.
    Includes phone/email + distance/ETA + coordinates.
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

    # Patient Details (now guaranteed to show phone/email)
    pdf.ln(4)
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 8, pdfsafe("Patient Details"), ln=True)
    pdf.set_font("Arial", "", 11)

    details = f"""
Patient Name: {patient.get('name','')}
Phone: {patient.get('phone','')}
Gender: {patient.get('gender','')}
Age: {patient.get('age','')}
Email: {patient.get('email','')}
Address: {patient.get('address','')}
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
            dist_text = (
                f"\nDistance: {appointment['distance_km']:.2f} km"
                f" | ETA: {appointment['eta_min']} min"
                f" | Avg speed: {appointment.get('avg_speed','—')} km/h"
                f"\nFrom (you): {_fmt_coords(appointment.get('user_lat'), appointment.get('user_lon'))}"
                f"\nTo (hospital): {_fmt_coords(appointment.get('hosp_lat'), appointment.get('hosp_lon'))}"
            )
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
Check-in Date: {bed_choice.get('checkin_date','')}
Check-out Date: {bed_choice.get('checkout_date','') or 'To be decided'}
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
        mark_booked(hospital, tier, cur)
        cur += timedelta(days=1)

def reset_inventory_for_date(hospital: str, tier: str, day: date):
    ensure_inventory()
    _ensure_rows_for(hospital, tier, day)
    df = pd.read_csv(INVENTORY_PATH)
    mask = (df["Hospital"].astype(str) == hospital) & (df["Tier"] == tier) & (df["Date"] == day.strftime("%Y-%m-%d"))
    df.loc[mask, "Status"] = "available"
    df.to_csv(INVENTORY_PATH, index=False)

# ---------- Availability across a date range ----------
def units_available_for_range(hospital: str, tier: str, start_day: date, end_day: date) -> list[str]:
    """Return units that are available for ALL days in [start_day, end_day]."""
    ensure_inventory()
    if end_day < start_day:
        start_day, end_day = end_day, start_day
    avail_sets = []
    cur = start_day
    while cur <= end_day:
        _ensure_rows_for(hospital, tier, cur)
        inv = get_inventory(hospital, tier, cur)
        avail = set(inv[inv["Status"] != "booked"]["UnitID"].astype(str).tolist())
        avail_sets.append(avail)
        cur += timedelta(days=1)
    if not avail_sets:
        return []
    return sorted(set.intersection(*avail_sets)) if len(avail_sets) > 1 else sorted(avail_sets[0])

# ------------------------------------------------------------------------------------
# Waitlist helpers (CSV-backed)
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
# Realtime slot helpers
# ------------------------------------------------------------------------------------
def _extract_slot_start_time_str(slot_label: str) -> str | None:
    if not slot_label:
        return None
    m = re.search(r'(\d{1,2}:\d{2}\s*[APap][Mm])', slot_label)
    if m:
        return m.group(1).upper().replace(" ", "")
    m = re.search(r'\b(\d{1,2}:\d{2})\b', slot_label)
    if m:
        return m.group(1)
    return None

def _slot_start_datetime(the_day: date, slot_label: str) -> datetime | None:
    s = _extract_slot_start_time_str(slot_label)
    if not s:
        return None
    try:
        if s.lower().endswith(("am", "pm")) or ("AM" in s or "PM" in s):
            t = datetime.strptime(s.replace("AM"," AM").replace("PM"," PM"), "%I:%M %p").time()
        else:
            t = datetime.strptime(s, "%H:%M").time()
        return datetime.combine(the_day, t)
    except Exception:
        return None

def filter_future_slots_for_date(visiting_time_str: str, the_day: date, doctor_name: str) -> list[str]:
    all_slots = generate_slots(visiting_time_str)
    already = booked_slot_times_for(doctor_name, the_day)
    remaining = [s for s in all_slots if s not in already]
    if the_day == date.today():
        now_dt = datetime.now()
        out = []
        for s in remaining:
            sd = _slot_start_datetime(the_day, s)
            if not sd or sd > now_dt:
                out.append(s)
        return out
    return remaining

def fmt_countdown(seconds: int) -> str:
    if seconds < 0:
        seconds = 0
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    return f"{h:02d}hrs {m:02d}min {s:02d}sec"

# ------------------------------------------------------------------------------------
# Session state
# ------------------------------------------------------------------------------------
if "flow_step" not in st.session_state:
    st.session_state.flow_step = "home"  # home -> chat -> summary etc.

defaults = {
    "mode": None,  # "normal" or "emergency"
    "chat_stage": None,  # "greet","symptoms","doctor","slot","bed_ask","bed_select","details","card"
    "user_name": "",
    "user_location": {"lat": None, "lon": None},
    "symptoms_selected": [],
    "symptoms_typed": "",
    "recommendations": [],
    "chosen_doctor": None,
    "chosen_date": None,
    "chosen_slot": "",
    "appointment": None,
    "selected_hospital": None,
    "need_bed": None,
    "bed_choice": None,  # {'tier','unit_id','price','features','checkin_date','checkout_date'}
    "details_step": 0,
    "patient_details": {"name":"", "phone":"", "gender":"", "age":"", "email":"", "address":""},
    "doctor_message": "",
    "avg_speed": 25,  # NEW: default avg traffic speed (km/h)
}
for k,v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ------------------------------------------------------------------------------------
# HOME — two buttons
# ------------------------------------------------------------------------------------
st.title("🧑‍⚕️ Doctigo – AI Doctor & Bed/Cabin Booking")
col1, col2 = st.columns([1, 1])
with col1:
    if st.button("🔎 Normal Booking"):
        st.session_state.flow_step = "chat"
        st.session_state.mode = "normal"
        st.session_state.chat_stage = "greet"
with col2:
    if st.button("🚨 Emergency Booking"):
        st.session_state.flow_step = "chat"
        st.session_state.mode = "emergency"
        st.session_state.chat_stage = "greet"

# ------------------------------------------------------------------------------------
# CHAT FLOW
# ------------------------------------------------------------------------------------
def chat_bot(msg: str):
    st.markdown(f"🕷️ **Doc:** {msg}")

def chat_user(msg: str):
    st.markdown(f"👤 **You:** {msg}")

def geo_widget():
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

if st.session_state.flow_step == "chat":
    st.markdown("### 💬 Chat with Spider Doc")
    box = st.container(border=True)

    with box:
        # Stage: greet / ask name
        if st.session_state.chat_stage == "greet":
            chat_bot("Hello! I am **Doc**, your friendly neighborhood **Spider Doc** 🕷️🩺. What's your name?")
            name = st.text_input("Your name", key="chat_name_input")
            # also allow quick geolocation up front
            st.text_input("Latitude", value=st.session_state.user_location["lat"] or "", key="lat_input")
            st.text_input("Longitude", value=st.session_state.user_location["lon"] or "", key="lon_input")
            geo_widget()
            if st.button("Submit Name"):
                st.session_state.user_name = (name or "").strip()
                try:
                    st.session_state.user_location["lat"] = float(st.session_state.get("lat_input") or "") if st.session_state.get("lat_input") else None
                    st.session_state.user_location["lon"] = float(st.session_state.get("lon_input") or "") if st.session_state.get("lon_input") else None
                except:
                    pass
                if st.session_state.user_name:
                    chat_user(st.session_state.user_name)
                    choice = "normal booking" if st.session_state.mode == "normal" else "emergency booking"
                    chat_bot(f"Hello **{st.session_state.user_name}** — so you opted for **{choice}**.")
                    st.session_state.chat_stage = "symptoms"
                else:
                    st.warning("Please enter your name to continue.")

        # Stage: symptoms prompt
        elif st.session_state.chat_stage == "symptoms":
            if st.session_state.mode == "normal":
                chat_bot("Enter your symptoms **manually** or **select** from the drop-down. If no symptoms, type **Next**.")
            else:
                chat_bot("Woooo it's an **EMERGENCY**! Don't worry — I'm here with you. Enter the patient's symptoms **manually** or **select** from the drop-down. If you just want to book a doctor of your preference, type **Next**.")
            symptom_options = list(symptom_specialization_map.keys())
            st.session_state.symptoms_selected = st.multiselect("Select symptoms", options=symptom_options, default=st.session_state.symptoms_selected)
            st.session_state.symptoms_typed = st.text_input("Type symptoms (comma-separated)", value=st.session_state.symptoms_typed)
            nxt = st.text_input("Type 'Next' to skip symptoms (optional)", key="sym_next")
            proceed = st.button("Continue ➞")
            if proceed:
                typed = [s.strip() for s in st.session_state.symptoms_typed.split(",") if s.strip()]
                all_syms = sorted(list(set(st.session_state.symptoms_selected + typed)))
                st.session_state.symptoms_selected = all_syms
                if all_syms:
                    chat_user(", ".join(all_syms))
                elif (nxt or "").strip().casefold() == "next":
                    chat_user("Next")
                else:
                    st.warning("Please add symptoms or type Next.")
                    st.stop()

                doctors_df = load_doctors_file()
                if (nxt or "").strip().casefold() == "next" and not all_syms:
                    chat_bot("Select a doctor of your preference:")
                    st.session_state.recommendations = doctors_df.rename(
                        columns={"Doctor Name":"Doctor"}
                    )[["Doctor","Specialization","Chamber","Visiting Time"]].to_dict("records")
                    st.session_state.doctor_message = "Here are available doctors:"
                else:
                    msg, recs = recommend_doctors(st.session_state.symptoms_selected)
                    st.session_state.recommendations = recs or []
                    st.session_state.doctor_message = msg or "Recommended doctors:"
                st.session_state.chat_stage = "doctor"

        # Stage: choose doctor + date + real-time slot + distance/ETA + departure tip
        elif st.session_state.chat_stage == "doctor":
            doctors_df = load_doctors_file()
            chat_bot(st.session_state.doctor_message or "Recommended doctors:")
            if not st.session_state.recommendations:
                st.session_state.recommendations = doctors_df.rename(
                    columns={"Doctor Name":"Doctor"}
                )[["Doctor","Specialization","Chamber","Visiting Time"]].to_dict("records")

            # NEW: speed selector (affects ETA)
            st.session_state.avg_speed = st.select_slider(
                "Assumed average city speed (km/h) for ETA",
                options=[15, 20, 25, 30, 35, 40, 45, 50],
                value=st.session_state.get("avg_speed", 25)
            )

            # NEW: compute & show distance/ETA inside doctor choices when hospital can be detected
            hospitals_df = load_hospitals()
            doc_labels = []
            doc_map = []
            for d in st.session_state.recommendations:
                name = d["Doctor"]
                spec = d.get("Specialization","")
                chamber = d.get("Chamber","")
                label = f"Dr. {name} — {spec}"
                if hospitals_df is not None and not hospitals_df.empty and st.session_state.user_location.get("lat") is not None:
                    hosp = detect_hospital_for_chamber(chamber, hospitals_df)
                    if hosp:
                        dist_km, eta_min, _, _ = compute_distance_and_eta(
                            st.session_state.user_location, hosp, hospitals_df, speed_kmph=st.session_state.avg_speed
                        )
                        if dist_km is not None:
                            label += f"  •  ~{dist_km:.1f} km / {eta_min} min to {hosp}"
                doc_labels.append(label)
                doc_map.append(d)

            sel = st.selectbox("Choose a doctor", options=doc_labels)
            idx = doc_labels.index(sel) if sel in doc_labels else 0
            chosen = doc_map[idx]
            st.write(f"**Chamber:** {chosen.get('Chamber','')}")
            st.write(f"**Visiting Time:** {chosen.get('Visiting Time','')}")

            # Detect hospital for the chosen doctor's chamber
            hospitals_df = load_hospitals()
            hosp = detect_hospital_for_chamber(chosen.get("Chamber",""), hospitals_df)
            if hosp:
                st.session_state.selected_hospital = hosp
                st.info(f"🏥 Hospital detected: **{hosp}**")

            the_day = st.date_input("Choose appointment date", min_value=date.today(),
                                    value=st.session_state.chosen_date or date.today())
            available_slots = filter_future_slots_for_date(chosen.get("Visiting Time",""), the_day, chosen["Doctor"])
            if not available_slots:
                st.warning("No future slots available on this date.")
            else:
                slot = st.selectbox("Select a slot", options=available_slots, key="chat_slot")
                lat_val = st.text_input("Latitude", value=st.session_state.user_location["lat"] or "", key="lat_input_doc")
                lon_val = st.text_input("Longitude", value=st.session_state.user_location["lon"] or "", key="lon_input_doc")
                st.caption("Tip: use Detect My Location to auto-fill:")
                geo_widget()
                try:
                    st.session_state.user_location["lat"] = float(lat_val) if lat_val else None
                    st.session_state.user_location["lon"] = float(lon_val) if lon_val else None
                except:
                    pass

                dist_km, eta_min, hosp_lat, hosp_lon = None, None, None, None
                if st.session_state.selected_hospital:
                    dist_km, eta_min, hosp_lat, hosp_lon = compute_distance_and_eta(
                        st.session_state.user_location, st.session_state.selected_hospital, hospitals_df, speed_kmph=st.session_state.avg_speed
                    )

                serial_num = compute_serial(slot, chosen.get("Visiting Time",""))
                if dist_km is not None:
                    st.info(f"**Serial #{serial_num}**  |  **Distance:** {dist_km:.2f} km  |  **ETA:** ~{eta_min} min (avg {st.session_state.avg_speed} km/h)")
                else:
                    st.info(f"**Serial #{serial_num}**  |  Enable location to compute distance/ETA.")

                start_dt = _slot_start_datetime(the_day, slot)
                if start_dt and eta_min is not None:
                    depart_at = start_dt - timedelta(minutes=eta_min)
                    secs = int((depart_at - datetime.now()).total_seconds())
                    tip = (
                        f"hey you should set off **now** for the chamber. you will need approx **{eta_min} min** to reach."
                        if secs <= 0 else
                        f"hey you should set off in **{fmt_countdown(secs)}** for the chamber. you will need approx **{eta_min} min** to reach."
                    )
                    chat_bot(tip)

                if st.button("🗓 Book This Appointment"):
                    full_slot = f"{slot} on {the_day.strftime('%d %B %Y')}"
                    ok, msg, serial_saved = save_appointment_row(
                        patient_name=st.session_state.user_name,
                        symptoms='; '.join(st.session_state.symptoms_selected) if st.session_state.symptoms_selected else (st.session_state.mode.title() + " Flow"),
                        doctor=chosen["Doctor"],
                        chamber=chosen.get("Chamber",""),
                        slot_full=full_slot,
                        visiting_time_str=chosen.get("Visiting Time",""),
                        distance_km=dist_km,
                        eta_min=eta_min,
                        user_loc=st.session_state.user_location,        # NEW
                        hospital=st.session_state.selected_hospital,    # NEW
                        hosp_lat=hosp_lat,                              # NEW
                        hosp_lon=hosp_lon,                              # NEW
                        avg_speed=st.session_state.avg_speed            # NEW
                    )
                    if ok:
                        st.session_state.chosen_doctor = chosen
                        st.session_state.chosen_date = the_day
                        st.session_state.chosen_slot = slot
                        st.session_state.appointment = {
                            "doctor": chosen["Doctor"],
                            "chamber": chosen.get("Chamber",""),
                            "slot": full_slot,
                            "symptoms": '; '.join(st.session_state.symptoms_selected) if st.session_state.symptoms_selected else "",
                            "serial": serial_saved,
                            "distance_km": dist_km,
                            "eta_min": eta_min,
                            "user_lat": st.session_state.user_location.get("lat"),
                            "user_lon": st.session_state.user_location.get("lon"),
                            "hospital": st.session_state.selected_hospital,
                            "hosp_lat": hosp_lat,
                            "hosp_lon": hosp_lon,
                            "avg_speed": st.session_state.avg_speed
                        }
                        st.success(f"✅ Appointment booked. Serial #{serial_saved}.")
                        st.session_state.chat_stage = "bed_ask"
                    else:
                        st.error(msg)

        # Stage: ask bed/cabin
        elif st.session_state.chat_stage == "bed_ask":
            chat_bot('Do u wanna book a **Bed or Cabin**? If **Yes** type Yes, if **No** type No.')
            yn = st.text_input("Type Yes or No", key="bed_yesno")
            go = st.button("Submit ➞")
            if go:
                v = (yn or "").strip().casefold()
                if v not in ("yes","no"):
                    st.warning("Please type Yes or No.")
                else:
                    st.session_state.need_bed = (v == "yes")
                    chat_user("Yes" if st.session_state.need_bed else "No")
                    st.session_state.chat_stage = "bed_select" if st.session_state.need_bed else "details"

        # Stage: bed/cabin selection (multi-night support, same hospital as doctor)
        elif st.session_state.chat_stage == "bed_select":
            hospital_name_for_beds = st.session_state.selected_hospital or "Doctigo Partner Hospital"
            chat_bot(f"Great. Let's pick your **Bed/Cabin** at **{hospital_name_for_beds}**.")
            checkin = st.date_input("Choose check-in date", value=st.session_state.chosen_date or date.today(), key="bed_checkin")
            unknown = st.checkbox("I don't know the check-out date yet", value=False, key="bed_unknown")
            checkout = None
            if not unknown:
                checkout = st.date_input("Choose check-out date", value=checkin, min_value=checkin, key="bed_checkout")

            tiers = [
                {"tier": "General Bed",   "price": 100,  "features": ["1 bed","1 chair","bed table"]},
                {"tier": "General Cabin", "price": 1000, "features": ["2 beds","attached washroom","bed table","chair","food x3 times"]},
                {"tier": "VIP Cabin",     "price": 4000, "features": ["premium bed x2","sofa","Air Conditioning","attached washroom","TV","fridge","bed table x2","coffee table","2 chairs"]},
            ]
            tier_names = [t["tier"] for t in tiers]
            pick_tier = st.selectbox("Select type", options=tier_names)
            tier_obj = next(t for t in tiers if t["tier"] == pick_tier)

            # Compute availability across the full range
            start_day = checkin
            end_day = checkout if checkout else checkin
            avail_units = units_available_for_range(hospital_name_for_beds, tier_obj["tier"], start_day, end_day)

            if not avail_units:
                st.warning("All units are sold out for the selected stay. You may try a different type or dates, or continue without bed.")
            unit_id = st.selectbox("Select a unit", options=(avail_units or ["None"]))
            st.caption("After selection, type **Next** below.")
            nxt = st.text_input("Type Next to continue", key="bed_next")

            if st.button("Save Bed/Cabin ➞"):
                if avail_units and unit_id not in avail_units:
                    st.warning("Please select an available unit.")
                else:
                    if avail_units:
                        st.session_state.bed_choice = {
                            "tier": tier_obj["tier"],
                            "unit_id": unit_id,
                            "price": tier_obj["price"],
                            "features": tier_obj["features"],
                            "checkin_date": checkin.strftime("%d %B %Y"),
                            "checkout_date": checkout.strftime("%d %B %Y") if checkout else "",
                        }
                        st.success(
                            f"Selected {pick_tier} • Unit {unit_id} "
                            f"for {checkin.strftime('%d %b %Y')}"
                            + (f" → {checkout.strftime('%d %b %Y')}" if checkout else "")
                        )
                    else:
                        st.session_state.bed_choice = None

            if (st.session_state.get("bed_next") or "").strip().casefold() == "next":
                st.session_state.chat_stage = "details"

        # Stage: details one-by-one
        elif st.session_state.chat_stage == "details":
            steps = [
                ("name", "Please Enter Patient's Name"),
                ("phone", "Please Enter Phone Number"),
                ("gender", "Please Enter Gender"),
                ("age", "Please Enter Age"),
                ("email", "Please Enter Email address"),
                ("address", "Please Enter Address"),
            ]
            field, prompt = steps[st.session_state.details_step]
            chat_bot(prompt)
            if field == "age":
                val = st.number_input("Age", min_value=0, max_value=120, value=int(st.session_state.patient_details["age"] or 30), step=1, key="detail_age")
                clicked = st.button("Save ➞")
                if clicked:
                    st.session_state.patient_details["age"] = int(val)
                    st.session_state.details_step += 1
            elif field == "gender":
                val = st.selectbox("Gender", options=["Male","Female","Other"], index=0 if not st.session_state.patient_details["gender"] else ["Male","Female","Other"].index(st.session_state.patient_details["gender"]))
                clicked = st.button("Save ➞")
                if clicked:
                    st.session_state.patient_details["gender"] = val
                    st.session_state.details_step += 1
            else:
                key_map = {"name":"detail_name","phone":"detail_phone","email":"detail_email","address":"detail_address"}
                val = st.text_input("Type here", value=st.session_state.patient_details[field], key=key_map.get(field,f"detail_{field}"))
                clicked = st.button("Save ➞")
                if clicked:
                    st.session_state.patient_details[field] = val.strip()
                    st.session_state.details_step += 1

            if st.session_state.details_step >= len(steps):
                st.session_state.chat_stage = "card"

        # Stage: appointment card + download, and reserve bed (multi-night)
        elif st.session_state.chat_stage == "card":
            ap = st.session_state.appointment or {}
            bc = st.session_state.bed_choice
            pdets = st.session_state.patient_details

            chat_bot("Here is your **appointment card** with all details:")
            st.markdown("#### 🏥 Appointment Summary")
            if ap:
                extra = ""
                if ap.get("serial"):
                    extra += f"  \n**Serial #:** {ap['serial']}"
                if ap.get("distance_km") is not None and ap.get("eta_min") is not None:
                    extra += f"  \n**Distance/ETA:** {ap['distance_km']:.2f} km / {ap['eta_min']} min (avg {ap.get('avg_speed','—')} km/h)"
                st.markdown(f"**Doctor:** Dr. {ap.get('doctor','')}  \n**Chamber:** {ap.get('chamber','')}  \n**Slot:** {ap.get('slot','')}{extra}")
            else:
                st.info("No appointment found.")

            st.markdown("#### 👤 Patient")
            st.markdown(
                f"**Name:** {pdets.get('name','')}  \n"
                f"**Phone:** {pdets.get('phone','')}  \n"
                f"**Gender:** {pdets.get('gender','')}  \n"
                f"**Age:** {pdets.get('age','')}  \n"
                f"**Email:** {pdets.get('email','')}  \n"
                f"**Address:** {pdets.get('address','')}"
            )

            if bc:
                st.markdown("#### 🛏️ Bed/Cabin")
                st.markdown(
                    f"**Type:** {bc['tier']} (₹{bc['price']}/night)  \n"
                    f"**Unit ID:** {bc.get('unit_id','Any')}  \n"
                    f"**Check-in:** {bc.get('checkin_date','')}  \n"
                    f"**Check-out:** {bc.get('checkout_date','') or 'To be decided'}  \n"
                    f"**Features:** {', '.join(bc['features'])}"
                )

            # NEW: enrich appointment object with any missing pieces for PDF
            if ap:
                ap.setdefault("user_lat", st.session_state.user_location.get("lat"))
                ap.setdefault("user_lon", st.session_state.user_location.get("lon"))
                ap.setdefault("hospital", st.session_state.selected_hospital)
                # hosp_lat/lon may be None if not detected earlier; try again for PDF
                if ap.get("hosp_lat") is None or ap.get("hosp_lon") is None:
                    hospitals_df = load_hospitals()
                    if st.session_state.selected_hospital and not hospitals_df.empty:
                        coords = get_hospital_coords(st.session_state.selected_hospital, hospitals_df)
                        if coords:
                            ap["hosp_lat"], ap["hosp_lon"] = coords[0], coords[1]
                ap.setdefault("avg_speed", st.session_state.avg_speed)

            pdf_buf = generate_full_pdf(
                hospital_name=st.session_state.selected_hospital or "Doctigo Partner Hospital",
                patient=pdets,
                appointment=ap,
                bed_choice=bc
            )
            st.download_button("⬇️ Download Appointment Card (PDF)", data=pdf_buf, file_name="doctigo_appointment_card.pdf", mime="application/pdf")

            # Reserve bed in inventory now across the full date range
            if bc and bc.get("unit_id"):
                start_day = datetime.strptime(bc["checkin_date"], "%d %B %Y").date() if bc.get("checkin_date") else date.today()
                end_day = datetime.strptime(bc["checkout_date"], "%d %B %Y").date() if bc.get("checkout_date") else start_day
                if st.button("✅ Confirm & Reserve Selected Bed/Cabin"):
                    mark_booked_range(
                        hospital=st.session_state.selected_hospital or "Doctigo Partner Hospital",
                        tier=bc["tier"],
                        unit_id=bc["unit_id"],
                        start_day=start_day,
                        end_day=end_day
                    )
                    st.success(
                        "Bed/Cabin reserved for the selected date" +
                        ("" if start_day == end_day else f" range ({start_day.strftime('%d %b %Y')} → {end_day.strftime('%d %b %Y')}).")
                    )

            st.markdown("---")
            if st.button("🏠 Back to Home"):
                for k in ["mode","chat_stage","user_name","symptoms_selected","symptoms_typed","recommendations",
                          "chosen_doctor","chosen_date","chosen_slot","appointment","need_bed","bed_choice",
                          "details_step","patient_details","doctor_message"]:
                    if k in st.session_state:
                        del st.session_state[k]
                st.session_state.flow_step = "home"

    st.stop()

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
        st.subheader(f"📄 Appointments Done Till Now: **{total}**")

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
            filtered_df['Parsed Date'] = pd.to_datetime(filtered_df['Slot'].str.extract(r'on (.+)$')[0], errors='coerce')
            upcoming = filtered_df[filtered_df['Parsed Date'].between(today, today + timedelta(days=3))]

        if not upcoming.empty:
            st.markdown("### ⏰ Upcoming Appointment Reminders")
            date_col = "ParsedDate" if "ParsedDate" in upcoming.columns else "Parsed Date"
            for _, row in upcoming.iterrows():
                if pd.notnull(row[date_col]):
                    st.info(f"🗓 {row[date_col].strftime('%d %b %Y')} - {row.get('Patient Name','(Name)')} with Dr. {row['Doctor']} (Serial #{row.get('Serial','-')})")

        st.download_button("⬇️ Export All Appointments CSV", filtered_df.to_csv(index=False).encode('utf-8'),
                           "appointments_admin.csv", mime="text/csv")

        st.markdown("### 🧰 Bed Inventory Tools")
        hospitals_df = load_hospitals()
        h_options = hospitals_df["Hospital"].tolist() if not hospitals_df.empty else []
        sel_hosp = st.selectbox("Hospital", options=h_options) if h_options else st.text_input("Hospital (type)")
        sel_tier = st.selectbox("Tier", options=["General Bed", "General Cabin", "VIP Cabin"])
        sel_day = st.date_input("Date to reset", value=date.today())

        if st.button("♻️ Reset availability for that date"):
            reset_inventory_for_date(sel_hosp, sel_tier, sel_day)
            st.success("Availability reset to ALL AVAILABLE for that date/tier.")

        st.markdown("### 🧾 Waitlist")
        wl_df = load_waitlist()
        if wl_df.empty:
            st.info("No waitlist entries yet.")
        else:
            st.dataframe(wl_df, use_container_width=True)
            if st.button("🔧 Run Auto-Match Now"):
                assigned_count, changes = auto_match_waitlist()
                if assigned_count > 0:
                    st.success(f"Assigned {assigned_count} waitlist request(s): " + ", ".join([f\"#{i}→{u}\" for i, u in changes]))
                else:
                    st.info("No matches found right now.")
            wl_df = load_waitlist()
            st.download_button("⬇️ Export Waitlist CSV", wl_df.to_csv(index=False).encode("utf-8"), "waitlist.csv", mime="text/csv")
else:
    st.warning("🔐 Admin access required to view dashboard.")

st.markdown("---")
st.caption("Built with ❤️ by Doctigo AI Booking System")
