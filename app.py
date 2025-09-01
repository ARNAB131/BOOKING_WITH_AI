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

def eta_minutes_from_distance_km(d_km: float) -> int:
    """<=5km→15m, <=10km→20m, <=20km→30m, else ~2min/km."""
    if d_km is None or (isinstance(d_km, float) and math.isnan(d_km)):
        return 0
    if d_km <= 5:
        return 15
    if d_km <= 10:
        return 20
    if d_km <= 20:
        return 30
    return int(round(d_km * 2))

def extract_time_and_date_from_slot_field(slot_field: str) -> tuple[str, str]:
    """From '10:00 AM - 10:15 AM on 01 September 2025' → ('10:00 AM - 10:15 AM','01 September 2025')"""
    if not isinstance(slot_field, str):
        return ("", "")
    if " on " in slot_field:
        t, d = slot_field.split(" on ", 1)
        return (t.strip(), d.strip())
    return (slot_field.strip(), "")

def get_booked_times_for(doctor_name: str, day: date) -> set[str]:
    """Return a set of time-texts already booked for doctor on the given date."""
    booked = set()
    ap_path = "appointments.csv"
    if not os.path.exists(ap_path):
        return booked
    try:
        df = pd.read_csv(ap_path)
    except Exception:
        return booked
    if "Doctor" not in df.columns or "Slot" not in df.columns:
        return booked
    day_str = day.strftime("%d %B %Y")
    df = df[df["Doctor"].astype(str) == str(doctor_name)]
    for s in df["Slot"].dropna().astype(str).tolist():
        t, d = extract_time_and_date_from_slot_field(s)
        if d == day_str and t:
            booked.add(t)
    return booked

def pick_first_available_slot(slots: list[str], doctor_name: str, day: date) -> str | None:
    """Return the first non-booked slot for that doctor/day."""
    taken = get_booked_times_for(doctor_name, day)
    for s in slots:
        if s not in taken:
            return s
    return None

def hospital_coords_for_chamber(chamber_text: str, hospitals_df: pd.DataFrame) -> tuple[float | None, float | None, str]:
    """Try to map a doctor's 'Chamber' to a hospital in hospitals.csv."""
    if hospitals_df.empty or not isinstance(chamber_text, str):
        return (None, None, "")
    chn = _normalize_text(chamber_text)
    for _, row in hospitals_df.iterrows():
        hname = str(row["Hospital"])
        if _normalize_text(hname) in chn:
            try:
                return (float(row["Latitude"]), float(row["Longitude"]), hname)
            except Exception:
                return (None, None, hname)
    return (None, None, "")

# --- Original appointment PDF (kept, unicode-safe) ---
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
    pdf = FPDF()
    pdf.add_page()

    pdf.set_font("Arial", "B", 18)
    pdf.cell(0, 10, pdfsafe(hospital_name or "Doctigo"), ln=True, align='C')

    pdf.set_font("Arial", "B", 12)
    pdf.ln(3)
    pdf.cell(0, 8, pdfsafe("Booking Summary"), ln=True, align='C')
    pdf.set_font("Arial", "", 11)
    pdf.cell(0, 5, pdfsafe("----------------------------------------"), ln=True, align='C')

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

    if appointment:
        pdf.ln(3)
        pdf.set_font("Arial", "B", 12)
        pdf.cell(0, 8, pdfsafe("Doctor Appointment"), ln=True)
        pdf.set_font("Arial", "", 11)
        ap_text = f"""
Doctor: Dr. {appointment.get('doctor','')}
Chamber: {appointment.get('chamber','')}
Slot: {appointment.get('slot','')}
Symptoms: {appointment.get('symptoms','')}
""".strip()
        pdf.multi_cell(0, 7, pdfsafe(ap_text))

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
INVENTORY_PATH = "beds_inventory.csv"

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
                today_str = datetime.today().strftime('%d %B %Y')
                today_dt = datetime.today().date()
                for _, row in df_bulk.iterrows():
                    name = row["Patient Name"]
                    symptoms = [s.strip() for s in row["Symptoms"].split(',') if s.strip()]
                    msg, recs = recommend_doctors(symptoms)
                    if recs:
                        selected = recs[0]
                        first_avail = pick_first_available_slot(selected.get('Slots', []), selected['Doctor'], today_dt)
                        if not first_avail:
                            continue
                        full_slot = f"{first_avail} on {today_str}"
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
                st.success(f"✅ Successfully booked (where available) for {len(df_bulk)} patients.")
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
# EMERGENCY → Auto-pick nearest + manual radio selection
# ------------------------------------------------------------------------------------
if st.session_state.flow_step == "emergency":
    st.subheader("🚑 Nearest Hospitals (Live distance & ETA)")
    st.caption("Tap **Start Live Tracking** to auto-detect your location. We’ll pick the nearest hospital, and you can switch manually if needed.")

    # Read lat/lon from URL query params (set by JS), using new API
    params = st.query_params
    # params.get returns a string (new API). But be defensive if a list-like is returned.
    def _as_str(v):
        if isinstance(v, list) and v:
            return v[0]
        return v
    qp_lat = _as_str(params.get("lat"))
    qp_lon = _as_str(params.get("lon"))

    # If URL contains lat/lon, persist to session
    if qp_lat and qp_lon:
        try:
            st.session_state.user_location["lat"] = float(qp_lat)
            st.session_state.user_location["lon"] = float(qp_lon)
        except Exception:
            pass

    # Show small control panel (JS writes lat/lon to URL and reloads the app)
    components.html(
        """
        <div style="display:flex;gap:8px;align-items:center;margin:.25rem 0 .75rem 0;">
          <button id="start" style="padding:.45rem .9rem;border-radius:10px;border:1px solid #2a9d8f;color:#2a9d8f;background:#fff;cursor:pointer;">▶ Start Live Tracking</button>
          <button id="stop"  style="padding:.45rem .9rem;border-radius:10px;border:1px solid #94a3b8;color:#475569;background:#fff;cursor:pointer;">⏸ Stop</button>
          <span id="status" style="font-size:.9rem;color:#475569;margin-left:.25rem;">Waiting for location…</span>
        </div>
        <script>
          let watchId = null;
          const statusEl = document.getElementById("status");

          function reloadWith(lat, lon){
            try {
              const u = new URL(window.parent.location.href);
              u.searchParams.set('lat', lat.toFixed(6));
              u.searchParams.set('lon', lon.toFixed(6));
              window.parent.location = u.toString();
            } catch(e) {
              // fallback
              window.parent.location.reload();
            }
          }

          function onFirstFix(lat, lon){
            if (statusEl) statusEl.textContent = "Lat: " + lat.toFixed(6) + ", Lon: " + lon.toFixed(6);
            // Immediately reload the app with ?lat&lon so Python can react.
            reloadWith(lat, lon);
          }

          document.getElementById("start").addEventListener("click", () => {
            if (!navigator.geolocation){
              alert("Geolocation not supported in this browser.");
              return;
            }
            if (watchId !== null) return;
            navigator.geolocation.getCurrentPosition((pos) => {
              onFirstFix(pos.coords.latitude, pos.coords.longitude);
            }, (err) => {
              alert("Unable to get location: " + err.message);
            }, { enableHighAccuracy:true, timeout: 10000 });

            // Optional: keep tracking to keep status fresh (not needed for backend)
            watchId = navigator.geolocation.watchPosition((pos) => {
              if (statusEl) statusEl.textContent = "Lat: " + pos.coords.latitude.toFixed(6) + ", Lon: " + pos.coords.longitude.toFixed(6);
            }, (err) => {
              console.log("Geo error", err);
            }, { enableHighAccuracy:true, maximumAge: 1000, timeout: 10000 });
          });

          document.getElementById("stop").addEventListener("click", () => {
            if (watchId !== null){
              navigator.geolocation.clearWatch(watchId);
              watchId = null;
              if (statusEl) statusEl.textContent = "Tracking paused.";
            }
          });
        </script>
        """,
        height=80,
    )

    hospitals_df = load_hospitals()
    if hospitals_df.empty:
        st.info("Upload or prepare `hospitals.csv` to see nearby hospitals.")
        st.stop()

    lat0 = st.session_state.user_location["lat"]
    lon0 = st.session_state.user_location["lon"]

    if lat0 is None or lon0 is None:
        st.info("Click **Start Live Tracking**. The page will refresh and show nearby hospitals automatically.")
    else:
        # Compute distance & ETA
        def _row_calc(r):
            try:
                d = haversine_km(lat0, lon0, float(r["Latitude"]), float(r["Longitude"]))
                return pd.Series({"Distance_km": d, "ETA_min": eta_minutes_from_distance_km(d)})
            except Exception:
                return pd.Series({"Distance_km": float("nan"), "ETA_min": 0})

        extra = hospitals_df.apply(_row_calc, axis=1)
        hospitals_df = pd.concat([hospitals_df, extra], axis=1).sort_values("Distance_km")
        hospitals_df = hospitals_df.reset_index(drop=True)

        # Auto-pick nearest hospital
        if not hospitals_df.empty:
            nearest = hospitals_df.iloc[0]
            auto_name = str(nearest["Hospital"])
            auto_km = nearest["Distance_km"]
            auto_eta = nearest["ETA_min"]
            st.session_state.selected_hospital = auto_name  # auto-select
            st.success(f"Nearest hospital auto-selected: **{auto_name}** — {auto_km:.2f} km (~{auto_eta} min)")

        # Manual override via radio
        labels = []
        values = []
        for _, r in hospitals_df.iterrows():
            hname = str(r["Hospital"])
            km = r["Distance_km"]
            eta = r["ETA_min"]
            lab = f"{hname} — {km:.2f} km (~{eta} min)"
            labels.append(lab)
            values.append(hname)

        default_index = 0 if values else None
        choice = st.radio(
            "Or choose a different hospital:",
            options=range(len(values)),
            format_func=lambda i: labels[i],
            index=default_index if default_index is not None else 0
        )
        chosen_name = values[choice] if values else None

        if chosen_name and chosen_name != st.session_state.selected_hospital:
            st.session_state.selected_hospital = chosen_name

        # Show a compact table, too
        show_df = hospitals_df[["Hospital", "Address", "Distance_km", "ETA_min"]].rename(
            columns={"Distance_km": "Distance (km)", "ETA_min": "ETA (min)"}
        )
        st.dataframe(show_df, use_container_width=True)

        # Proceed
        if st.button("Continue ➜"):
            st.session_state.flow_step = "hospital"

# ------------------------------------------------------------------------------------
# HOSPITAL/DOCTORS PAGE (Normal Booking logic unchanged)
# ------------------------------------------------------------------------------------
if st.session_state.flow_step in ("hospital", "doctors"):
    doctor_df = load_doctors_file()
    if doctor_df.empty:
        st.stop()

    hospitals_df = load_hospitals()
    chosen_hospital = st.session_state.selected_hospital

    if st.session_state.flow_step == "hospital" and chosen_hospital:
        st.subheader(f"🏥 {chosen_hospital}")
    else:
        st.subheader("🧑‍⚕️ Find Doctors")

    hospital_address = ""
    if chosen_hospital and not hospitals_df.empty:
        row = hospitals_df[hospitals_df["Hospital"].astype(str).str.casefold() == chosen_hospital.casefold()]
        if not row.empty and "Address" in row.columns:
            hospital_address = str(row.iloc[0]["Address"])

    patient_name = st.text_input("Enter your name:", key="patient_name")

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

    symptom_options = list(symptom_specialization_map.keys())
    symptoms_selected = st.multiselect("Select your symptom(s):", options=symptom_options)
    symptoms_typed = st.text_input("Or type your symptoms (comma-separated):", key="typed_symptoms")
    all_symptoms = list(set(symptoms_selected + [s.strip() for s in symptoms_typed.split(',') if s.strip()]))

    filtered_doctors = doctor_df.copy()
    if chosen_hospital:
        filtered_doctors = filter_doctors_by_hospital(doctor_df, chosen_hospital, hospital_address)
        if filtered_doctors.empty:
            st.info("No exact matches by Hospital/Chamber; showing all doctors.")
            filtered_doctors = doctor_df

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

    st.markdown("---")
    if st.button("🛏️ Book Beds/Cabins"):
        if not st.session_state.selected_hospital and chosen_hospital:
            st.session_state.selected_hospital = chosen_hospital
        st.session_state.flow_step = "beds"

# ------------------------------------------------------------------------------------
# BEDS/CABINS — Checkbox selection (no typing)
# ------------------------------------------------------------------------------------
if st.session_state.flow_step == "beds":
    hospital_name_for_beds = st.session_state.selected_hospital or "Doctigo Partner Hospital"
    st.subheader(f"🛏️ Beds & Cabins – {hospital_name_for_beds}")
    st.caption("Pick a date, then tick exactly one box to select a unit. Sold units are disabled.")

    beds_day = st.date_input(
        "Availability date (usually check-in):",
        value=st.session_state.get("beds_avail_day", date.today()),
        key="beds_avail_day"
    )

    tiers = [
        {"tier": "General Bed",   "price": 100,  "features": ["1 bed","1 chair","bed table"], "ids": [f"G-{i}" for i in range(1,41)], "cols": 10},
        {"tier": "General Cabin", "price": 1000, "features": ["2 beds","attached washroom","bed table","chair","food x3 times"], "ids": [f"C-{i}" for i in range(1,21)], "cols": 7},
        {"tier": "VIP Cabin",     "price": 4000, "features": ["premium bed x2","sofa","Air Conditioning","attached washroom","TV","fridge","bed table x2","coffee table","2 chairs"], "ids": [f"V-{i}" for i in range(1,11)], "cols": 5},
    ]
    tier_names = [t["tier"] for t in tiers]
    pick_tier = st.radio("Select type", options=tier_names)
    tier_obj = next(t for t in tiers if t["tier"] == pick_tier)

    inv = get_inventory(hospital_name_for_beds, tier_obj["tier"], beds_day)
    sold = set(inv[inv["Status"] == "booked"]["UnitID"].tolist())

    with st.expander(f"ℹ️ What is included in {pick_tier}?"):
        st.markdown(f"**Rs {tier_obj['price']} per night**")
        st.markdown("- " + "\n- ".join(tier_obj["features"]))

    st.markdown("#### Choose a specific unit")
    ncols = tier_obj["cols"]
    cols = st.columns(ncols)
    selected_ids = []
    day_key = beds_day.isoformat()
    tier_key = pick_tier.replace(" ", "_")
    hosp_key = (hospital_name_for_beds or "Hosp").replace(" ", "_")

    for i, uid in enumerate(tier_obj["ids"]):
        c = cols[i % ncols]
        disabled = uid in sold
        default_checked = (st.session_state.get("seat_selected") == uid)
        with c:
            checked = st.checkbox(
                uid,
                key=f"chk_{hosp_key}_{tier_key}_{day_key}_{uid}",
                disabled=disabled,
                value=default_checked and not disabled
            )
        if checked and not disabled:
            selected_ids.append(uid)

    if len(selected_ids) == 0:
        st.info("No unit selected yet.")
    elif len(selected_ids) == 1:
        st.success(f"Selected: **{selected_ids[0]}** for **{beds_day.strftime('%d %b %Y')}**")
    else:
        st.warning("Please keep only **one** checkbox ticked.")

    can_next = (len(selected_ids) == 1)
    if st.button("Next ➜", disabled=not can_next):
        chosen_uid = selected_ids[0]
        st.session_state.bed_choice = {
            "tier": tier_obj["tier"],
            "unit_id": chosen_uid,
            "price": tier_obj["price"],
            "features": tier_obj["features"],
        }
        st.session_state.seat_selected = chosen_uid
        st.session_state.selected_beds_day = beds_day
        st.session_state.flow_step = "details"

# ------------------------------------------------------------------------------------
# PATIENT DETAILS (kept)
# ------------------------------------------------------------------------------------
if st.session_state.flow_step == "details":
    st.subheader("👤 Patient & Stay Details")
    with st.form("patient_form"):
        name = st.text_input("Patient's Name", value=st.session_state.get("patient_name",""))
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
# SUMMARY + COMBINED PDF + CONFIRM
# ------------------------------------------------------------------------------------
if st.session_state.flow_step == "summary":
    hospital_header = st.session_state.selected_hospital or "Doctigo Partner Hospital"
    st.success("✅ Details captured. Your combined summary is ready.")

    st.markdown("---")
    st.markdown(f"### 🏥 {hospital_header}")

    if st.session_state.appointment:
        ap = st.session_state.appointment
        st.markdown(f"**Doctor:** Dr. {ap['doctor']}  \n**Chamber:** {ap['chamber']}  \n**Slot:** {ap['slot']}  \n**Symptoms:** {ap.get('symptoms','')}")
    else:
        st.info("No doctor appointment selected.")

    if st.session_state.bed_choice:
        bc = st.session_state.bed_choice
        st.markdown(f"**Bed/Cabin:** {bc['tier']} (Rs {bc['price']}/night)  \n**Unit ID:** {bc.get('unit_id','Any')}  \n**Features:** {', '.join(bc['features'])}")
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

    if st.session_state.appointment:
        ap = st.session_state.appointment
        time_text, date_text = extract_time_and_date_from_slot_field(ap["slot"])
        try:
            ap_day = datetime.strptime(date_text, "%d %B %Y").date()
        except Exception:
            ap_day = datetime.today().date()
        already = get_booked_times_for(ap["doctor"], ap_day)
        if time_text in already:
            st.error("❌ That appointment timeframe was just taken. Please go back and choose another slot.")
        else:
            if st.button("✅ Confirm & Save Appointment"):
                ap_df = pd.DataFrame([{
                    "Patient Name": (st.session_state.patient_details or {}).get("name", st.session_state.get("patient_name","")),
                    "Symptoms": ap.get("symptoms",""),
                    "Doctor": ap["doctor"],
                    "Chamber": ap["chamber"],
                    "Slot": ap["slot"],
                    "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }])
                if os.path.exists("appointments.csv"):
                    ap_df.to_csv("appointments.csv", mode="a", header=False, index=False)
                else:
                    ap_df.to_csv("appointments.csv", mode="w", header=True, index=False)
                st.success("✅ Appointment saved and timeframe locked for the day.")

    if st.session_state.bed_choice and st.session_state.bed_choice.get("unit_id"):
        start_day = datetime.strptime(st.session_state.patient_details.get("checkin_date"), "%d %B %Y").date()
        if st.session_state.patient_details.get("checkout_mode") == "date" and st.session_state.patient_details.get("checkout_date"):
            end_day = datetime.strptime(st.session_state.patient_details["checkout_date"], "%d %B %Y").date()
        else:
            end_day = start_day
        if st.button("🛏️ Confirm & Reserve Bed/Cabin"):
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

    csv_file = io.StringIO()
    csv_file.write("Doctor,Chamber,Slot,Symptoms\n")
    csv_file.write(f"{st.session_state.booked_doctor},{chamber_val},{st.session_state.slot},{'; '.join(st.session_state.symptoms_used)}\n")
    st.download_button("⬇️ Download CSV", data=csv_file.getvalue(), file_name="appointment.csv", mime="text/csv")

    pdf_buffer = generate_pdf_receipt(
        st.session_state.get("patient_name", ""),
        st.session_state.booked_doctor,
        chamber_val,
        st.session_state.slot,
        '; '.join(st.session_state.symptoms_used)
    )
    st.download_button("⬇️ Download PDF", data=pdf_buffer, file_name="appointment.pdf", mime="application/pdf")

# ------------------------------------------------------------------------------------
# Admin Dashboard (kept) + Bed Inventory Tools
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

        st.markdown("### 🧰 Bed Inventory Tools")
        hospitals_df = load_hospitals()
        h_options = hospitals_df["Hospital"].tolist() if not hospitals_df.empty else []
        sel_hosp = st.selectbox("Hospital", options=h_options) if h_options else st.text_input("Hospital (type)")
        sel_tier = st.selectbox("Tier", options=["General Bed", "General Cabin", "VIP Cabin"])
        sel_day = st.date_input("Date to reset", value=date.today())

        if st.button("♻️ Reset availability for that date"):
            reset_inventory_for_date(sel_hosp, sel_tier, sel_day)
            st.success("Availability reset to ALL AVAILABLE for that date/tier.")
else:
    st.warning("🔐 Admin access required to view dashboard.")

st.markdown("---")
st.caption("Built with ❤️ by Doctigo AI Booking System")
