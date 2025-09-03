# app.py
# Doctigo – AI Doctor & Bed/Cabin Booking
# Modern chat UI (like screenshot) + bottom composer + contact backfill + final departure tip
# ----------------------------------------------------------------------------------------------
import os, io, re, csv, math
from datetime import datetime, timedelta, date
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from fpdf import FPDF

# ------------------------ optional fallback for ai_booking ------------------------
try:
    from ai_booking import recommend_doctors, symptom_specialization_map, generate_slots  # type: ignore
except Exception:
    symptom_specialization_map = {
        "fever": ["General Medicine"], "cough": ["General Medicine"],
        "chest pain": ["Cardiologist"], "ear pain": ["ENT Surgeon"],
        "throat pain": ["ENT Surgeon"], "hypertension": ["Cardiologist"],
        "gastric": ["General Medicine"],
    }
    def recommend_doctors(symptoms):
        specs = set()
        for s in (symptoms or []):
            specs.update(symptom_specialization_map.get(str(s).lower(), []))
        msg = "Recommended doctors based on your symptoms." if specs else "Here are available doctors:"
        df = pd.DataFrame()
        try:
            df = pd.read_csv("doctor.csv") if os.path.exists("doctor.csv") else pd.read_csv("doctors.csv")
        except Exception:
            pass
        if df.empty: return msg, []
        if "Doctor Name" in df.columns:
            df = df.rename(columns={"Doctor Name": "Doctor"})
        if specs:
            df = df[df["Specialization"].astype(str).isin(specs)]
        return msg, df[["Doctor","Specialization","Chamber","Visiting Time"]].fillna("").to_dict("records")
    def generate_slots(visiting_time_str: str):
        pat = r"(\d{1,2})[.:](\d{2})\s*([AaPp][Mm])\s*-\s*(\d{1,2})[.:](\d{2})\s*([AaPp][Mm])"
        m = re.search(pat, str(visiting_time_str))
        if not m: return ["11:00 AM","11:20 AM","11:40 AM","12:00 PM","12:20 PM","12:40 PM","01:00 PM"]
        h1,m1,ap1,h2,m2,ap2 = m.groups()
        fmt = "%I:%M %p"
        start = datetime.strptime(f"{int(h1)}:{int(m1)} {ap1.upper()}", fmt)
        end   = datetime.strptime(f"{int(h2)}:{int(m2)} {ap2.upper()}", fmt)
        if end <= start: end += timedelta(hours=12)
        out, cur = [], start
        while cur <= end:
            out.append(cur.strftime("%I:%M %p"))
            cur += timedelta(minutes=20)
        return out

# ------------------------ setup & paths ------------------------
os.makedirs("data", exist_ok=True)
st.set_page_config(page_title="Doctigo – AI Doctor & Bed/Cabin Booking", page_icon="🩺", layout="centered")
APPOINTMENTS_PATH = "appointments.csv"
INVENTORY_PATH    = "beds_inventory.csv"
WAITLIST_PATH     = "waitlist.csv"

# ------------------------ global CSS (match screenshot) ------------------------
st.markdown("""
<style>
:root{
  --bg:#0b0b0b; --panel:#101010; --bubble:#141414; --bubble-user:#191919;
  --ring:#232323; --text:#ececec; --muted:#9a9a9a; --accent:#ffd76a;
}
html,body,[class^="css"]{background:var(--bg) !important; color:var(--text) !important;}
/* Board */
.chat-shell{
  border:1px solid var(--ring); border-radius:16px; background:var(--panel);
  padding:14px 12px 70px 12px;  /* leave space for composer */
  position:relative; min-height:420px;
}
/* top right mini icons (screenshot vibe) */
.chat-shell .mini{
  position:absolute; top:8px; right:12px; color:var(--muted); font-size:12px;
}

/* bubbles */
.bbl{display:flex; gap:10px; align-items:flex-start; background:var(--bubble);
  border:1px solid var(--ring); border-radius:12px; padding:10px 12px; margin:10px 6px;}
.bbl.user{background:var(--bubble-user);}
.bbl .l{width:22px; text-align:center; font-size:16px; opacity:.95; margin-top:2px;}
.bbl .c{flex:1; font-size:15px; line-height:1.45}
.bbl .r{font-size:11px; color:var(--muted); white-space:nowrap; align-self:flex-start}

/* appointment card */
.round-card{border:1px solid var(--ring); border-radius:14px; padding:12px; background:#121212;}

/* composer */
.composer{
  position:sticky; bottom:0; left:0; right:0; padding:10px 0 0 0;
}
.comp-wrap{
  display:flex; align-items:center; gap:8px; border:1px solid var(--ring);
  background:#0f0f10; border-radius:12px; padding:6px 8px;
}
.comp-wrap input{
  flex:1; background:transparent; border:none; outline:none; color:var(--text);
}
.comp-btn{
  border:1px solid var(--ring); border-radius:10px; padding:6px 10px; background:#1d2430; color:#e8efff;
}
.comp-mic{
  border:1px solid var(--ring); border-radius:10px; padding:6px 10px; background:#131313; color:#cfcfcf;
}
.tag{display:inline-block;padding:2px 8px;border:1px solid var(--ring);border-radius:999px;color:var(--muted);font-size:12px;margin:4px 0 10px 4px;}
.small{color:var(--muted); font-size:12px;}
</style>
""", unsafe_allow_html=True)

# ------------------------ utils ------------------------
def pdfsafe(s) -> str:
    if s is None: return ""
    s = str(s)
    s = (s.replace("₹","Rs ").replace("–","-").replace("—","-").replace("•","*")
           .replace("“",'"').replace("”",'"').replace("’","'").replace("‘","'"))
    return s.encode("latin-1","replace").decode("latin-1")

@st.cache_data(show_spinner=False)
def load_hospitals():
    p = "hospitals.csv"
    if not os.path.exists(p): return pd.DataFrame()
    try: df = pd.read_csv(p)
    except Exception: return pd.DataFrame()
    need = {"Hospital","Address","Latitude","Longitude"}
    return df if need.issubset(df.columns) else pd.DataFrame()

@st.cache_data(show_spinner=False)
def load_doctors_file():
    path = "doctor.csv" if os.path.exists("doctor.csv") else ("doctors.csv" if os.path.exists("doctors.csv") else None)
    if not path: return pd.DataFrame()
    try: df = pd.read_csv(path)
    except Exception: return pd.DataFrame()
    if "Doctor Name" in df.columns: df = df.rename(columns={"Doctor Name":"Doctor"})
    need = {"Doctor","Specialization","Chamber","Visiting Time"}
    return df[sorted(list(need))].fillna("") if need.issubset(df.columns) else pd.DataFrame()

def haversine_km(lat1, lon1, lat2, lon2):
    R=6371.0; p1,p2=math.radians(lat1),math.radians(lat2)
    dphi=math.radians(lat2-lat1); dl=math.radians(lon2-lon1)
    a=math.sin(dphi/2)**2+math.cos(p1)*math.cos(p2)*math.sin(dl/2)**2
    return 2*R*math.atan2(math.sqrt(a),math.sqrt(1-a))

def travel_eta_minutes(distance_km, speed_kmph=None):
    if distance_km is None or distance_km<=0: return 5
    if speed_kmph and speed_kmph>0: return max(5, int(round((distance_km/float(speed_kmph))*60)))
    if distance_km<=5: return 15
    if distance_km<=10: return 20
    if distance_km<=20: return 30
    return 45

def detect_hospital_for_chamber(chamber_val: str, hospitals_df: pd.DataFrame) -> str|None:
    if hospitals_df.empty or not chamber_val: return None
    ch = str(chamber_val).casefold()
    for _,r in hospitals_df.iterrows():
        h = str(r["Hospital"])
        if h and h.casefold() in ch: return h
    return None

def get_hospital_coords(hospital_name: str, hospitals_df: pd.DataFrame):
    if not hospital_name or hospitals_df.empty: return None
    m = hospitals_df[hospitals_df["Hospital"].astype(str).str.casefold()==hospital_name.casefold()]
    if m.empty: return None
    row = m.iloc[0]
    try: return float(row["Latitude"]), float(row["Longitude"])
    except Exception: return None

def compute_distance_and_eta(user_loc: dict, hospital_name: str, hospitals_df: pd.DataFrame, speed_kmph=None):
    if not user_loc or user_loc.get("lat") is None or user_loc.get("lon") is None: return None,None,None,None
    coords = get_hospital_coords(hospital_name, hospitals_df)
    if not coords: return None,None,None,None
    d = haversine_km(user_loc["lat"], user_loc["lon"], coords[0], coords[1])
    return d, travel_eta_minutes(d, speed_kmph), coords[0], coords[1]

def parse_slot(slot_str: str):
    if not isinstance(slot_str, str): return "", ""
    m = re.match(r"^(.*?)\s+on\s+(.*)$", slot_str.strip())
    return (m.group(1).strip(), m.group(2).strip()) if m else (slot_str.strip(), "")

def _extract_slot_start_time_str(slot_label: str) -> str|None:
    if not slot_label: return None
    m=re.search(r'(\d{1,2}:\d{2}\s*[APap][Mm])',slot_label)
    if m: return m.group(1).upper().replace(" ","")
    m=re.search(r'\b(\d{1,2}:\d{2})\b',slot_label)
    return m.group(1) if m else None

def _slot_start_datetime(the_day: date, slot_label: str):
    s=_extract_slot_start_time_str(slot_label)
    if not s: return None
    try:
        if s.lower().endswith(("am","pm")) or ("AM" in s or "PM" in s):
            t=datetime.strptime(s.replace("AM"," AM").replace("PM"," PM"), "%I:%M %p").time()
        else: t=datetime.strptime(s, "%H:%M").time()
        return datetime.combine(the_day, t)
    except Exception: return None

def format_ampm(dt: datetime) -> str: return dt.strftime("%I:%M %p").lstrip("0")

# ------------------------ Appointments CSV ------------------------
APPT_HEADERS = [
    "Patient Name","Symptoms","Doctor","Chamber","Slot","Timestamp","Date","SlotTime",
    "Serial","DistanceKm","ETAmin","UserLat","UserLon","Hospital","HospitalLat","HospitalLon","AvgSpeedKmph",
    "Phone","Email"
]
def ensure_appointments_file():
    if not os.path.exists(APPOINTMENTS_PATH) or os.stat(APPOINTMENTS_PATH).st_size==0:
        with open(APPOINTMENTS_PATH,"w",newline="",encoding="utf-8") as f:
            csv.writer(f).writerow(APPT_HEADERS)

@st.cache_data(show_spinner=False)
def load_appointments_df():
    ensure_appointments_file()
    try: df = pd.read_csv(APPOINTMENTS_PATH)
    except Exception: return pd.DataFrame(columns=APPT_HEADERS)
    for c in APPT_HEADERS:
        if c not in df.columns: df[c]=""
    for i,row in df.iterrows():
        if not row.get("Date") or not row.get("SlotTime"):
            stime,sdate = parse_slot(str(row.get("Slot","")))
            df.at[i,"Date"]=sdate; df.at[i,"SlotTime"]=stime
    return df

def booked_slot_times_for(doctor_name: str, day: date) -> set:
    df = load_appointments_df()
    if df.empty: return set()
    day_str = day.strftime("%d %B %Y")
    sub = df[(df["Doctor"].astype(str)==str(doctor_name))&(df["Date"]==day_str)]
    return set(sub["SlotTime"].astype(str).tolist())

def compute_serial(slot_time: str, visiting_time_str: str) -> int:
    try: return generate_slots(visiting_time_str).index(slot_time)+1
    except Exception: return 1

def save_appointment_row(patient_name, symptoms, doctor, chamber, slot_full, visiting_time_str,
                         distance_km=None, eta_min=None, user_loc=None, hospital=None,
                         hosp_lat=None, hosp_lon=None, avg_speed=None, phone="", email=""):
    ensure_appointments_file()
    slot_time, date_str = parse_slot(slot_full)
    if not doctor or not slot_time or not date_str:
        return False, "Invalid slot or doctor.", None
    df = load_appointments_df()
    dup = df[(df["Doctor"].astype(str)==str(doctor))&(df["Date"]==date_str)&(df["SlotTime"]==slot_time)]
    if not dup.empty: return False, "That time frame is already booked for this doctor and date. Please choose another slot.", None
    serial = compute_serial(slot_time, visiting_time_str)
    new_row = {
        "Patient Name": patient_name or "", "Symptoms": symptoms or "", "Doctor": doctor,
        "Chamber": chamber or "", "Slot": slot_full, "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Date": date_str, "SlotTime": slot_time, "Serial": serial,
        "DistanceKm": round(distance_km,2) if isinstance(distance_km,(int,float)) else "",
        "ETAmin": int(eta_min) if isinstance(eta_min,(int,float)) else "",
        "UserLat": (user_loc or {}).get("lat",""), "UserLon": (user_loc or {}).get("lon",""),
        "Hospital": hospital or "", "HospitalLat": hosp_lat if hosp_lat is not None else "",
        "HospitalLon": hosp_lon if hosp_lon is not None else "", "AvgSpeedKmph": int(avg_speed) if isinstance(avg_speed,(int,float)) else "",
        "Phone": phone or "", "Email": email or ""
    }
    first = not (os.path.exists(APPOINTMENTS_PATH) and os.stat(APPOINTMENTS_PATH).st_size>0)
    with open(APPOINTMENTS_PATH,"a",newline="",encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=APPT_HEADERS)
        if first: w.writeheader()
        w.writerow(new_row)
    load_appointments_df.clear()
    return True, "✅ Appointment booked and saved.", serial

def update_appointment_contact(patient_name: str, doctor: str, slot_full: str, phone: str, email: str):
    ensure_appointments_file()
    try: df = pd.read_csv(APPOINTMENTS_PATH)
    except Exception: return
    if df.empty: return
    mask = (
        (df["Patient Name"].astype(str)==str(patient_name)) &
        (df["Doctor"].astype(str)==str(doctor)) &
        (df["Slot"].astype(str)==str(slot_full))
    )
    idxs = df[mask].index.tolist()
    if not idxs: return
    i = idxs[-1]
    df.at[i,"Phone"]=phone or ""; df.at[i,"Email"]=email or ""
    df.to_csv(APPOINTMENTS_PATH,index=False); load_appointments_df.clear()

# ------------------------ static mini-map ------------------------
def render_static_map(user_lat, user_lon, hosp_lat, hosp_lon):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from io import BytesIO
    ulat,ulon=float(user_lat),float(user_lon); hlat,hlon=float(hosp_lat),float(hosp_lon)
    lats=[ulat,hlat]; lons=[ulon,hlon]
    pad_lat=max(0.01,(max(lats)-min(lats))*0.2); pad_lon=max(0.01,(max(lons)-min(lons))*0.2)
    fig,ax=plt.subplots(figsize=(4,4),dpi=150)
    ax.set_xlim(min(lons)-pad_lon,max(lons)+pad_lon); ax.set_ylim(min(lats)-pad_lat,max(lats)+pad_lat)
    ax.plot([ulon,hlon],[ulat,hlat],linewidth=1.5)
    ax.scatter([ulon],[ulat],s=35,label="You",zorder=3); ax.scatter([hlon],[hlat],s=55,marker="s",label="Hospital",zorder=3)
    ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude"); ax.set_title("Approx path (straight-line)")
    ax.grid(True,alpha=.3); ax.legend(loc="upper left",frameon=True)
    buf=BytesIO(); fig.tight_layout(); fig.savefig(buf,format="png"); plt.close(fig); buf.seek(0); return buf

# ------------------------ PDF ------------------------
def _fmt_coords(lat, lon):
    try:
        if lat in (None,"") or lon in (None,""): return ""
        return f"{float(lat):.6f}, {float(lon):.6f}"
    except Exception: return ""

def generate_full_pdf(hospital_name, patient, appointment, bed_choice):
    mini_map_path = None
    try:
        if appointment and all(appointment.get(k) not in (None,"") for k in ("user_lat","user_lon","hosp_lat","hosp_lon")):
            buf = render_static_map(appointment["user_lat"], appointment["user_lon"], appointment["hosp_lat"], appointment["hosp_lon"])
            mini_map_path = os.path.join("data", "doctigo_mini_map.png")
            with open(mini_map_path,"wb") as f: f.write(buf.getbuffer())
    except Exception: mini_map_path = None

    pdf=FPDF(); pdf.add_page()
    pdf.set_font("Arial","B",18); pdf.cell(0,10,pdfsafe(hospital_name or "Doctigo"),ln=True,align='C')
    pdf.set_font("Arial","B",12); pdf.ln(3); pdf.cell(0,8,pdfsafe("Booking Summary"),ln=True,align='C')
    pdf.set_font("Arial","",11); pdf.cell(0,5,pdfsafe("----------------------------------------"),ln=True,align='C')

    pdf.ln(4); pdf.set_font("Arial","B",12); pdf.cell(0,8,pdfsafe("Patient Details"),ln=True); pdf.set_font("Arial","",11)
    details=f"""Patient Name: {patient.get('name','')}
Phone: {patient.get('phone','')}
Gender: {patient.get('gender','')}
Age: {patient.get('age','')}
Email: {patient.get('email','')}
Address: {patient.get('address','')}
Issued On: {datetime.now().strftime('%d %B %Y, %I:%M %p')}"""
    pdf.multi_cell(0,7,pdfsafe(details))

    if appointment:
        pdf.ln(3); pdf.set_font("Arial","B",12); pdf.cell(0,8,pdfsafe("Doctor Appointment"),ln=True); pdf.set_font("Arial","",11)
        serial_text = f"\nSerial #: {appointment.get('serial','')}" if appointment.get("serial") else ""
        dist_text=""
        if appointment.get("distance_km") not in (None,"") and appointment.get("eta_min") not in (None,""):
            dist_text=(f"\nDistance: {float(appointment['distance_km']):.2f} km"
                       f" | ETA: {int(appointment['eta_min'])} min"
                       f" | Avg speed: {appointment.get('avg_speed','—')} km/h"
                       f"\nFrom (you): {_fmt_coords(appointment.get('user_lat'), appointment.get('user_lon'))}"
                       f"\nTo (hospital): {_fmt_coords(appointment.get('hosp_lat'), appointment.get('hosp_lon'))}")
        ap_text=f"""Doctor: Dr. {appointment.get('doctor','')}
Chamber: {appointment.get('chamber','')}
Slot: {appointment.get('slot','')}{serial_text}{dist_text}
Symptoms: {appointment.get('symptoms','')}"""
        pdf.multi_cell(0,7,pdfsafe(ap_text))
        if mini_map_path and os.path.exists(mini_map_path):
            pdf.ln(2); pdf.set_font("Arial","B",12); pdf.cell(0,8,pdfsafe("Map Preview"),ln=True)
            try: pdf.image(mini_map_path, w=100)
            except Exception: pass

    if bed_choice:
        pdf.ln(3); pdf.set_font("Arial","B",12); pdf.cell(0,8,pdfsafe("Bed/Cabin Booking"),ln=True); pdf.set_font("Arial","",11)
        features_text = ", ".join(bed_choice.get('features', []))
        bed_text=f"""Type: {bed_choice.get('tier','')}
Check-in Date: {bed_choice.get('checkin_date','')}
Check-out Date: {bed_choice.get('checkout_date','') or 'To be decided'}
Unit ID: {bed_choice.get('unit_id','Any')}
Price per night: Rs {bed_choice.get('price','')}
Features: {features_text}"""
        pdf.multi_cell(0,7,pdfsafe(bed_text))

    pdf.ln(6); pdf.set_font("Arial","I",9); pdf.set_text_color(120)
    pdf.cell(0,8,pdfsafe("This receipt is auto-generated by Doctigo AI System."),ln=True,align="C")
    pdf.set_text_color(0)
    return io.BytesIO(pdf.output(dest="S").encode("latin-1"))

# ------------------------ Bed inventory & waitlist (unchanged logic) ------------------------
def _tier_ids(): return {"General Bed":[f"G-{i}" for i in range(1,41)],
                         "General Cabin":[f"C-{i}" for i in range(1,21)],
                         "VIP Cabin":[f"V-{i}" for i in range(1,11)]}
def ensure_inventory():
    if not os.path.exists(INVENTORY_PATH) or os.stat(INVENTORY_PATH).st_size==0:
        with open(INVENTORY_PATH,"w",newline="",encoding="utf-8") as f:
            csv.writer(f).writerow(["Hospital","Tier","UnitID","Date","Status"])
def _ensure_rows_for(hospital:str,tier:str,day:date):
    ensure_inventory()
    try: df=pd.read_csv(INVENTORY_PATH)
    except Exception: df=pd.DataFrame(columns=["Hospital","Tier","UnitID","Date","Status"])
    ids=_tier_ids()[tier]; day_str=day.strftime("%Y-%m-%d")
    mask=(df["Hospital"].astype(str)==hospital)&(df["Tier"]==tier)&(df["Date"]==day_str)
    existing=set(df.loc[mask,"UnitID"].astype(str).tolist())
    missing=[uid for uid in ids if uid not in existing]
    if missing:
        add=pd.DataFrame({"Hospital":hospital,"Tier":tier,"UnitID":missing,"Date":day_str,"Status":"available"})
        pd.concat([df,add],ignore_index=True).to_csv(INVENTORY_PATH,index=False)
def get_inventory(hospital:str,tier:str,day:date)->pd.DataFrame:
    ensure_inventory(); _ensure_rows_for(hospital,tier,day)
    try: df=pd.read_csv(INVENTORY_PATH)
    except Exception: return pd.DataFrame(columns=["Hospital","Tier","UnitID","Date","Status"])
    return df[(df["Hospital"].astype(str)==hospital)&(df["Tier"]==tier)&(df["Date"]==day.strftime("%Y-%m-%d"))].copy()
def mark_booked(hospital:str,tier:str,unit_id:str,day:date):
    ensure_inventory(); _ensure_rows_for(hospital,tier,day); df=pd.read_csv(INVENTORY_PATH)
    mask=((df["Hospital"].astype(str)==hospital)&(df["Tier"]==tier)&(df["UnitID"]==unit_id)&(df["Date"]==day.strftime("%Y-%m-%d")))
    if mask.any(): df.loc[mask,"Status"]="booked"; df.to_csv(INVENTORY_PATH,index=False)
def mark_booked_range(hospital:str,tier:str,unit_id:str,start_day:date,end_day:date|None):
    last=end_day or start_day; cur=start_day
    while cur<=last: mark_booked(hospital,tier,unit_id,cur); cur+=timedelta(days=1)
def reset_inventory_for_date(hospital:str,tier:str,day:date):
    ensure_inventory(); _ensure_rows_for(hospital,tier,day); df=pd.read_csv(INVENTORY_PATH)
    mask=(df["Hospital"].astype(str)==hospital)&(df["Tier"]==tier)&(df["Date"]==day.strftime("%Y-%m-%d"))
    df.loc[mask,"Status"]="available"; df.to_csv(INVENTORY_PATH,index=False)
def units_available_for_range(hospital:str,tier:str,start_day:date,end_day:date)->list[str]:
    ensure_inventory(); 
    if end_day<start_day: start_day,end_day=end_day,start_day
    sets=[]; cur=start_day
    while cur<=end_day:
        _ensure_rows_for(hospital,tier,cur); inv=get_inventory(hospital,tier,cur)
        sets.append(set(inv[inv["Status"]!="booked"]["UnitID"].astype(str).tolist()))
        cur+=timedelta(days=1)
    return sorted(set.intersection(*sets)) if sets else []

# ------------------------ Session ------------------------
if "flow_step" not in st.session_state: st.session_state.flow_step="home"
defaults = {
    "mode":None,"chat_stage":None,"user_name":"", "user_location":{"lat":None,"lon":None},
    "symptoms_selected":[],"symptoms_typed":"","recommendations":[], "chosen_doctor":None,
    "chosen_date":None,"chosen_slot":"","appointment":None,"selected_hospital":None,
    "need_bed":None,"bed_choice":None,"details_step":0,
    "patient_details":{"name":"","phone":"","gender":"","age":"","email":"","address":""},
    "doctor_message":"","avg_speed":25,"admin_logged_in":False,
}
for k,v in defaults.items():
    if k not in st.session_state: st.session_state[k]=v

# ------------------------ Chat helpers ------------------------
def _now_ts(): return datetime.now().strftime("%I:%M %p").lstrip("0")
def bubble_bot(msg:str):
    st.markdown(f'<div class="bbl"><div class="l">🤖</div><div class="c">{msg}</div><div class="r">{_now_ts()}</div></div>', unsafe_allow_html=True)
def bubble_user(msg:str):
    st.markdown(f'<div class="bbl user"><div class="l">👤</div><div class="c">{msg}</div><div class="r">{_now_ts()}</div></div>', unsafe_allow_html=True)

def geo_widget(lat_key: str, lon_key: str, label="📍 Detect My Location"):
    components.html(f"""
    <button onclick="getLoc()" style="padding:8px 14px;border-radius:10px;border:1px solid #2a2a2a;background:#181818;color:#eaeaea;">{label}</button>
    <p id="locout" style="margin-top:8px;color:#bdbdbd;"></p>
    <script>
    function getLoc(){{
      if(navigator.geolocation){{
        navigator.geolocation.getCurrentPosition(function(pos){{
          const lat=pos.coords.latitude.toFixed(6); const lon=pos.coords.longitude.toFixed(6);
          document.getElementById('locout').innerText="✓ Location captured: "+lat+", "+lon;
          const inputs=window.parent.document.querySelectorAll('input');
          for(let i=0;i<inputs.length;i++){{const el=inputs[i]; const a=(el.getAttribute('aria-label')||'').toLowerCase();
            if(a.includes('{lat_key.lower()}')){{el.value=lat; el.dispatchEvent(new Event('input',{{bubbles:true}}));}}
            if(a.includes('{lon_key.lower()}')){{el.value=lon; el.dispatchEvent(new Event('input',{{bubbles:true}}));}}
          }}
        }},function(err){{document.getElementById('locout').innerText="❌ "+err.message;}});}} else {{
        document.getElementById('locout').innerText="❌ Geolocation not supported.";
      }}
    }}</script>
    """, height=120)

# Single bottom composer (works like screenshot). Returns submitted text or None.
def composer(key="composer", placeholder="Type your message..."):
    st.markdown('<div class="composer">', unsafe_allow_html=True)
    c1,c2,c3 = st.columns([0.1,0.8,0.1])
    with c1: st.markdown('<div class="small"> </div>', unsafe_allow_html=True)
    with c2:
        txt = st.text_input("", key=key, placeholder=placeholder, label_visibility="collapsed")
    with c3:
        sent = st.button("➤", key=f"{key}_send", help="Send", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
    return txt.strip() if sent and txt.strip() else None

# ------------------------ Header & entry buttons ------------------------
st.markdown('<div class="tag">Your AI-powered medical booking assistant</div>', unsafe_allow_html=True)
wrap = st.container()
with wrap:
    st.markdown('<div class="chat-shell"><div class="mini">🕒</div>', unsafe_allow_html=True)

    # Entry buttons (outside composer)
    if st.session_state.flow_step=="home":
        c1,c2 = st.columns(2)
        with c1:
            if st.button("🔎 Normal Booking"): st.session_state.update({"flow_step":"chat","mode":"normal","chat_stage":"greet"})
        with c2:
            if st.button("🚨 Emergency Booking"): st.session_state.update({"flow_step":"chat","mode":"emergency","chat_stage":"greet"})

    # ------------------------ CHAT FLOW ------------------------
    if st.session_state.flow_step=="chat":

        # GREET
        if st.session_state.chat_stage=="greet":
            bubble_bot("Hello! I am **Doc**, your friendly neighborhood **Spider Doc** 🕷️🩺. What's your name?")
            # hidden fields to capture optional location early
            lat_in = st.text_input("Latitude (greet)", value=str(st.session_state.user_location["lat"] or ""), key="lat_g", label_visibility="collapsed")
            lon_in = st.text_input("Longitude (greet)", value=str(st.session_state.user_location["lon"] or ""), key="lon_g", label_visibility="collapsed")
            geo_widget("Latitude (greet)","Longitude (greet)")
            msg = composer(key="comp_greet")
            if msg:
                bubble_user(msg)
                st.session_state.user_name = msg
                # keep any typed location
                try:
                    st.session_state.user_location["lat"] = float(lat_in) if lat_in else None
                    st.session_state.user_location["lon"] = float(lon_in) if lon_in else None
                except Exception: pass
                choice = "normal booking" if st.session_state.mode=="normal" else "emergency booking"
                bubble_bot(f"Hello **{st.session_state.user_name}** — so you opted for **{choice}**.")
                st.session_state.chat_stage="symptoms"
                st.rerun()

        # SYMPTOMS
        elif st.session_state.chat_stage=="symptoms":
            if st.session_state.mode=="normal":
                bubble_bot("Enter your symptoms **manually** or type **Next** to skip.")
            else:
                bubble_bot("It's an **EMERGENCY**. Enter symptoms or type **Next** to pick any doctor.")
            msg = composer(key="comp_sym", placeholder="e.g., fever, cough  |  or type: Next")
            if msg:
                bubble_user(msg)
                if msg.strip().lower()=="next":
                    st.session_state.symptoms_selected=[]
                else:
                    st.session_state.symptoms_selected = [s.strip().lower() for s in msg.split(",") if s.strip()]
                # compute recommendations
                docs = load_doctors_file()
                if not st.session_state.symptoms_selected:
                    bubble_bot("Select a doctor of your preference:")
                    st.session_state.recommendations = (docs[["Doctor","Specialization","Chamber","Visiting Time"]].to_dict("records")
                                                        if not docs.empty else [])
                    st.session_state.doctor_message = "Here are available doctors:"
                else:
                    msgx,recs = recommend_doctors(st.session_state.symptoms_selected)
                    st.session_state.recommendations = recs or []
                    st.session_state.doctor_message = msgx or "Recommended doctors:"
                st.session_state.chat_stage="doctor"; st.rerun()

        # DOCTOR + DATE + SLOT
        elif st.session_state.chat_stage=="doctor":
            bubble_bot(st.session_state.doctor_message or "Recommended doctors:")
            doctors_df = load_doctors_file()
            if not st.session_state.recommendations:
                st.session_state.recommendations = (doctors_df[["Doctor","Specialization","Chamber","Visiting Time"]]
                                                     .to_dict("records")) if not doctors_df.empty else []
            st.session_state.avg_speed = st.select_slider("Assumed average city speed (km/h) for ETA",
                                                          options=[15,20,25,30,35,40,45,50],
                                                          value=int(st.session_state.get("avg_speed",25)))
            hospitals_df = load_hospitals()
            labels, mapping = [], []
            for d in st.session_state.recommendations:
                label = f"Dr. {d.get('Doctor','')} — {d.get('Specialization','')}"
                if not hospitals_df.empty and st.session_state.user_location.get("lat") is not None:
                    hosp = detect_hospital_for_chamber(d.get("Chamber",""), hospitals_df)
                    if hosp:
                        dist,eta,_,_ = compute_distance_and_eta(st.session_state.user_location, hosp, hospitals_df,
                                                                speed_kmph=st.session_state.avg_speed)
                        if dist is not None: label += f"  •  ~{dist:.1f} km / {eta} min to {hosp}"
                labels.append(label); mapping.append(d)
            sel = st.selectbox("Choose a doctor", options=labels)
            idx = labels.index(sel) if sel in labels else 0
            chosen = mapping[idx]
            st.write(f"**Chamber:** {chosen.get('Chamber','')}"); st.write(f"**Visiting Time:** {chosen.get('Visiting Time','')}")

            # hospital detect & date/slot
            hosp_df = load_hospitals()
            hosp_name = detect_hospital_for_chamber(chosen.get("Chamber",""), hosp_df) if not hosp_df.empty else None
            if hosp_name: st.session_state.selected_hospital = hosp_name; st.info(f"🏥 Hospital detected: **{hosp_name}**")
            the_day = st.date_input("Choose appointment date", min_value=date.today(),
                                    value=st.session_state.chosen_date or date.today())
            slots = generate_slots(chosen.get("Visiting Time",""))
            # remove already booked & past today
            already = booked_slot_times_for(chosen.get("Doctor",""), the_day)
            if the_day==date.today():
                now = datetime.now()
                slots=[s for s in slots if (_slot_start_datetime(the_day,s) or now+timedelta(hours=1))>now]
            slots=[s for s in slots if s not in already]
            if not slots: st.warning("No future slots available on this date.")
            else:
                slot = st.selectbox("Select a slot", options=slots, key="slot_pick")
                # location (hidden fields)
                lat_val = st.text_input("Latitude (doctor)", value=str(st.session_state.user_location["lat"] or ""),
                                        key="lat_doc", label_visibility="collapsed")
                lon_val = st.text_input("Longitude (doctor)", value=str(st.session_state.user_location["lon"] or ""),
                                        key="lon_doc", label_visibility="collapsed")
                st.caption("Tip: use Detect My Location to auto-fill:"); geo_widget("Latitude (doctor)","Longitude (doctor)")
                try:
                    st.session_state.user_location["lat"] = float(lat_val) if lat_val else None
                    st.session_state.user_location["lon"] = float(lon_val) if lon_val else None
                except Exception: pass

                dist_km, eta_min, hosp_lat, hosp_lon = None,None,None,None
                if st.session_state.selected_hospital:
                    dist_km,eta_min,hosp_lat,hosp_lon = compute_distance_and_eta(
                        st.session_state.user_location, st.session_state.selected_hospital, hosp_df, speed_kmph=st.session_state.avg_speed
                    )
                serial = compute_serial(slot, chosen.get("Visiting Time",""))
                st.info(f"**Serial #{serial}**" + (f"  |  **Distance:** {dist_km:.2f} km  |  **ETA:** ~{eta_min} min (avg {st.session_state.avg_speed} km/h)"
                                                   if dist_km is not None else "  |  Enable location to compute distance/ETA."))
                if st.button("🗓 Book This Appointment"):
                    full_slot = f"{slot} on {the_day.strftime('%d %B %Y')}"
                    ok,msg,serial_saved = save_appointment_row(
                        patient_name=st.session_state.user_name,
                        symptoms='; '.join(st.session_state.symptoms_selected) if st.session_state.symptoms_selected else (st.session_state.mode.title()+" Flow"),
                        doctor=chosen.get("Doctor",""), chamber=chosen.get("Chamber",""),
                        slot_full=full_slot, visiting_time_str=chosen.get("Visiting Time",""),
                        distance_km=dist_km, eta_min=eta_min, user_loc=st.session_state.user_location,
                        hospital=st.session_state.selected_hospital, hosp_lat=hosp_lat, hosp_lon=hosp_lon,
                        avg_speed=st.session_state.avg_speed,
                        phone=st.session_state.patient_details.get("phone",""), email=st.session_state.patient_details.get("email","")
                    )
                    if ok:
                        st.session_state.chosen_doctor=chosen; st.session_state.chosen_date=the_day; st.session_state.chosen_slot=slot
                        st.session_state.appointment={"doctor":chosen.get("Doctor",""),"chamber":chosen.get("Chamber",""),
                            "slot":full_slot,"symptoms":'; '.join(st.session_state.symptoms_selected) if st.session_state.symptoms_selected else "",
                            "serial":serial_saved,"distance_km":dist_km,"eta_min":eta_min,
                            "user_lat":st.session_state.user_location.get("lat"),"user_lon":st.session_state.user_location.get("lon"),
                            "hospital":st.session_state.selected_hospital,"hosp_lat":hosp_lat,"hosp_lon":hosp_lon,"avg_speed":st.session_state.avg_speed}
                        st.success(f"✅ Appointment booked. Serial #{serial_saved}."); st.session_state.chat_stage="bed_ask"; st.rerun()
                    else: st.error(msg)

        # BED ASK (composer yes/no)
        elif st.session_state.chat_stage=="bed_ask":
            bubble_bot('Do u wanna book a **Bed or Cabin**? If **Yes** type Yes, if **No** type No.')
            msg = composer(key="comp_bed", placeholder="Yes / No")
            if msg:
                bubble_user(msg)
                v = msg.lower()
                if v not in ("yes","no"): bubble_bot("Please type **Yes** or **No**.")
                else:
                    st.session_state.need_bed = (v=="yes")
                    st.session_state.chat_stage = "bed_select" if st.session_state.need_bed else "details"
                    st.rerun()

        # BED SELECT (structured)
        elif st.session_state.chat_stage=="bed_select":
            hospital_name_for_beds = st.session_state.selected_hospital or "Doctigo Partner Hospital"
            bubble_bot(f"Great. Let's pick your **Bed/Cabin** at **{hospital_name_for_beds}**.")
            checkin = st.date_input("Choose check-in date", value=st.session_state.chosen_date or date.today(), key="bed_checkin")
            unknown = st.checkbox("I don't know the check-out date yet", value=False, key="bed_unknown")
            checkout = None if unknown else st.date_input("Choose check-out date", value=checkin, min_value=checkin, key="bed_checkout")
            tiers=[{"tier":"General Bed","price":100,"features":["1 bed","1 chair","bed table"]},
                   {"tier":"General Cabin","price":1000,"features":["2 beds","attached washroom","bed table","chair","food x3 times"]},
                   {"tier":"VIP Cabin","price":4000,"features":["premium bed x2","sofa","AC","attached washroom","TV","fridge","bed table x2","coffee table","2 chairs"]}]
            pick_tier = st.selectbox("Select type", options=[t["tier"] for t in tiers])
            tier_obj = next(t for t in tiers if t["tier"]==pick_tier)
            start_day, end_day = checkin, (checkout if checkout else checkin)
            avail_units = units_available_for_range(hospital_name_for_beds, tier_obj["tier"], start_day, end_day)
            unit_id = st.selectbox("Select a unit", options=(avail_units or ["None"]))
            if st.button("Save Bed/Cabin ➞"):
                st.session_state.bed_choice = None if not avail_units else {
                    "tier": tier_obj["tier"], "unit_id": unit_id, "price": tier_obj["price"],
                    "features": tier_obj["features"], "checkin_date": checkin.strftime("%d %B %Y"),
                    "checkout_date": checkout.strftime("%d %B %Y") if checkout else "",
                }
                st.success("Bed/Cabin selection saved.")
            # advance via composer "Next"
            msg = composer(key="comp_bednext", placeholder='Type "Next" to continue')
            if msg and msg.lower()=="next":
                st.session_state.chat_stage="details"; st.rerun()

        # DETAILS (composer-driven sequence)
        elif st.session_state.chat_stage=="details":
            steps=[("name","Please Enter Patient's Name"),
                   ("phone","Please Enter Phone Number"),
                   ("gender","Please Enter Gender (Male/Female/Other)"),
                   ("age","Please Enter Age (number)"),
                   ("email","Please Enter Email address"),
                   ("address","Please Enter Address")]
            field,prompt = steps[st.session_state.details_step]
            bubble_bot(prompt)
            msg = composer(key=f"comp_det_{field}")
            if msg:
                bubble_user(msg)
                if field=="age":
                    try: st.session_state.patient_details[field]=int(float(msg))
                    except Exception: st.session_state.patient_details[field]=msg
                else:
                    st.session_state.patient_details[field]=msg.strip()
                st.session_state.details_step+=1
                if st.session_state.details_step>=len(steps):
                    # backfill to CSV
                    ap = st.session_state.get("appointment") or {}
                    if ap:
                        update_appointment_contact(
                            patient_name=st.session_state.get("user_name",""),
                            doctor=ap.get("doctor",""),
                            slot_full=ap.get("slot",""),
                            phone=st.session_state.patient_details.get("phone",""),
                            email=st.session_state.patient_details.get("email","")
                        )
                    st.session_state.chat_stage="card"
                st.rerun()

        # CARD + PDF + FINAL DEPARTURE TIP
        elif st.session_state.chat_stage=="card":
            ap = st.session_state.appointment or {}
            bc = st.session_state.bed_choice
            pdets = st.session_state.patient_details
            bubble_bot("Here is your **appointment card** with all details:")
            st.markdown('<div class="round-card">', unsafe_allow_html=True)
            st.markdown("#### 🏥 Appointment Summary")
            if ap:
                extra=""
                if ap.get("serial"): extra+=f"  \n**Serial #:** {ap['serial']}"
                if ap.get("distance_km") not in (None,"") and ap.get("eta_min") not in (None,""):
                    extra+=f"  \n**Distance/ETA:** {ap['distance_km']:.2f} km / {ap['eta_min']} min (avg {ap.get('avg_speed','—')} km/h)"
                st.markdown(f"**Doctor:** Dr. {ap.get('doctor','')}  \n**Chamber:** {ap.get('chamber','')}  \n**Slot:** {ap.get('slot','')}{extra}")
            else:
                st.info("No appointment found.")
            st.markdown("#### 👤 Patient")
            st.markdown(f"**Name:** {pdets.get('name','')}  \n**Phone:** {pdets.get('phone','')}  \n**Gender:** {pdets.get('gender','')}  \n**Age:** {pdets.get('age','')}  \n**Email:** {pdets.get('email','')}  \n**Address:** {pdets.get('address','')}")
            if bc:
                st.markdown("#### 🛏️ Bed/Cabin")
                st.markdown(f"**Type:** {bc['tier']} (₹{bc['price']}/night)  \n**Unit ID:** {bc.get('unit_id','Any')}  \n**Check-in:** {bc.get('checkin_date','')}  \n**Check-out:** {bc.get('checkout_date','') or 'To be decided'}  \n**Features:** {', '.join(bc['features'])}")
            st.markdown('</div>', unsafe_allow_html=True)

            # enrich for PDF
            if ap:
                ap.setdefault("user_lat", st.session_state.user_location.get("lat"))
                ap.setdefault("user_lon", st.session_state.user_location.get("lon"))
                ap.setdefault("hospital", st.session_state.selected_hospital)
                if ap.get("hosp_lat") in (None,"") or ap.get("hosp_lon") in (None,""):
                    hdf = load_hospitals()
                    if st.session_state.selected_hospital and not hdf.empty:
                        c = get_hospital_coords(st.session_state.selected_hospital, hdf)
                        if c: ap["hosp_lat"], ap["hosp_lon"] = c[0], c[1]
                ap.setdefault("avg_speed", st.session_state.avg_speed)

            pdf_buf = generate_full_pdf(
                hospital_name=st.session_state.selected_hospital or "Doctigo Partner Hospital",
                patient=pdets, appointment=ap, bed_choice=bc
            )
            st.download_button("⬇️ Download Appointment Card (PDF)", data=pdf_buf, file_name="doctigo_appointment_card.pdf", mime="application/pdf")

            # Bed reserve button
            if bc and bc.get("unit_id"):
                start_day = datetime.strptime(bc["checkin_date"], "%d %B %Y").date() if bc.get("checkin_date") else date.today()
                end_day   = datetime.strptime(bc["checkout_date"], "%d %B %Y").date() if bc.get("checkout_date") else start_day
                if st.button("✅ Confirm & Reserve Selected Bed/Cabin"):
                    mark_booked_range(st.session_state.selected_hospital or "Doctigo Partner Hospital",
                                      bc["tier"], bc["unit_id"], start_day, end_day)
                    st.success("Bed/Cabin reserved for the selected date" + ("" if start_day==end_day else f" range ({start_day.strftime('%d %b %Y')} → {end_day.strftime('%d %b %Y')})."))

            # Final departure tip
            if ap:
                slot_time_str, slot_date_str = parse_slot(ap.get("slot",""))
                try: the_day = datetime.strptime(slot_date_str, "%d %B %Y").date()
                except Exception: the_day=None
                if the_day:
                    start_dt=_slot_start_datetime(the_day, slot_time_str)
                    if start_dt:
                        leave30 = start_dt - timedelta(minutes=30)
                        leave60 = start_dt - timedelta(minutes=60)
                        bubble_bot(f'Set off at **{format_ampm(leave30)}** to reach in time, but I recommend setting off **1 hour early** at **{format_ampm(leave60)}**.')

            # back to home via composer
            msg = composer(key="comp_done", placeholder='Type "Home" to go back')
            if msg and msg.lower()=="home":
                for k in ["mode","chat_stage","user_name","symptoms_selected","symptoms_typed","recommendations",
                          "chosen_doctor","chosen_date","chosen_slot","appointment","need_bed","bed_choice",
                          "details_step","patient_details","doctor_message"]:
                    if k in st.session_state: del st.session_state[k]
                st.session_state.flow_step="home"; st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)  # end chat-shell

# ------------------------ Admin (unchanged core) ------------------------
ADMIN_PASSWORD = "admin123"
st.header("🔒 Admin Dashboard Login")
with st.expander("Login as Admin"):
    entered = st.text_input("Enter admin password", type="password", key="admin_pass_input")
    if st.button("Login", key="admin_login_btn"):
        st.session_state.admin_logged_in = (entered==ADMIN_PASSWORD)
        st.success("✅ Access granted.") if st.session_state.admin_logged_in else st.error("❌ Incorrect password")

if st.session_state.get("admin_logged_in", False):
    st.header("📊 Admin Dashboard - All Appointments")
    uploaded = st.file_uploader("Upload Past Appointments CSV (Optional)", type=["csv"])
    if uploaded is not None:
        with open("data/appointments_past.csv","wb") as f: f.write(uploaded.getbuffer())
        st.success("✅ Past appointment file saved successfully!")

    dfs=[]
    if os.path.exists("data/appointments_past.csv"):
        try:
            past=pd.read_csv("data/appointments_past.csv")
            if "Slot" not in past.columns and "Visiting Time" in past.columns and "Appointment Date" in past.columns:
                past["Slot"]=past["Visiting Time"]+" on "+past["Appointment Date"]
            if "Doctor Name" in past.columns: past.rename(columns={"Doctor Name":"Doctor"},inplace=True)
            dfs.append(past)
        except Exception: pass
    if os.path.exists(APPOINTMENTS_PATH):
        try:
            cur=pd.read_csv(APPOINTMENTS_PATH)
            for c in APPT_HEADERS:
                if c not in cur.columns: cur[c]=""
            dfs.append(cur)
        except Exception: pass

    if dfs:
        df=pd.concat(dfs,ignore_index=True)
        st.subheader(f"📄 Appointments Done Till Now: **{len(df)}**")
        doc_filter = st.selectbox("Select Doctor:", ["All"] + sorted([d for d in df["Doctor"].dropna().unique().tolist() if d!=""]))
        view = df if doc_filter=="All" else df[df["Doctor"]==doc_filter]
        st.dataframe(view, use_container_width=True)

        q = st.text_input("Search by Patient Name:")
        if q: st.dataframe(view[view["Patient Name"].astype(str).str.contains(q, case=False, na=False)], use_container_width=True)

        try:
            view=view.copy()
            if "Date" in view.columns:
                parsed=pd.to_datetime(view["Date"], errors="coerce")
                view=view.assign(ParsedDate=parsed)
                today=pd.to_datetime(datetime.today().date())
                upcoming=view[view['ParsedDate'].between(today, today+pd.Timedelta(days=3), inclusive="both")]
            else:
                view['Parsed Date']=pd.to_datetime(view['Slot'].str.extract(r'on (.+)$')[0], errors='coerce')
                today=pd.to_datetime(datetime.today().date())
                upcoming=view[view['Parsed Date'].between(today,today+pd.Timedelta(days=3))]
        except Exception:
            upcoming=pd.DataFrame()
        if not upcoming.empty:
            st.markdown("### ⏰ Upcoming Appointment Reminders")
            dcol="ParsedDate" if "ParsedDate" in upcoming.columns else "Parsed Date"
            for _,r in upcoming.iterrows():
                if pd.notnull(r[dcol]):
                    st.info(f"🗓 {r[dcol].strftime('%d %b %Y')} - {r.get('Patient Name','(Name)')} with Dr. {r.get('Doctor','')} (Serial #{r.get('Serial','-')})")

        st.download_button("⬇️ Export All Appointments CSV", view.to_csv(index=False).encode('utf-8'),
                           "appointments_admin.csv", mime="text/csv")

    st.markdown("### 🧰 Bed Inventory Tools")
    hdf=load_hospitals(); h_opts=hdf["Hospital"].tolist() if not hdf.empty else []
    sel_h = st.selectbox("Hospital", options=h_opts) if h_opts else st.text_input("Hospital (type)")
    sel_t = st.selectbox("Tier", options=["General Bed","General Cabin","VIP Cabin"])
    sel_d = st.date_input("Date to reset", value=date.today())
    if st.button("♻️ Reset availability for that date"):
        if sel_h: reset_inventory_for_date(sel_h, sel_t, sel_d); st.success("Availability reset to ALL AVAILABLE for that date/tier.")
        else: st.warning("Please select or type a hospital.")

    st.markdown("### 🧾 Waitlist")
    def ensure_waitlist():
        if not os.path.exists(WAITLIST_PATH) or os.stat(WAITLIST_PATH).st_size==0:
            with open(WAITLIST_PATH,"w",newline="",encoding="utf-8") as f:
                csv.writer(f).writerow(["Timestamp","Hospital","Tier","Date","Status","Patient","Phone","AssignedUnit"])
    def load_waitlist():
        ensure_waitlist()
        try: return pd.read_csv(WAITLIST_PATH)
        except Exception: return pd.DataFrame(columns=["Timestamp","Hospital","Tier","Date","Status","Patient","Phone","AssignedUnit"])
    def save_waitlist(df): df.to_csv(WAITLIST_PATH, index=False)
    def auto_match_waitlist():
        df=load_waitlist(); assigned=0; changes=[]
        if df.empty: return assigned, changes
        for idx,row in df[df["Status"]=="waiting"].sort_values("Timestamp").iterrows():
            hosp=str(row["Hospital"]); tier=str(row["Tier"])
            try: day=datetime.strptime(str(row["Date"]),"%Y-%m-%d").date()
            except Exception: continue
            inv=get_inventory(hosp,tier,day); avail=inv[inv["Status"]!="booked"]["UnitID"].astype(str).tolist()
            if not avail: continue
            unit_id=sorted(avail, key=lambda x:(len(x),x))[0]
            mark_booked(hosp,tier,unit_id,day)
            df.at[idx,"Status"]="assigned"; df.at[idx,"AssignedUnit"]=unit_id; assigned+=1; changes.append((idx,unit_id))
        if assigned>0: save_waitlist(df)
        return assigned, changes

    wl_df = load_waitlist()
    if wl_df.empty:
        st.info("No waitlist entries yet.")
    else:
        st.dataframe(wl_df, use_container_width=True)
        if st.button("🔧 Run Auto-Match Now"):
            n,chg=auto_match_waitlist()
            st.success(f"Assigned {n} waitlist request(s): "+", ".join([f"#{i}→{u}" for i,u in chg])) if n>0 else st.info("No matches found right now.")
        wl_df=load_waitlist()
        st.download_button("⬇️ Export Waitlist CSV", wl_df.to_csv(index=False).encode("utf-8"), "waitlist.csv", mime="text/csv")
else:
    st.warning("🔐 Admin access required to view dashboard.")

st.markdown("---")
st.caption("Built with ❤️ by Doctigo AI Booking System")


