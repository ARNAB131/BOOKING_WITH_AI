import pandas as pd
from datetime import datetime, timedelta

# Load Doctor Database
doctor_df = pd.read_csv("doctors.csv")

# Expanded Symptom-to-Specialization Map
symptom_specialization_map = {
    "fever": "General Medicine",
    "cold": "General Medicine",
    "cough": "General Medicine",
    "headache": "General Medicine",
    "body ache": "General Medicine",
    "heart pain": "Cardiologist",
    "chest pain": "Cardiologist",
    "breathlessness": "Cardiologist",
    "ear pain": "ENT Surgeon",
    "throat pain": "ENT Surgeon",
    "hearing loss": "ENT Surgeon",
    "skin rash": "Dermatology",
    "pimple": "Dermatology",
    "acne": "Dermatology",
    "eczema": "Dermatology",
    "tooth pain": "Dentist",
    "cavity": "Dentist",
    "gum bleeding": "Dentist",
    "urine issue": "Urologist",
    "kidney pain": "Urologist",
    "pregnancy": "Gynecologist",
    "irregular periods": "Gynecologist",
    "child fever": "Pediatrician",
    "infant cough": "Pediatrician",
    "muscle pain": "Orthopedic",
    "joint pain": "Orthopedic",
    "back pain": "Orthopedic",
    "mental stress": "Neuropsychiatrist",
    "anxiety": "Neuropsychiatrist",
    "depression": "Neuropsychiatrist",
    "vision problem": "Ophthalmologist",
    "eye redness": "Ophthalmologist",
    "diabetes": "Endocrinologist",
    "thyroid issue": "Endocrinologist",
    "gastric issue": "Gastroenterologist",
    "stomach pain": "Gastroenterologist",
    "liver issue": "Gastroenterologist",
    "allergy": "Allergist",
    "asthma": "Pulmonologist",
    "breathing problem": "Pulmonologist"
}

def predict_specialization(symptoms):
    matched_specializations = []
    for symptom in symptoms:
        for keyword, specialization in symptom_specialization_map.items():
            if keyword in symptom.lower():
                matched_specializations.append(specialization)
    return matched_specializations if matched_specializations else ["General Medicine"]

def generate_slots(visiting_time):
    try:
        if visiting_time.lower() == "not specified":
            return ["Visiting Time Not Specified"]

        start, end = visiting_time.replace('pm', 'PM').replace('am', 'AM').split('-')
        start_time = datetime.strptime(start.strip(), "%I.%M%p")
        end_time = datetime.strptime(end.strip(), "%I.%M%p")

        slots = []
        while start_time + timedelta(minutes=30) <= end_time:
            slot = f"{start_time.strftime('%I:%M %p')} - {(start_time + timedelta(minutes=30)).strftime('%I:%M %p')}"
            slots.append(slot)
            start_time += timedelta(minutes=30)
        return slots
    except Exception as e:
        return ["Slot Info Unavailable"]

def recommend_doctors(symptoms, preferred_time=None):
    specializations = predict_specialization(symptoms)
    filtered = doctor_df[doctor_df['Specialization'].str.contains('|'.join(specializations), case=False, na=False)]

    general_doctors = doctor_df[doctor_df['Specialization'].str.contains("General Medicine", case=False, na=False)]

    if filtered.empty:
        print(f"❗ No exact match found for {specializations}, showing General Medicine doctors.")
        filtered = general_doctors
    elif len(filtered) < 3:
        filtered = pd.concat([filtered, general_doctors]).drop_duplicates()

    if filtered.empty:
        return f"\n❌ No doctors available.", []

    recommendations = []
    for _, row in filtered.iterrows():
        slots = generate_slots(row['Visiting Time'])
        recommendations.append({
            "Doctor": row['Doctor Name'],
            "Specialization": row['Specialization'],
            "Chamber": row['Chamber'],
            "Slots": slots
        })

    return f"\n✅ Recommended Doctors for symptoms '{', '.join(symptoms)}':", recommendations

# Example Testing Code
if __name__ == "__main__":
    symptoms_input = input("Enter your symptoms (comma separated): ")
    symptoms = [s.strip() for s in symptoms_input.split(",")]
    message, doctor_recommendations = recommend_doctors(symptoms)
    print(message)
    for doc in doctor_recommendations:
        print(f"\nDoctor: {doc['Doctor']}\nSpecialization: {doc['Specialization']}\nChamber: {doc['Chamber']}\nSlots: {doc['Slots']}")
