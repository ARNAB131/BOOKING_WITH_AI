import React, { useState, useEffect, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Heart, AlertTriangle, Phone } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Card, CardContent } from '@/components/ui/card';
import { Appointment } from '@/entities/Appointment';
import { Doctor } from '@/entities/Doctor';
import { Hospital } from '@/entities/Hospital';

import ChatMessage from '../components/chat/ChatMessage';
import ChatInput from '../components/chat/ChatInput';
import SymptomSelector from '../components/chat/SymptomSelector';
import DoctorCard from '../components/chat/DoctorCard';
import AppointmentCard from '../components/chat/AppointmentCard';
import BedSelector from '../components/chat/BedSelector';

const conversationSteps = {
  INITIAL: 'initial',
  ASK_NAME: 'ask_name',
  CONFIRM_BOOKING_TYPE: 'confirm_booking_type',
  ASK_SYMPTOMS: 'ask_symptoms',
  SHOW_DOCTORS: 'show_doctors',
  ASK_BED: 'ask_bed',
  COLLECT_DETAILS: 'collect_details',
  FINAL_CARD: 'final_card'
};

const patientDetailSteps = [
  { key: 'patient_phone', label: 'phone number', type: 'tel' },
  { key: 'patient_gender', label: 'gender (male/female/other)', type: 'text' },
  { key: 'patient_age', label: 'age', type: 'number' },
  { key: 'patient_email', label: 'email address', type: 'email' },
  { key: 'patient_address', label: 'address', type: 'text' }
];

export default function BookingPage() {
  const [currentStep, setCurrentStep] = useState(conversationSteps.INITIAL);
  const [messages, setMessages] = useState([]);
  const [bookingType, setBookingType] = useState('');
  const [patientName, setPatientName] = useState('');
  const [symptoms, setSymptoms] = useState([]);
  const [doctors, setDoctors] = useState([]);
  const [hospitals, setHospitals] = useState([]);
  const [selectedDoctor, setSelectedDoctor] = useState(null);
  const [bedSelection, setBedSelection] = useState(null);
  const [patientDetails, setPatientDetails] = useState({});
  const [currentDetailStep, setCurrentDetailStep] = useState(0);
  const [isTyping, setIsTyping] = useState(false);
  const [finalAppointment, setFinalAppointment] = useState(null);
  const [userLocation, setUserLocation] = useState(null);
  
  const messagesEndRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages, isTyping]);

  useEffect(() => {
    const loadData = async () => {
      const [fetchedDoctors, fetchedHospitals] = await Promise.all([
        Doctor.list(),
        Hospital.list()
      ]);
      setDoctors(fetchedDoctors);
      setHospitals(fetchedHospitals);
    };

    loadData();
    
    if (navigator.geolocation) {
      navigator.geolocation.getCurrentPosition((position) => {
        setUserLocation({
          lat: position.coords.latitude,
          lng: position.coords.longitude
        });
      });
    }
  }, []);

  const addMessage = (text, isBot = true) => {
    const newMessage = {
      id: Date.now(),
      text,
      isBot,
      timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
    };
    setMessages(prev => [...prev, newMessage]);
  };

  const addBotMessage = async (text, delay = 1000) => {
    setIsTyping(true);
    await new Promise(resolve => setTimeout(resolve, delay));
    setIsTyping(false);
    addMessage(text, true);
  };

  const calculateDistance = (lat1, lng1, lat2, lng2) => {
    if (lat1 === null || lng1 === null || lat2 === null || lng2 === null || !lat2 || !lng1) return null;
    const R = 6371;
    const dLat = (lat2 - lat1) * Math.PI / 180;
    const dLng = (lng2 - lng1) * Math.PI / 180;
    const a = Math.sin(dLat/2) * Math.sin(dLat/2) + Math.cos(lat1 * Math.PI / 180) * Math.cos(lat2 * Math.PI / 180) * Math.sin(dLng/2) * Math.sin(dLng/2);
    const c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1-a));
    return R * c;
  };

  const calculateETA = (distance) => {
    if (distance === null) return null;
    return Math.round(distance * 2); // Simple ETA: 30 km/h average
  };

  const handleBookingTypeSelect = async (type) => {
    setBookingType(type);
    addMessage(`I want ${type} booking`, false);
    setCurrentStep(conversationSteps.ASK_NAME);
    await addBotMessage("Hello! I am Doc, Your friendly neighborhood Spider Doc 🕷️🩺. What's your name?");
  };

  const handleNameSubmit = async (name) => {
    setPatientName(name);
    addMessage(name, false);
    setCurrentStep(conversationSteps.CONFIRM_BOOKING_TYPE);
    await addBotMessage(`Hello ${name}! So you opted for ${bookingType} booking.`);
    setCurrentStep(conversationSteps.ASK_SYMPTOMS);
    
    if (bookingType === 'emergency') {
      await addBotMessage("Woooo it's an EMERGENCY I see! Don't worry, everything will be fine. Just enter the patient's symptoms or if you just want to book a doctor of your preference then type 'next'.");
    } else {
      await addBotMessage("Enter your symptoms or if no symptoms then type 'next'.");
    }
  };

  const handleSymptomsSubmit = async (selectedSymptoms) => {
    setSymptoms(selectedSymptoms);
    if (selectedSymptoms.length > 0) {
      addMessage(`Selected symptoms: ${selectedSymptoms.join(', ')}`, false);
    } else {
      addMessage('No symptoms / Skip', false);
    }
    
    setCurrentStep(conversationSteps.SHOW_DOCTORS);
    await addBotMessage("Based on your information, here are the available doctors for appointment:");
  };

  const handleDoctorSelect = async (doctor) => {
    setSelectedDoctor(doctor);
    addMessage(`Selected Dr. ${doctor.name}`, false);
    
    let distanceInfo = '';
    const hospital = hospitals.find(h => h.name === doctor.hospital);
    
    if (userLocation && hospital) {
      const distance = calculateDistance(userLocation.lat, userLocation.lng, hospital.latitude, hospital.longitude);
      const eta = calculateETA(distance);
      
      const now = new Date();
      const departureTime = eta ? new Date(now.getTime() + (eta * 60000)) : null;
      
      if (eta && departureTime) {
        distanceInfo = ` You will need approximately ${eta} minutes to reach the chamber. Hey, you should set off at ${departureTime.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' })} for the chamber.`;
      }
    }

    await addBotMessage(`Great choice! Dr. ${doctor.name} is available.${distanceInfo}`);
    setCurrentStep(conversationSteps.ASK_BED);
    await addBotMessage("Do you need to book a Bed or Cabin? Please choose from the options below.");
  };

  const handleBedSelect = async (selection) => {
    setBedSelection(selection);
    
    if (selection) {
      addMessage(`Selected ${selection.type}`, false);
      await addBotMessage(`Great! I'll include the ${selection.type} booking. Now let me collect some patient details.`);
    } else {
      addMessage("No bed needed.", false);
      await addBotMessage("No problem! Now let me collect some patient details.");
    }
    
    setCurrentStep(conversationSteps.COLLECT_DETAILS);
    const firstDetail = patientDetailSteps[0];
    await addBotMessage(`Please enter the patient's ${firstDetail.label}:`);
  };

  const handleDetailSubmit = async (detail) => {
    const currentDetail = patientDetailSteps[currentDetailStep];
    setPatientDetails(prev => ({ ...prev, [currentDetail.key]: detail }));
    addMessage(detail, false);
    
    if (currentDetailStep < patientDetailSteps.length - 1) {
      setCurrentDetailStep(prev => prev + 1);
      const nextDetail = patientDetailSteps[currentDetailStep + 1];
      await addBotMessage(`Thank you! Now please enter the patient's ${nextDetail.label}:`);
    } else {
      setCurrentStep(conversationSteps.FINAL_CARD);
      await addBotMessage("Perfect! I have all the information needed. Preparing your appointment card...");
      
      const hospital = hospitals.find(h => h.name === selectedDoctor.hospital);
      const distance = userLocation && hospital ? calculateDistance(userLocation.lat, userLocation.lng, hospital.latitude, hospital.longitude) : null;
      const eta = calculateETA(distance);

      const appointmentData = {
        patient_name: patientName,
        booking_type: bookingType,
        symptoms: symptoms,
        doctor_name: selectedDoctor.name,
        hospital_name: selectedDoctor.hospital,
        appointment_date: new Date().toISOString(),
        appointment_time: selectedDoctor.available_slots[0],
        needs_bed: !!bedSelection,
        bed_type: bedSelection ? bedSelection.type : null,
        bed_details: bedSelection ? JSON.stringify(bedSelection) : null,
        distance_km: distance,
        estimated_travel_time: eta,
        ...patientDetails,
        ...patientDetailSteps.reduce((acc, step) => {
          if (step.key === currentDetail.key) acc[step.key] = detail;
          else acc[step.key] = patientDetails[step.key];
          return acc;
        }, {})
      };
      
      try {
        const savedAppointment = await Appointment.create(appointmentData);
        setFinalAppointment(savedAppointment);
      } catch (error) {
        console.error('Error saving appointment:', error);
        setFinalAppointment(appointmentData); // Fallback to local data
      }
      await addBotMessage("🎉 Appointment confirmed! Here's your appointment card:");
    }
  };

  const handleChatInput = (message) => {
    const lowerMsg = message.toLowerCase();
    
    if (currentStep === conversationSteps.ASK_NAME) handleNameSubmit(message);
    else if (currentStep === conversationSteps.COLLECT_DETAILS) handleDetailSubmit(message);
    else if (currentStep === conversationSteps.ASK_SYMPTOMS && lowerMsg === 'next') handleSymptomsSubmit([]);
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 via-white to-green-50">
      <div className="max-w-4xl mx-auto px-4 py-8">
        <div className="text-center mb-8">
          <motion.div initial={{ opacity: 0, y: -20 }} animate={{ opacity: 1, y: 0 }} className="flex items-center justify-center gap-2 mb-4">
            <div className="w-12 h-12 bg-gradient-to-r from-blue-500 to-green-500 rounded-full flex items-center justify-center">
              <Heart className="w-6 h-6 text-white" />
            </div>
            <h1 className="text-3xl font-bold bg-gradient-to-r from-blue-600 to-green-600 bg-clip-text text-transparent">Doctigo AI</h1>
          </motion.div>
          <p className="text-gray-600">Your AI-powered medical booking assistant</p>
        </div>

        {currentStep === conversationSteps.INITIAL && (
          <motion.div initial={{ opacity: 0, scale: 0.9 }} animate={{ opacity: 1, scale: 1 }} className="grid md:grid-cols-2 gap-6 max-w-2xl mx-auto">
            <Card className="cursor-pointer hover:shadow-lg transition-all duration-300 border-2 border-blue-200 hover:border-blue-400" onClick={() => handleBookingTypeSelect('normal')}>
              <CardContent className="p-8 text-center">
                <div className="w-16 h-16 bg-blue-100 rounded-full flex items-center justify-center mx-auto mb-4">
                  <Heart className="w-8 h-8 text-blue-600" />
                </div>
                <h3 className="text-xl font-semibold text-gray-900 mb-2">Normal Booking</h3>
                <p className="text-gray-600">Schedule a regular appointment</p>
              </CardContent>
            </Card>

            <Card className="cursor-pointer hover:shadow-lg transition-all duration-300 border-2 border-red-200 hover:border-red-400" onClick={() => handleBookingTypeSelect('emergency')}>
              <CardContent className="p-8 text-center">
                <div className="w-16 h-16 bg-red-100 rounded-full flex items-center justify-center mx-auto mb-4">
                  <AlertTriangle className="w-8 h-8 text-red-600" />
                </div>
                <h3 className="text-xl font-semibold text-gray-900 mb-2">Emergency Booking</h3>
                <p className="text-gray-600">For immediate medical attention</p>
                <div className="flex items-center justify-center gap-2 mt-3 text-red-600">
                  <Phone className="w-4 h-4" />
                  <span className="text-sm font-medium">Call 108 for ambulance</span>
                </div>
              </CardContent>
            </Card>
          </motion.div>
        )}

        {currentStep !== conversationSteps.INITIAL && (
          <div className="bg-white rounded-2xl shadow-lg border border-gray-200 overflow-hidden max-w-3xl mx-auto">
            <div className="h-[32rem] overflow-y-auto p-4 space-y-4 bg-gray-50">
              <AnimatePresence>
                {messages.map((message) => <ChatMessage key={message.id} {...message} />)}
                {isTyping && <ChatMessage message="" isBot={true} isTyping={true} />}
              </AnimatePresence>
              <div ref={messagesEndRef} />
            </div>

            <div className="p-4 bg-white border-t border-gray-200">
              {currentStep === conversationSteps.ASK_SYMPTOMS && <SymptomSelector onSubmit={handleSymptomsSubmit} onSkip={() => handleSymptomsSubmit([])} />}
              {currentStep === conversationSteps.SHOW_DOCTORS && (
                <div className="space-y-4 max-h-96 overflow-y-auto">
                  {doctors.map((doctor, index) => {
                    const hospital = hospitals.find(h => h.name === doctor.hospital);
                    const distance = userLocation && hospital ? calculateDistance(userLocation.lat, userLocation.lng, hospital.latitude, hospital.longitude) : null;
                    const eta = calculateETA(distance);
                    return <DoctorCard key={index} doctor={doctor} availableSlots={doctor.available_slots} estimatedTime={eta} distance={distance} onBook={handleDoctorSelect} />;
                  })}
                </div>
              )}
              {currentStep === conversationSteps.ASK_BED && <BedSelector onSelect={handleBedSelect} />}
              {currentStep === conversationSteps.FINAL_CARD && finalAppointment && <AppointmentCard appointment={finalAppointment} />}
            </div>

            {![conversationSteps.ASK_SYMPTOMS, conversationSteps.SHOW_DOCTORS, conversationSteps.ASK_BED, conversationSteps.FINAL_CARD].includes(currentStep) && (
              <ChatInput onSend={handleChatInput} placeholder={currentStep === conversationSteps.COLLECT_DETAILS ? `Enter patient's ${patientDetailSteps[currentDetailStep].label}...` : "Type your message..."} disabled={isTyping} />
            )}
          </div>
        )}
      </div>
    </div>
  );
}
