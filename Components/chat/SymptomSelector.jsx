import React, { useState } from 'react';
import { Check, X } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { motion } from 'framer-motion';

const commonSymptoms = [
  'Fever', 'Headache', 'Cough', 'Sore throat', 'Body ache', 
  'Nausea', 'Vomiting', 'Diarrhea', 'Chest pain', 'Shortness of breath',
  'Dizziness', 'Fatigue', 'Loss of appetite', 'Stomach pain', 'Joint pain'
];

export default function SymptomSelector({ onSubmit, onSkip }) {
  const [selectedSymptoms, setSelectedSymptoms] = useState([]);
  const [customSymptom, setCustomSymptom] = useState('');

  const toggleSymptom = (symptom) => {
    setSelectedSymptoms(prev => 
      prev.includes(symptom) 
        ? prev.filter(s => s !== symptom)
        : [...prev, symptom]
    );
  };

  const addCustomSymptom = () => {
    if (customSymptom.trim() && !selectedSymptoms.includes(customSymptom.trim())) {
      setSelectedSymptoms(prev => [...prev, customSymptom.trim()]);
      setCustomSymptom('');
    }
  };

  return (
    <motion.div 
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      className="bg-white rounded-lg border border-gray-200 p-4 space-y-4"
    >
      <div>
        <h3 className="font-medium text-gray-900 mb-3">Select your symptoms:</h3>
        <div className="flex flex-wrap gap-2">
          {commonSymptoms.map((symptom) => (
            <Badge
              key={symptom}
              variant={selectedSymptoms.includes(symptom) ? "default" : "outline"}
              className={`cursor-pointer transition-colors ${
                selectedSymptoms.includes(symptom) 
                  ? 'bg-blue-500 text-white hover:bg-blue-600' 
                  : 'hover:bg-blue-50 hover:text-blue-600'
              }`}
              onClick={() => toggleSymptom(symptom)}
            >
              {symptom}
              {selectedSymptoms.includes(symptom) && <Check className="w-3 h-3 ml-1" />}
            </Badge>
          ))}
        </div>
      </div>

      <div className="flex gap-2">
        <input
          type="text"
          value={customSymptom}
          onChange={(e) => setCustomSymptom(e.target.value)}
          placeholder="Add custom symptom..."
          className="flex-1 px-3 py-2 border border-gray-200 rounded-md text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
          onKeyPress={(e) => e.key === 'Enter' && addCustomSymptom()}
        />
        <Button onClick={addCustomSymptom} size="sm">Add</Button>
      </div>

      {selectedSymptoms.length > 0 && (
        <div className="space-y-2">
          <p className="text-sm text-gray-600">Selected symptoms:</p>
          <div className="flex flex-wrap gap-2">
            {selectedSymptoms.map((symptom) => (
              <Badge key={symptom} variant="secondary" className="flex items-center gap-1">
                {symptom}
                <X 
                  className="w-3 h-3 cursor-pointer hover:text-red-500" 
                  onClick={() => toggleSymptom(symptom)}
                />
              </Badge>
            ))}
          </div>
        </div>
      )}

      <div className="flex gap-2 pt-2">
        <Button 
          onClick={() => onSubmit(selectedSymptoms)} 
          disabled={selectedSymptoms.length === 0}
          className="flex-1 bg-blue-500 hover:bg-blue-600"
        >
          Continue with Symptoms ({selectedSymptoms.length})
        </Button>
        <Button 
          onClick={onSkip} 
          variant="outline"
          className="flex-1"
        >
          Skip / No Symptoms
        </Button>
      </div>
    </motion.div>
  );
}
