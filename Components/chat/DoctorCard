import React from 'react';
import { Clock, MapPin, Star, Calendar, Award } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Card, CardContent } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { motion } from 'framer-motion';

export default function DoctorCard({ 
  doctor, 
  availableSlots, 
  estimatedTime, 
  distance, 
  onBook 
}) {
  return (
    <motion.div
      initial={{ opacity: 0, scale: 0.95 }}
      animate={{ opacity: 1, scale: 1 }}
      transition={{ duration: 0.3 }}
    >
      <Card className="hover:shadow-md transition-shadow border-gray-200">
        <CardContent className="p-4">
          <div className="flex justify-between items-start mb-3">
            <div>
              <h3 className="font-semibold text-gray-900 text-lg">Dr. {doctor.name}</h3>
              <p className="text-blue-600 text-sm">{doctor.specialization}</p>
            </div>
            <div className="flex items-center gap-1 text-sm text-gray-600">
              <Award className="w-4 h-4 text-green-600" />
              <span>{doctor.experience} Exp.</span>
            </div>
          </div>

          <div className="space-y-2 mb-4">
            <div className="flex items-center gap-2 text-sm text-gray-600">
              <MapPin className="w-4 h-4" />
              <span>{doctor.chamber}</span>
            </div>
            
            {distance && (
              <div className="flex items-center gap-2 text-sm text-gray-600">
                <Clock className="w-4 h-4" />
                <span>{distance.toFixed(1)}km away • ~{estimatedTime} min travel</span>
              </div>
            )}
          </div>

          {availableSlots && availableSlots.length > 0 && (
            <div className="mb-4">
              <p className="text-sm text-gray-700 mb-2">Available today:</p>
              <div className="flex flex-wrap gap-2">
                {availableSlots.slice(0, 3).map((slot, index) => (
                  <Badge key={index} variant="outline" className="text-xs">
                    <Calendar className="w-3 h-3 mr-1" />
                    {slot}
                  </Badge>
                ))}
                {availableSlots.length > 3 && (
                  <Badge variant="outline" className="text-xs">
                    +{availableSlots.length - 3} more
                  </Badge>
                )}
              </div>
            </div>
          )}

          <Button 
            onClick={() => onBook(doctor)} 
            className="w-full bg-green-500 hover:bg-green-600 text-white"
            disabled={!availableSlots || availableSlots.length === 0}
          >
            Book Appointment
          </Button>
        </CardContent>
      </Card>
    </motion.div>
  );
}
