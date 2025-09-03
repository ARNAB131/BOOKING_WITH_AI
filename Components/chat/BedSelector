import React from 'react';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { motion } from 'framer-motion';
import { Bed, UserCheck, Tv, Wind, Check } from 'lucide-react';

const bedOptions = [
  {
    type: "General Bed",
    price: 100,
    features: ["1 bed", "1 chair", "bed table"],
    icon: <Bed className="w-8 h-8 text-blue-500" />,
  },
  {
    type: "General Cabin",
    price: 1000,
    features: ["2 beds", "attached washroom", "bed table", "chair", "food x3 times"],
    icon: <UserCheck className="w-8 h-8 text-green-500" />,
  },
  {
    type: "VIP Cabin",
    price: 4000,
    features: ["premium bed x2", "sofa", "Air Conditioning", "attached washroom", "TV", "fridge", "bed table x2", "coffee table", "2 chairs"],
    icon: <div className="flex gap-1"><Tv className="w-8 h-8 text-purple-500" /><Wind className="w-8 h-8 text-purple-500" /></div>,
  }
];

export default function BedSelector({ onSelect }) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      className="bg-white rounded-lg p-4 space-y-4"
    >
      <div className="grid md:grid-cols-3 gap-4">
        {bedOptions.map((option, index) => (
          <Card key={index} className="flex flex-col">
            <CardHeader>
              <div className="flex justify-between items-center">
                <CardTitle className="text-lg">{option.type}</CardTitle>
                <div className="text-blue-600">{option.icon}</div>
              </div>
              <p className="text-xl font-bold text-gray-800">₹{option.price} <span className="text-sm font-normal text-gray-500">/ night</span></p>
            </CardHeader>
            <CardContent className="flex-grow">
              <ul className="space-y-2 text-sm text-gray-600">
                {option.features.map((feature, i) => (
                  <li key={i} className="flex items-start gap-2">
                    <Check className="w-4 h-4 text-green-500 mt-0.5 flex-shrink-0" />
                    <span>{feature}</span>
                  </li>
                ))}
              </ul>
            </CardContent>
            <div className="p-4 pt-0">
              <Button onClick={() => onSelect(option)} className="w-full">Select</Button>
            </div>
          </Card>
        ))}
      </div>
      <Button onClick={() => onSelect(null)} variant="outline" className="w-full mt-4">
        No, thanks
      </Button>
    </motion.div>
  );
}
