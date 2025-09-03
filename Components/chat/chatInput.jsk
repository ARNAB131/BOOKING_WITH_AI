import React, { useState } from 'react';
import { Send, Mic } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';

export default function ChatInput({ onSend, placeholder = "Type your message...", disabled = false }) {
  const [message, setMessage] = useState('');

  const handleSubmit = (e) => {
    e.preventDefault();
    if (message.trim() && !disabled) {
      onSend(message.trim());
      setMessage('');
    }
  };

  return (
    <form onSubmit={handleSubmit} className="flex gap-2 p-4 bg-white border-t border-gray-200">
      <div className="flex-1 relative">
        <Input
          value={message}
          onChange={(e) => setMessage(e.target.value)}
          placeholder={placeholder}
          disabled={disabled}
          className="pr-12 bg-gray-50 border-gray-200 focus:border-blue-500 focus:ring-blue-500"
        />
        <Button
          type="button"
          size="icon"
          variant="ghost"
          className="absolute right-1 top-1/2 -translate-y-1/2 h-8 w-8 text-gray-400 hover:text-gray-600"
          onClick={() => {
            // Voice input could be implemented here
            alert('Voice input feature coming soon!');
          }}
        >
          <Mic className="w-4 h-4" />
        </Button>
      </div>
      <Button 
        type="submit" 
        disabled={!message.trim() || disabled}
        className="bg-blue-500 hover:bg-blue-600 text-white px-6"
      >
        <Send className="w-4 h-4" />
      </Button>
    </form>
  );
}
