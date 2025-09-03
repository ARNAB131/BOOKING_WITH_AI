import React from 'react';
import { motion } from 'framer-motion';
import { Bot, User, Clock } from 'lucide-react';

export default function ChatMessage({ message, isBot, timestamp, isTyping = false }) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.3 }}
      className={`flex gap-3 mb-4 ${isBot ? 'justify-start' : 'justify-end'}`}
    >
      {isBot && (
        <div className="flex-shrink-0 w-10 h-10 bg-gradient-to-r from-blue-500 to-blue-600 rounded-full flex items-center justify-center">
          <Bot className="w-5 h-5 text-white" />
        </div>
      )}
      
      <div className={`max-w-xs md:max-w-md ${isBot ? 'order-2' : 'order-1'}`}>
        <div className={`rounded-2xl px-4 py-3 shadow-sm ${
          isBot 
            ? 'bg-white border border-gray-200 text-gray-800' 
            : 'bg-gradient-to-r from-blue-500 to-blue-600 text-white'
        }`}>
          {isTyping ? (
            <div className="flex items-center gap-1">
              <div className="flex gap-1">
                <div className="w-2 h-2 bg-blue-400 rounded-full animate-bounce"></div>
                <div className="w-2 h-2 bg-blue-400 rounded-full animate-bounce" style={{animationDelay: '0.1s'}}></div>
                <div className="w-2 h-2 bg-blue-400 rounded-full animate-bounce" style={{animationDelay: '0.2s'}}></div>
              </div>
              <span className="text-sm text-gray-500 ml-2">Doc is typing...</span>
            </div>
          ) : (
            <p className="text-sm leading-relaxed">{message}</p>
          )}
        </div>
        {timestamp && (
          <div className={`flex items-center gap-1 mt-1 px-2 ${isBot ? 'justify-start' : 'justify-end'}`}>
            <Clock className="w-3 h-3 text-gray-400" />
            <span className="text-xs text-gray-400">{timestamp}</span>
          </div>
        )}
      </div>

      {!isBot && (
        <div className="flex-shrink-0 w-10 h-10 bg-gradient-to-r from-green-500 to-green-600 rounded-full flex items-center justify-center order-3">
          <User className="w-5 h-5 text-white" />
        </div>
      )}
    </motion.div>
  );
}
