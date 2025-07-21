import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { motion } from 'framer-motion';
import { 
  ArrowLeft, 
  User, 
  Bell, 
  Shield, 
  Database, 
  Save,
  Key,
  Mail,
  Smartphone
} from 'lucide-react';
import { Switch } from '@headlessui/react';
import toast from 'react-hot-toast';

export const Settings: React.FC = () => {
  const navigate = useNavigate();
  const [isLoading, setIsLoading] = useState(false);
  
  const [profile, setProfile] = useState({
    name: 'Demo User',
    email: 'demo@algohive.com',
    phone: '+91 9876543210',
  });

  const [notifications, setNotifications] = useState({
    emailAlerts: true,
    smsAlerts: false,
    tradeExecutions: true,
    dailyReports: true,
    riskAlerts: true,
    systemAlerts: true,
  });

  const [apiKeys, setApiKeys] = useState({
    zerodhaApiKey: '********',
    zerodhaApiSecret: '********',
    openaiApiKey: '********',
    anthropicApiKey: '********',
  });

  const [dataRetention, setDataRetention] = useState({
    tradeHistory: 90,
    performanceData: 180,
    logRetention: 30,
  });

  const handleSaveProfile = async () => {
    setIsLoading(true);
    try {
      // Simulate API call
      await new Promise(resolve => setTimeout(resolve, 1000));
      toast.success('Profile updated successfully');
    } catch (error) {
      toast.error('Failed to update profile');
    } finally {
      setIsLoading(false);
    }
  };

  const handleSaveNotifications = async () => {
    setIsLoading(true);
    try {
      await new Promise(resolve => setTimeout(resolve, 1000));
      toast.success('Notification preferences saved');
    } catch (error) {
      toast.error('Failed to save preferences');
    } finally {
      setIsLoading(false);
    }
  };

  const handleSaveApiKeys = async () => {
    setIsLoading(true);
    try {
      await new Promise(resolve => setTimeout(resolve, 1000));
      toast.success('API keys updated successfully');
    } catch (error) {
      toast.error('Failed to update API keys');
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-gray-50">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <div className="mb-8">
          <button
            onClick={() => navigate('/')}
            className="flex items-center text-gray-600 hover:text-gray-900 mb-4"
          >
            <ArrowLeft className="h-5 w-5 mr-2" />
            Back to Dashboard
          </button>
          <h1 className="text-3xl font-bold text-gray-900">Settings</h1>
          <p className="mt-2 text-gray-600">Manage your account and system preferences</p>
        </div>

        <div className="space-y-6">
          {/* Profile Settings */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className="bg-white rounded-lg shadow p-6"
          >
            <h2 className="text-lg font-semibold mb-4 flex items-center">
              <User className="h-5 w-5 mr-2" />
              Profile Settings
            </h2>
            
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  Full Name
                </label>
                <input
                  type="text"
                  value={profile.name}
                  onChange={(e) => setProfile({ ...profile, name: e.target.value })}
                  className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-primary-500"
                />
              </div>
              
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  Email Address
                </label>
                <div className="relative">
                  <Mail className="absolute left-3 top-2.5 h-5 w-5 text-gray-400" />
                  <input
                    type="email"
                    value={profile.email}
                    onChange={(e) => setProfile({ ...profile, email: e.target.value })}
                    className="w-full pl-10 pr-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-primary-500"
                  />
                </div>
              </div>
              
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  Phone Number
                </label>
                <div className="relative">
                  <Smartphone className="absolute left-3 top-2.5 h-5 w-5 text-gray-400" />
                  <input
                    type="tel"
                    value={profile.phone}
                    onChange={(e) => setProfile({ ...profile, phone: e.target.value })}
                    className="w-full pl-10 pr-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-primary-500"
                  />
                </div>
              </div>
            </div>
            
            <button
              onClick={handleSaveProfile}
              disabled={isLoading}
              className="mt-4 bg-primary-600 text-white px-4 py-2 rounded-md hover:bg-primary-700 disabled:opacity-50 flex items-center"
            >
              <Save className="h-4 w-4 mr-2" />
              Save Profile
            </button>
          </motion.div>

          {/* Notification Preferences */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.1 }}
            className="bg-white rounded-lg shadow p-6"
          >
            <h2 className="text-lg font-semibold mb-4 flex items-center">
              <Bell className="h-5 w-5 mr-2" />
              Notification Preferences
            </h2>
            
            <div className="space-y-4">
              <div className="flex items-center justify-between">
                <div>
                  <p className="font-medium">Email Alerts</p>
                  <p className="text-sm text-gray-600">Receive alerts via email</p>
                </div>
                <Switch
                  checked={notifications.emailAlerts}
                  onChange={(checked) => setNotifications({ ...notifications, emailAlerts: checked })}
                  className={`${
                    notifications.emailAlerts ? 'bg-primary-600' : 'bg-gray-300'
                  } relative inline-flex h-6 w-11 items-center rounded-full transition-colors`}
                >
                  <span
                    className={`${
                      notifications.emailAlerts ? 'translate-x-6' : 'translate-x-1'
                    } inline-block h-4 w-4 transform rounded-full bg-white transition-transform`}
                  />
                </Switch>
              </div>
              
              <div className="flex items-center justify-between">
                <div>
                  <p className="font-medium">SMS Alerts</p>
                  <p className="text-sm text-gray-600">Receive alerts via SMS</p>
                </div>
                <Switch
                  checked={notifications.smsAlerts}
                  onChange={(checked) => setNotifications({ ...notifications, smsAlerts: checked })}
                  className={`${
                    notifications.smsAlerts ? 'bg-primary-600' : 'bg-gray-300'
                  } relative inline-flex h-6 w-11 items-center rounded-full transition-colors`}
                >
                  <span
                    className={`${
                      notifications.smsAlerts ? 'translate-x-6' : 'translate-x-1'
                    } inline-block h-4 w-4 transform rounded-full bg-white transition-transform`}
                  />
                </Switch>
              </div>
              
              <div className="flex items-center justify-between">
                <div>
                  <p className="font-medium">Trade Executions</p>
                  <p className="text-sm text-gray-600">Notify when trades are executed</p>
                </div>
                <Switch
                  checked={notifications.tradeExecutions}
                  onChange={(checked) => setNotifications({ ...notifications, tradeExecutions: checked })}
                  className={`${
                    notifications.tradeExecutions ? 'bg-primary-600' : 'bg-gray-300'
                  } relative inline-flex h-6 w-11 items-center rounded-full transition-colors`}
                >
                  <span
                    className={`${
                      notifications.tradeExecutions ? 'translate-x-6' : 'translate-x-1'
                    } inline-block h-4 w-4 transform rounded-full bg-white transition-transform`}
                  />
                </Switch>
              </div>
              
              <div className="flex items-center justify-between">
                <div>
                  <p className="font-medium">Daily Reports</p>
                  <p className="text-sm text-gray-600">Receive daily performance reports</p>
                </div>
                <Switch
                  checked={notifications.dailyReports}
                  onChange={(checked) => setNotifications({ ...notifications, dailyReports: checked })}
                  className={`${
                    notifications.dailyReports ? 'bg-primary-600' : 'bg-gray-300'
                  } relative inline-flex h-6 w-11 items-center rounded-full transition-colors`}
                >
                  <span
                    className={`${
                      notifications.dailyReports ? 'translate-x-6' : 'translate-x-1'
                    } inline-block h-4 w-4 transform rounded-full bg-white transition-transform`}
                  />
                </Switch>
              </div>
            </div>
            
            <button
              onClick={handleSaveNotifications}
              disabled={isLoading}
              className="mt-6 bg-primary-600 text-white px-4 py-2 rounded-md hover:bg-primary-700 disabled:opacity-50 flex items-center"
            >
              <Save className="h-4 w-4 mr-2" />
              Save Preferences
            </button>
          </motion.div>

          {/* API Keys */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.2 }}
            className="bg-white rounded-lg shadow p-6"
          >
            <h2 className="text-lg font-semibold mb-4 flex items-center">
              <Key className="h-5 w-5 mr-2" />
              API Keys
            </h2>
            
            <div className="space-y-4">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  Zerodha API Key
                </label>
                <input
                  type="password"
                  value={apiKeys.zerodhaApiKey}
                  onChange={(e) => setApiKeys({ ...apiKeys, zerodhaApiKey: e.target.value })}
                  className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-primary-500"
                />
              </div>
              
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  Zerodha API Secret
                </label>
                <input
                  type="password"
                  value={apiKeys.zerodhaApiSecret}
                  onChange={(e) => setApiKeys({ ...apiKeys, zerodhaApiSecret: e.target.value })}
                  className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-primary-500"
                />
              </div>
              
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  OpenAI API Key
                </label>
                <input
                  type="password"
                  value={apiKeys.openaiApiKey}
                  onChange={(e) => setApiKeys({ ...apiKeys, openaiApiKey: e.target.value })}
                  className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-primary-500"
                />
              </div>
              
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  Anthropic API Key
                </label>
                <input
                  type="password"
                  value={apiKeys.anthropicApiKey}
                  onChange={(e) => setApiKeys({ ...apiKeys, anthropicApiKey: e.target.value })}
                  className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-primary-500"
                />
              </div>
            </div>
            
            <button
              onClick={handleSaveApiKeys}
              disabled={isLoading}
              className="mt-4 bg-primary-600 text-white px-4 py-2 rounded-md hover:bg-primary-700 disabled:opacity-50 flex items-center"
            >
              <Save className="h-4 w-4 mr-2" />
              Update API Keys
            </button>
          </motion.div>

          {/* Data Retention */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.3 }}
            className="bg-white rounded-lg shadow p-6"
          >
            <h2 className="text-lg font-semibold mb-4 flex items-center">
              <Database className="h-5 w-5 mr-2" />
              Data Retention
            </h2>
            
            <div className="space-y-4">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  Trade History (days)
                </label>
                <input
                  type="number"
                  value={dataRetention.tradeHistory}
                  onChange={(e) => setDataRetention({ 
                    ...dataRetention, 
                    tradeHistory: parseInt(e.target.value) 
                  })}
                  className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-primary-500"
                />
              </div>
              
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  Performance Data (days)
                </label>
                <input
                  type="number"
                  value={dataRetention.performanceData}
                  onChange={(e) => setDataRetention({ 
                    ...dataRetention, 
                    performanceData: parseInt(e.target.value) 
                  })}
                  className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-primary-500"
                />
              </div>
              
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  Log Retention (days)
                </label>
                <input
                  type="number"
                  value={dataRetention.logRetention}
                  onChange={(e) => setDataRetention({ 
                    ...dataRetention, 
                    logRetention: parseInt(e.target.value) 
                  })}
                  className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-primary-500"
                />
              </div>
            </div>
          </motion.div>
        </div>
      </div>
    </div>
  );
};