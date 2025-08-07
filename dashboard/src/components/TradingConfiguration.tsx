import React, { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import {
  Settings,
  DollarSign,
  Shield,
  TrendingUp,
  AlertTriangle,
  Save,
  RefreshCw,
  Key,
  Eye,
  EyeOff
} from 'lucide-react';
import { Switch } from '@headlessui/react';
import toast from 'react-hot-toast';

interface TradingConfig {
  // Budget & Position Sizing
  totalBudget: number;
  maxPositionSize: number;
  maxPositionValue: number;
  maxConcurrentPositions: number;

  // Risk Management
  maxRiskPerTrade: number;
  maxDailyLoss: number;
  emergencyStopAmount: number;

  // Position Management
  autoStopLoss: boolean;
  autoStopLossPercent: number;
  autoTakeProfit: boolean;
  autoTakeProfitPercent: number;

  // Trading Preferences
  tradingTypes: string[];
  minStockPrice: number;
  maxStockPrice: number;
  minVolumeThreshold: number;

  // API Configuration
  kiteApiKey: string;
  kiteApiSecret: string;
  kiteAccessToken: string;
}

interface TradingConfigurationProps {
  className?: string;
}

export const TradingConfiguration: React.FC<TradingConfigurationProps> = ({ className = '' }) => {
  const [config, setConfig] = useState<TradingConfig>({
    totalBudget: 1000,
    maxPositionSize: 200,
    maxPositionValue: 300,
    maxConcurrentPositions: 3,
    maxRiskPerTrade: 5,
    maxDailyLoss: 10,
    emergencyStopAmount: 80,
    autoStopLoss: true,
    autoStopLossPercent: 5,
    autoTakeProfit: true,
    autoTakeProfitPercent: 10,
    tradingTypes: ['intraday'],
    minStockPrice: 50,
    maxStockPrice: 5000,
    minVolumeThreshold: 10000,
    kiteApiKey: '',
    kiteApiSecret: '',
    kiteAccessToken: '',
  });

  const [isLoading, setIsLoading] = useState(false);
  const [showApiKeys, setShowApiKeys] = useState(false);
  const [activeTab, setActiveTab] = useState<'budget' | 'risk' | 'trading' | 'api'>('budget');

  useEffect(() => {
    loadConfiguration();
  }, []);

  const loadConfiguration = async () => {
    try {
      // Load configuration from API
      const response = await fetch('/api/v1/system/trading-config');
      if (response.ok) {
        const data = await response.json();
        setConfig(prev => ({ ...prev, ...data }));
      }
    } catch (error) {
      console.error('Failed to load configuration:', error);
    }
  };

  const saveConfiguration = async () => {
    setIsLoading(true);
    try {
      const response = await fetch('/api/v1/system/trading-config', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(config),
      });

      if (response.ok) {
        toast.success('Configuration saved successfully');
      } else {
        throw new Error('Failed to save configuration');
      }
    } catch (error) {
      console.error('Failed to save configuration:', error);
      toast.error('Failed to save configuration');
    } finally {
      setIsLoading(false);
    }
  };

  const testApiConnection = async () => {
    setIsLoading(true);
    try {
      const response = await fetch('/api/v1/system/test-kite-connection', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          apiKey: config.kiteApiKey,
          apiSecret: config.kiteApiSecret,
          accessToken: config.kiteAccessToken,
        }),
      });

      const result = await response.json();

      if (result.success) {
        toast.success('API connection successful!');
      } else {
        toast.error(`API connection failed: ${result.message}`);
      }
    } catch (error) {
      toast.error('Failed to test API connection');
    } finally {
      setIsLoading(false);
    }
  };

  const updateConfig = (field: keyof TradingConfig, value: any) => {
    setConfig(prev => ({ ...prev, [field]: value }));
  };

  const toggleTradingType = (type: string) => {
    setConfig(prev => ({
      ...prev,
      tradingTypes: prev.tradingTypes.includes(type)
        ? prev.tradingTypes.filter(t => t !== type)
        : [...prev.tradingTypes, type]
    }));
  };

  const renderBudgetTab = () => (
    <div className="space-y-6">
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">
            Total Trading Budget (₹)
          </label>
          <input
            type="number"
            value={config.totalBudget}
            onChange={(e) => updateConfig('totalBudget', Number(e.target.value))}
            className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
          />
          <p className="text-xs text-gray-500 mt-1">Total amount available for trading</p>
        </div>

        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">
            Max Position Size (₹)
          </label>
          <input
            type="number"
            value={config.maxPositionSize}
            onChange={(e) => updateConfig('maxPositionSize', Number(e.target.value))}
            className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
          />
          <p className="text-xs text-gray-500 mt-1">Maximum amount per trade</p>
        </div>

        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">
            Max Position Value (₹)
          </label>
          <input
            type="number"
            value={config.maxPositionValue}
            onChange={(e) => updateConfig('maxPositionValue', Number(e.target.value))}
            className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
          />
          <p className="text-xs text-gray-500 mt-1">Absolute maximum per position</p>
        </div>

        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">
            Max Concurrent Positions
          </label>
          <input
            type="number"
            value={config.maxConcurrentPositions}
            onChange={(e) => updateConfig('maxConcurrentPositions', Number(e.target.value))}
            className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
          />
          <p className="text-xs text-gray-500 mt-1">Maximum number of open positions</p>
        </div>
      </div>
    </div>
  );

  const renderRiskTab = () => (
    <div className="space-y-6">
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">
            Max Risk Per Trade (%)
          </label>
          <input
            type="number"
            step="0.1"
            value={config.maxRiskPerTrade}
            onChange={(e) => updateConfig('maxRiskPerTrade', Number(e.target.value))}
            className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
          />
        </div>

        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">
            Max Daily Loss (%)
          </label>
          <input
            type="number"
            step="0.1"
            value={config.maxDailyLoss}
            onChange={(e) => updateConfig('maxDailyLoss', Number(e.target.value))}
            className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
          />
        </div>

        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">
            Emergency Stop Amount (₹)
          </label>
          <input
            type="number"
            value={config.emergencyStopAmount}
            onChange={(e) => updateConfig('emergencyStopAmount', Number(e.target.value))}
            className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
          />
        </div>
      </div>

      <div className="border-t pt-6">
        <h3 className="text-lg font-medium mb-4">Automatic Position Management</h3>

        <div className="space-y-4">
          <div className="flex items-center justify-between">
            <div>
              <label className="text-sm font-medium text-gray-700">Auto Stop Loss</label>
              <p className="text-xs text-gray-500">Automatically set stop loss orders</p>
            </div>
            <Switch
              checked={config.autoStopLoss}
              onChange={(checked) => updateConfig('autoStopLoss', checked)}
              className={`${config.autoStopLoss ? 'bg-blue-600' : 'bg-gray-300'} relative inline-flex h-6 w-11 items-center rounded-full transition-colors focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2`}
            >
              <span className={`${config.autoStopLoss ? 'translate-x-6' : 'translate-x-1'} inline-block h-4 w-4 transform rounded-full bg-white transition-transform`} />
            </Switch>
          </div>

          {config.autoStopLoss && (
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Stop Loss Percentage (%)
              </label>
              <input
                type="number"
                step="0.1"
                value={config.autoStopLossPercent}
                onChange={(e) => updateConfig('autoStopLossPercent', Number(e.target.value))}
                className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
              />
            </div>
          )}

          <div className="flex items-center justify-between">
            <div>
              <label className="text-sm font-medium text-gray-700">Auto Take Profit</label>
              <p className="text-xs text-gray-500">Automatically set take profit orders</p>
            </div>
            <Switch
              checked={config.autoTakeProfit}
              onChange={(checked) => updateConfig('autoTakeProfit', checked)}
              className={`${config.autoTakeProfit ? 'bg-blue-600' : 'bg-gray-300'} relative inline-flex h-6 w-11 items-center rounded-full transition-colors focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2`}
            >
              <span className={`${config.autoTakeProfit ? 'translate-x-6' : 'translate-x-1'} inline-block h-4 w-4 transform rounded-full bg-white transition-transform`} />
            </Switch>
          </div>

          {config.autoTakeProfit && (
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Take Profit Percentage (%)
              </label>
              <input
                type="number"
                step="0.1"
                value={config.autoTakeProfitPercent}
                onChange={(e) => updateConfig('autoTakeProfitPercent', Number(e.target.value))}
                className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
              />
            </div>
          )}
        </div>
      </div>
    </div>
  );

  return (
    <div className={`bg-white rounded-lg shadow-lg ${className}`}>
      <div className="p-6 border-b">
        <h2 className="text-xl font-bold flex items-center">
          <Settings className="w-6 h-6 mr-2" />
          Trading Configuration
        </h2>
        <p className="text-gray-600 mt-1">Configure your automated trading parameters</p>
      </div>

      {/* Tab Navigation */}
      <div className="border-b">
        <nav className="flex space-x-8 px-6">
          {[
            { id: 'budget', label: 'Budget & Sizing', icon: DollarSign },
            { id: 'risk', label: 'Risk Management', icon: Shield },
            { id: 'trading', label: 'Trading Preferences', icon: TrendingUp },
            { id: 'api', label: 'API Configuration', icon: Key },
          ].map(({ id, label, icon: Icon }) => (
            <button
              key={id}
              onClick={() => setActiveTab(id as any)}
              className={`py-4 px-2 border-b-2 font-medium text-sm flex items-center ${
                activeTab === id
                  ? 'border-blue-500 text-blue-600'
                  : 'border-transparent text-gray-500 hover:text-gray-700'
              }`}
            >
              <Icon className="w-4 h-4 mr-2" />
              {label}
            </button>
          ))}
        </nav>
      </div>

      {/* Tab Content */}
      <div className="p-6">
        {activeTab === 'budget' && renderBudgetTab()}
        {activeTab === 'risk' && renderRiskTab()}
        {/* Add other tabs as needed */}
      </div>

      {/* Save Button */}
      <div className="px-6 py-4 bg-gray-50 border-t flex justify-between items-center">
        <div className="flex items-center text-sm text-gray-600">
          <AlertTriangle className="w-4 h-4 mr-2" />
          Changes will take effect after saving and restarting the trading system
        </div>

        <div className="flex space-x-3">
          <button
            onClick={loadConfiguration}
            disabled={isLoading}
            className="px-4 py-2 text-gray-700 bg-white border border-gray-300 rounded-md hover:bg-gray-50 disabled:opacity-50"
          >
            <RefreshCw className="w-4 h-4 mr-2 inline" />
            Reset
          </button>

          <button
            onClick={saveConfiguration}
            disabled={isLoading}
            className="px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 disabled:opacity-50"
          >
            <Save className="w-4 h-4 mr-2 inline" />
            {isLoading ? 'Saving...' : 'Save Configuration'}
          </button>
        </div>
      </div>
    </div>
  );
};
