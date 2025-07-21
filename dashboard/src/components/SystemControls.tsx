import React, { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import { 
  Power, 
  Settings, 
  Shield, 
  AlertTriangle, 
  Save,
  RefreshCw,
  Sliders,
  Activity,
  Pause,
  Play
} from 'lucide-react';
import { Switch } from '@headlessui/react';
import toast from 'react-hot-toast';
import { SystemStatus } from '@/services/websocket';
import wsService from '@/services/websocket';

interface SystemControlsProps {
  className?: string;
}

interface RiskParameters {
  maxPositionSize: number;
  maxPortfolioRisk: number;
  maxDailyLoss: number;
  stopLossPercent: number;
  takeProfitPercent: number;
  maxOpenPositions: number;
  allowShortSelling: boolean;
  useTrailingStop: boolean;
  trailingStopPercent: number;
}

interface AgentConfig {
  [key: string]: {
    enabled: boolean;
    weight: number;
    parameters: any;
  };
}

export const SystemControls: React.FC<SystemControlsProps> = ({ className = '' }) => {
  const [systemStatus, setSystemStatus] = useState<SystemStatus | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [riskParameters, setRiskParameters] = useState<RiskParameters>({
    maxPositionSize: 10,
    maxPortfolioRisk: 20,
    maxDailyLoss: 5,
    stopLossPercent: 2,
    takeProfitPercent: 4,
    maxOpenPositions: 5,
    allowShortSelling: false,
    useTrailingStop: true,
    trailingStopPercent: 1.5,
  });
  const [agentConfig, setAgentConfig] = useState<AgentConfig>({
    market: { enabled: true, weight: 0.25, parameters: {} },
    technical: { enabled: true, weight: 0.25, parameters: {} },
    sentiment: { enabled: true, weight: 0.25, parameters: {} },
    risk: { enabled: true, weight: 0.25, parameters: {} },
  });

  useEffect(() => {
    // Load initial system status and parameters
    loadSystemData();

    // Subscribe to system status updates
    const handleSystemStatus = (status: SystemStatus) => {
      setSystemStatus(status);
    };

    wsService.on('system:status', handleSystemStatus);

    return () => {
      wsService.off('system:status', handleSystemStatus);
    };
  }, []);

  const loadSystemData = async () => {
    try {
      const api = require('@/services/api').default;
      const [status, riskParams, agentStatus] = await Promise.all([
        api.getSystemStatus(),
        api.getRiskParameters(),
        api.getAgentStatus(),
      ]);

      setSystemStatus(status);
      setRiskParameters(riskParams);
      
      // Update agent config based on status
      const updatedConfig = { ...agentConfig };
      Object.keys(agentStatus).forEach(agent => {
        if (updatedConfig[agent]) {
          updatedConfig[agent].enabled = agentStatus[agent].enabled;
          updatedConfig[agent].parameters = agentStatus[agent].parameters || {};
        }
      });
      setAgentConfig(updatedConfig);
    } catch (error) {
      console.error('Failed to load system data:', error);
      toast.error('Failed to load system configuration');
    }
  };

  const toggleSystem = async () => {
    if (!systemStatus) return;

    setIsLoading(true);
    try {
      const newStatus = !systemStatus.isActive;
      wsService.toggleSystem(newStatus);
      
      // Optimistically update UI
      setSystemStatus({ ...systemStatus, isActive: newStatus });
      
      toast.success(newStatus ? 'Trading system activated' : 'Trading system deactivated');
    } catch (error) {
      console.error('Failed to toggle system:', error);
      toast.error('Failed to toggle system');
      // Revert on error
      setSystemStatus({ ...systemStatus, isActive: systemStatus.isActive });
    } finally {
      setIsLoading(false);
    }
  };

  const updateRiskParameters = async () => {
    setIsLoading(true);
    try {
      const api = require('@/services/api').default;
      await api.updateRiskParameters(riskParameters);
      wsService.updateRiskParameters(riskParameters);
      toast.success('Risk parameters updated successfully');
    } catch (error) {
      console.error('Failed to update risk parameters:', error);
      toast.error('Failed to update risk parameters');
    } finally {
      setIsLoading(false);
    }
  };

  const toggleAgent = async (agentType: string) => {
    const updatedConfig = { ...agentConfig };
    updatedConfig[agentType].enabled = !updatedConfig[agentType].enabled;
    
    setAgentConfig(updatedConfig);
    
    try {
      const api = require('@/services/api').default;
      await api.updateAgentConfig(agentType, updatedConfig[agentType]);
      toast.success(`${agentType} agent ${updatedConfig[agentType].enabled ? 'enabled' : 'disabled'}`);
    } catch (error) {
      console.error('Failed to toggle agent:', error);
      toast.error('Failed to update agent configuration');
      // Revert on error
      updatedConfig[agentType].enabled = !updatedConfig[agentType].enabled;
      setAgentConfig(updatedConfig);
    }
  };

  const updateAgentWeight = (agentType: string, weight: number) => {
    const updatedConfig = { ...agentConfig };
    updatedConfig[agentType].weight = weight;
    
    // Normalize weights to sum to 1
    const totalWeight = Object.values(updatedConfig).reduce((sum, cfg) => sum + cfg.weight, 0);
    Object.keys(updatedConfig).forEach(agent => {
      updatedConfig[agent].weight = updatedConfig[agent].weight / totalWeight;
    });
    
    setAgentConfig(updatedConfig);
  };

  if (!systemStatus) {
    return (
      <div className="flex items-center justify-center h-96">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-primary-500"></div>
      </div>
    );
  }

  return (
    <div className={`space-y-6 ${className}`}>
      {/* System Status */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="bg-white rounded-lg shadow-lg p-6"
      >
        <div className="flex items-center justify-between mb-6">
          <h2 className="text-xl font-bold flex items-center">
            <Power className="w-6 h-6 mr-2" />
            System Control
          </h2>
          <div className="flex items-center space-x-4">
            <span className={`text-sm font-medium ${
              systemStatus.isActive ? 'text-green-600' : 'text-gray-500'
            }`}>
              {systemStatus.isActive ? 'ACTIVE' : 'INACTIVE'}
            </span>
            <Switch
              checked={systemStatus.isActive}
              onChange={toggleSystem}
              disabled={isLoading}
              className={`${
                systemStatus.isActive ? 'bg-green-600' : 'bg-gray-300'
              } relative inline-flex h-6 w-11 items-center rounded-full transition-colors focus:outline-none focus:ring-2 focus:ring-primary-500 focus:ring-offset-2`}
            >
              <span
                className={`${
                  systemStatus.isActive ? 'translate-x-6' : 'translate-x-1'
                } inline-block h-4 w-4 transform rounded-full bg-white transition-transform`}
              />
            </Switch>
          </div>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <div className="flex items-center space-x-3">
            <Activity className={`w-5 h-5 ${systemStatus.isActive ? 'text-green-500' : 'text-gray-400'}`} />
            <div>
              <p className="text-sm text-gray-600">Active Agents</p>
              <p className="font-semibold">{systemStatus.activeAgents.length}</p>
            </div>
          </div>
          <div className="flex items-center space-x-3">
            <Shield className={`w-5 h-5 ${
              systemStatus.riskLevel === 'LOW' ? 'text-green-500' :
              systemStatus.riskLevel === 'MEDIUM' ? 'text-yellow-500' :
              'text-red-500'
            }`} />
            <div>
              <p className="text-sm text-gray-600">Risk Level</p>
              <p className="font-semibold">{systemStatus.riskLevel}</p>
            </div>
          </div>
          <div className="flex items-center space-x-3">
            <RefreshCw className="w-5 h-5 text-blue-500" />
            <div>
              <p className="text-sm text-gray-600">Last Update</p>
              <p className="font-semibold">
                {new Date(systemStatus.lastUpdate).toLocaleTimeString()}
              </p>
            </div>
          </div>
        </div>
      </motion.div>

      {/* Risk Parameters */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.1 }}
        className="bg-white rounded-lg shadow-lg p-6"
      >
        <h3 className="text-lg font-semibold mb-4 flex items-center">
          <Sliders className="w-5 h-5 mr-2" />
          Risk Parameters
        </h3>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              Max Position Size (%)
            </label>
            <input
              type="range"
              min="1"
              max="25"
              value={riskParameters.maxPositionSize}
              onChange={(e) => setRiskParameters({
                ...riskParameters,
                maxPositionSize: parseInt(e.target.value)
              })}
              className="w-full"
            />
            <span className="text-sm text-gray-600">{riskParameters.maxPositionSize}%</span>
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              Max Portfolio Risk (%)
            </label>
            <input
              type="range"
              min="5"
              max="50"
              value={riskParameters.maxPortfolioRisk}
              onChange={(e) => setRiskParameters({
                ...riskParameters,
                maxPortfolioRisk: parseInt(e.target.value)
              })}
              className="w-full"
            />
            <span className="text-sm text-gray-600">{riskParameters.maxPortfolioRisk}%</span>
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              Max Daily Loss (%)
            </label>
            <input
              type="range"
              min="1"
              max="10"
              step="0.5"
              value={riskParameters.maxDailyLoss}
              onChange={(e) => setRiskParameters({
                ...riskParameters,
                maxDailyLoss: parseFloat(e.target.value)
              })}
              className="w-full"
            />
            <span className="text-sm text-gray-600">{riskParameters.maxDailyLoss}%</span>
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              Stop Loss (%)
            </label>
            <input
              type="range"
              min="0.5"
              max="5"
              step="0.5"
              value={riskParameters.stopLossPercent}
              onChange={(e) => setRiskParameters({
                ...riskParameters,
                stopLossPercent: parseFloat(e.target.value)
              })}
              className="w-full"
            />
            <span className="text-sm text-gray-600">{riskParameters.stopLossPercent}%</span>
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              Take Profit (%)
            </label>
            <input
              type="range"
              min="1"
              max="10"
              step="0.5"
              value={riskParameters.takeProfitPercent}
              onChange={(e) => setRiskParameters({
                ...riskParameters,
                takeProfitPercent: parseFloat(e.target.value)
              })}
              className="w-full"
            />
            <span className="text-sm text-gray-600">{riskParameters.takeProfitPercent}%</span>
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              Max Open Positions
            </label>
            <input
              type="number"
              min="1"
              max="20"
              value={riskParameters.maxOpenPositions}
              onChange={(e) => setRiskParameters({
                ...riskParameters,
                maxOpenPositions: parseInt(e.target.value)
              })}
              className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-primary-500"
            />
          </div>
        </div>

        <div className="mt-4 space-y-3">
          <div className="flex items-center justify-between">
            <span className="text-sm font-medium text-gray-700">Allow Short Selling</span>
            <Switch
              checked={riskParameters.allowShortSelling}
              onChange={(checked) => setRiskParameters({
                ...riskParameters,
                allowShortSelling: checked
              })}
              className={`${
                riskParameters.allowShortSelling ? 'bg-primary-600' : 'bg-gray-300'
              } relative inline-flex h-6 w-11 items-center rounded-full transition-colors`}
            >
              <span
                className={`${
                  riskParameters.allowShortSelling ? 'translate-x-6' : 'translate-x-1'
                } inline-block h-4 w-4 transform rounded-full bg-white transition-transform`}
              />
            </Switch>
          </div>

          <div className="flex items-center justify-between">
            <span className="text-sm font-medium text-gray-700">Use Trailing Stop</span>
            <Switch
              checked={riskParameters.useTrailingStop}
              onChange={(checked) => setRiskParameters({
                ...riskParameters,
                useTrailingStop: checked
              })}
              className={`${
                riskParameters.useTrailingStop ? 'bg-primary-600' : 'bg-gray-300'
              } relative inline-flex h-6 w-11 items-center rounded-full transition-colors`}
            >
              <span
                className={`${
                  riskParameters.useTrailingStop ? 'translate-x-6' : 'translate-x-1'
                } inline-block h-4 w-4 transform rounded-full bg-white transition-transform`}
              />
            </Switch>
          </div>
        </div>

        <button
          onClick={updateRiskParameters}
          disabled={isLoading}
          className="mt-6 w-full bg-primary-600 text-white py-2 px-4 rounded-md hover:bg-primary-700 transition-colors flex items-center justify-center disabled:opacity-50"
        >
          <Save className="w-4 h-4 mr-2" />
          Save Risk Parameters
        </button>
      </motion.div>

      {/* Agent Configuration */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.2 }}
        className="bg-white rounded-lg shadow-lg p-6"
      >
        <h3 className="text-lg font-semibold mb-4 flex items-center">
          <Settings className="w-5 h-5 mr-2" />
          Agent Configuration
        </h3>

        <div className="space-y-4">
          {Object.entries(agentConfig).map(([agent, config]) => (
            <div key={agent} className="border rounded-lg p-4">
              <div className="flex items-center justify-between mb-2">
                <span className="font-medium capitalize">{agent} Agent</span>
                <Switch
                  checked={config.enabled}
                  onChange={() => toggleAgent(agent)}
                  className={`${
                    config.enabled ? 'bg-primary-600' : 'bg-gray-300'
                  } relative inline-flex h-6 w-11 items-center rounded-full transition-colors`}
                >
                  <span
                    className={`${
                      config.enabled ? 'translate-x-6' : 'translate-x-1'
                    } inline-block h-4 w-4 transform rounded-full bg-white transition-transform`}
                  />
                </Switch>
              </div>
              
              <div className="mt-2">
                <label className="block text-sm text-gray-600 mb-1">
                  Decision Weight: {(config.weight * 100).toFixed(0)}%
                </label>
                <input
                  type="range"
                  min="0"
                  max="100"
                  value={config.weight * 100}
                  onChange={(e) => updateAgentWeight(agent, parseInt(e.target.value) / 100)}
                  disabled={!config.enabled}
                  className="w-full"
                />
              </div>
            </div>
          ))}
        </div>
      </motion.div>
    </div>
  );
};