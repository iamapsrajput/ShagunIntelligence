import React, { useEffect, useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  Brain, 
  TrendingUp, 
  Shield, 
  BarChart3, 
  MessageSquare,
  Activity,
  AlertCircle,
  CheckCircle,
  XCircle
} from 'lucide-react';
import { format } from 'date-fns';
import { AgentActivity } from '@/services/websocket';
import wsService from '@/services/websocket';

interface AgentPanelProps {
  agentType: 'market' | 'technical' | 'sentiment' | 'risk' | 'coordinator';
  className?: string;
}

const agentConfig = {
  market: {
    title: 'Market Analyst',
    icon: TrendingUp,
    color: 'blue',
    description: 'Analyzes market trends and patterns',
  },
  technical: {
    title: 'Technical Indicator',
    icon: BarChart3,
    color: 'purple',
    description: 'Processes technical indicators and signals',
  },
  sentiment: {
    title: 'Sentiment Analyst',
    icon: MessageSquare,
    color: 'green',
    description: 'Evaluates market sentiment and news',
  },
  risk: {
    title: 'Risk Manager',
    icon: Shield,
    color: 'orange',
    description: 'Monitors and manages portfolio risk',
  },
  coordinator: {
    title: 'Coordinator',
    icon: Brain,
    color: 'indigo',
    description: 'Orchestrates agent decisions',
  },
};

export const AgentPanel: React.FC<AgentPanelProps> = ({ agentType, className = '' }) => {
  const [activities, setActivities] = useState<AgentActivity[]>([]);
  const [status, setStatus] = useState<'active' | 'idle' | 'error'>('idle');
  const [latestAnalysis, setLatestAnalysis] = useState<any>(null);
  
  const config = agentConfig[agentType];
  const Icon = config.icon;

  useEffect(() => {
    // Load initial agent data
    loadAgentData();

    // Subscribe to agent activities
    const handleAgentActivity = (activity: AgentActivity) => {
      if (activity.agentType === agentType) {
        setActivities(prev => [activity, ...prev].slice(0, 10)); // Keep last 10 activities
        setStatus('active');
        setLatestAnalysis(activity.analysis);
        
        // Set back to idle after 2 seconds
        setTimeout(() => setStatus('idle'), 2000);
      }
    };

    wsService.on('agent:activity', handleAgentActivity);

    return () => {
      wsService.off('agent:activity', handleAgentActivity);
    };
  }, [agentType]);

  const loadAgentData = async () => {
    try {
      const api = require('@/services/api').default;
      const data = await api.getAgentAnalysis(agentType);
      setLatestAnalysis(data.analysis);
      setActivities(data.recentActivities || []);
    } catch (error) {
      console.error('Failed to load agent data:', error);
      setStatus('error');
    }
  };

  const getStatusIcon = () => {
    switch (status) {
      case 'active':
        return <Activity className="w-4 h-4 text-green-500 animate-pulse" />;
      case 'error':
        return <XCircle className="w-4 h-4 text-red-500" />;
      default:
        return <CheckCircle className="w-4 h-4 text-gray-400" />;
    }
  };

  const getConfidenceColor = (confidence: number) => {
    if (confidence >= 0.8) return 'text-green-600';
    if (confidence >= 0.6) return 'text-yellow-600';
    return 'text-red-600';
  };

  const renderAnalysisContent = () => {
    if (!latestAnalysis) return null;

    switch (agentType) {
      case 'market':
        return (
          <div className="space-y-2">
            <div className="flex justify-between items-center">
              <span className="text-sm text-gray-600">Trend</span>
              <span className={`font-semibold ${
                latestAnalysis.trend === 'bullish' ? 'text-green-600' : 
                latestAnalysis.trend === 'bearish' ? 'text-red-600' : 'text-gray-600'
              }`}>
                {latestAnalysis.trend?.toUpperCase()}
              </span>
            </div>
            <div className="flex justify-between items-center">
              <span className="text-sm text-gray-600">Strength</span>
              <div className="flex items-center space-x-1">
                {[...Array(5)].map((_, i) => (
                  <div
                    key={i}
                    className={`w-2 h-2 rounded-full ${
                      i < (latestAnalysis.strength || 0) ? 'bg-blue-500' : 'bg-gray-300'
                    }`}
                  />
                ))}
              </div>
            </div>
            {latestAnalysis.keyLevels && (
              <div className="mt-2 p-2 bg-gray-50 rounded">
                <div className="text-xs text-gray-600">Key Levels</div>
                <div className="flex justify-between text-sm mt-1">
                  <span>S: ${latestAnalysis.keyLevels.support}</span>
                  <span>R: ${latestAnalysis.keyLevels.resistance}</span>
                </div>
              </div>
            )}
          </div>
        );

      case 'technical':
        return (
          <div className="space-y-2">
            {latestAnalysis.indicators && Object.entries(latestAnalysis.indicators).map(([key, value]: [string, any]) => (
              <div key={key} className="flex justify-between items-center">
                <span className="text-sm text-gray-600">{key}</span>
                <span className="text-sm font-medium">{value}</span>
              </div>
            ))}
            {latestAnalysis.signals && (
              <div className="mt-2 space-y-1">
                {latestAnalysis.signals.map((signal: any, idx: number) => (
                  <div key={idx} className="flex items-center space-x-2">
                    <div className={`w-2 h-2 rounded-full ${
                      signal.type === 'buy' ? 'bg-green-500' : 'bg-red-500'
                    }`} />
                    <span className="text-xs">{signal.indicator}: {signal.type}</span>
                  </div>
                ))}
              </div>
            )}
          </div>
        );

      case 'sentiment':
        return (
          <div className="space-y-2">
            <div className="flex justify-between items-center">
              <span className="text-sm text-gray-600">Overall</span>
              <span className={`font-semibold ${
                latestAnalysis.sentiment > 0.5 ? 'text-green-600' : 
                latestAnalysis.sentiment < -0.5 ? 'text-red-600' : 'text-gray-600'
              }`}>
                {latestAnalysis.sentiment > 0.5 ? 'POSITIVE' : 
                 latestAnalysis.sentiment < -0.5 ? 'NEGATIVE' : 'NEUTRAL'}
              </span>
            </div>
            <div className="flex justify-between items-center">
              <span className="text-sm text-gray-600">Score</span>
              <span className="text-sm font-medium">
                {(latestAnalysis.sentiment || 0).toFixed(2)}
              </span>
            </div>
            {latestAnalysis.sources && (
              <div className="mt-2 space-y-1">
                <div className="text-xs text-gray-600">Sources</div>
                {Object.entries(latestAnalysis.sources).map(([source, score]: [string, any]) => (
                  <div key={source} className="flex justify-between text-xs">
                    <span>{source}</span>
                    <span className={score > 0 ? 'text-green-600' : 'text-red-600'}>
                      {score.toFixed(2)}
                    </span>
                  </div>
                ))}
              </div>
            )}
          </div>
        );

      case 'risk':
        return (
          <div className="space-y-2">
            <div className="flex justify-between items-center">
              <span className="text-sm text-gray-600">Risk Level</span>
              <span className={`font-semibold ${
                latestAnalysis.riskLevel === 'low' ? 'text-green-600' : 
                latestAnalysis.riskLevel === 'high' ? 'text-red-600' : 'text-yellow-600'
              }`}>
                {latestAnalysis.riskLevel?.toUpperCase()}
              </span>
            </div>
            <div className="flex justify-between items-center">
              <span className="text-sm text-gray-600">Score</span>
              <span className="text-sm font-medium">
                {latestAnalysis.riskScore}/10
              </span>
            </div>
            {latestAnalysis.metrics && (
              <div className="mt-2 space-y-1">
                <div className="flex justify-between text-xs">
                  <span>VaR (95%)</span>
                  <span className="text-red-600">
                    -${latestAnalysis.metrics.var95?.toFixed(2)}
                  </span>
                </div>
                <div className="flex justify-between text-xs">
                  <span>Sharpe Ratio</span>
                  <span>{latestAnalysis.metrics.sharpeRatio?.toFixed(2)}</span>
                </div>
                <div className="flex justify-between text-xs">
                  <span>Max Drawdown</span>
                  <span className="text-red-600">
                    {latestAnalysis.metrics.maxDrawdown?.toFixed(1)}%
                  </span>
                </div>
              </div>
            )}
          </div>
        );

      case 'coordinator':
        return (
          <div className="space-y-2">
            <div className="flex justify-between items-center">
              <span className="text-sm text-gray-600">Decision</span>
              <span className={`font-semibold ${
                latestAnalysis.decision === 'buy' ? 'text-green-600' : 
                latestAnalysis.decision === 'sell' ? 'text-red-600' : 'text-gray-600'
              }`}>
                {latestAnalysis.decision?.toUpperCase()}
              </span>
            </div>
            <div className="flex justify-between items-center">
              <span className="text-sm text-gray-600">Consensus</span>
              <span className={`text-sm font-medium ${getConfidenceColor(latestAnalysis.consensus || 0)}`}>
                {((latestAnalysis.consensus || 0) * 100).toFixed(0)}%
              </span>
            </div>
            {latestAnalysis.agentVotes && (
              <div className="mt-2 space-y-1">
                <div className="text-xs text-gray-600">Agent Votes</div>
                {Object.entries(latestAnalysis.agentVotes).map(([agent, vote]: [string, any]) => (
                  <div key={agent} className="flex justify-between text-xs">
                    <span>{agent}</span>
                    <span className={
                      vote === 'buy' ? 'text-green-600' : 
                      vote === 'sell' ? 'text-red-600' : 'text-gray-600'
                    }>
                      {vote}
                    </span>
                  </div>
                ))}
              </div>
            )}
          </div>
        );

      default:
        return null;
    }
  };

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      className={`bg-white rounded-lg shadow-lg p-6 ${className}`}
    >
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center space-x-3">
          <div className={`p-2 rounded-lg bg-${config.color}-100`}>
            <Icon className={`w-6 h-6 text-${config.color}-600`} />
          </div>
          <div>
            <h3 className="font-semibold text-lg">{config.title}</h3>
            <p className="text-sm text-gray-500">{config.description}</p>
          </div>
        </div>
        {getStatusIcon()}
      </div>

      <div className="mb-4">
        {renderAnalysisContent()}
      </div>

      <div className="border-t pt-4">
        <h4 className="text-sm font-semibold text-gray-700 mb-2">Recent Activities</h4>
        <div className="space-y-2 max-h-40 overflow-y-auto">
          <AnimatePresence>
            {activities.map((activity, idx) => (
              <motion.div
                key={`${activity.timestamp}-${idx}`}
                initial={{ opacity: 0, x: -20 }}
                animate={{ opacity: 1, x: 0 }}
                exit={{ opacity: 0, x: 20 }}
                transition={{ duration: 0.2 }}
                className="flex items-center justify-between text-sm"
              >
                <div className="flex items-center space-x-2">
                  <div className={`w-2 h-2 rounded-full ${
                    activity.action === 'buy' ? 'bg-green-500' : 
                    activity.action === 'sell' ? 'bg-red-500' : 'bg-gray-400'
                  }`} />
                  <span className="text-gray-600">{activity.action}</span>
                </div>
                <div className="flex items-center space-x-2">
                  <span className={`text-xs ${getConfidenceColor(activity.confidence)}`}>
                    {(activity.confidence * 100).toFixed(0)}%
                  </span>
                  <span className="text-xs text-gray-400">
                    {format(new Date(activity.timestamp), 'HH:mm:ss')}
                  </span>
                </div>
              </motion.div>
            ))}
          </AnimatePresence>
          {activities.length === 0 && (
            <div className="text-sm text-gray-400 text-center py-2">
              No recent activities
            </div>
          )}
        </div>
      </div>
    </motion.div>
  );
};