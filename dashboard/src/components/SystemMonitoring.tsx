import React, { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import {
  Monitor,
  Cpu,
  Database,
  Wifi,
  AlertTriangle,
  CheckCircle,
  Clock,
  TrendingUp,
  Activity,
  Zap,
  Shield,
  Globe
} from 'lucide-react';

interface SystemMetrics {
  cpu_usage: number;
  memory_usage: number;
  disk_usage: number;
  network_latency: number;
  api_response_time: number;
  active_connections: number;
  error_rate: number;
  uptime: number;
}

interface ServiceStatus {
  name: string;
  status: 'healthy' | 'warning' | 'error' | 'unknown';
  response_time: number;
  last_check: string;
  message: string;
}

interface SystemMonitoringProps {
  className?: string;
}

export const SystemMonitoring: React.FC<SystemMonitoringProps> = ({ className = '' }) => {
  const [metrics, setMetrics] = useState<SystemMetrics>({
    cpu_usage: 0,
    memory_usage: 0,
    disk_usage: 0,
    network_latency: 0,
    api_response_time: 0,
    active_connections: 0,
    error_rate: 0,
    uptime: 0,
  });

  const [services, setServices] = useState<ServiceStatus[]>([
    { name: 'Kite API', status: 'unknown', response_time: 0, last_check: '', message: '' },
    { name: 'Database', status: 'unknown', response_time: 0, last_check: '', message: '' },
    { name: 'WebSocket', status: 'unknown', response_time: 0, last_check: '', message: '' },
    { name: 'AI Agents', status: 'unknown', response_time: 0, last_check: '', message: '' },
    { name: 'Data Pipeline', status: 'unknown', response_time: 0, last_check: '', message: '' },
    { name: 'Risk Manager', status: 'unknown', response_time: 0, last_check: '', message: '' },
  ]);

  const [alerts, setAlerts] = useState<Array<{
    id: string;
    level: 'info' | 'warning' | 'error';
    message: string;
    timestamp: string;
  }>>([]);

  useEffect(() => {
    const fetchSystemMetrics = async () => {
      try {
        const response = await fetch('/api/v1/system/metrics');
        if (response.ok) {
          const data = await response.json();
          setMetrics(data.metrics);
          setServices(data.services);
          setAlerts(data.alerts || []);
        }
      } catch (error) {
        console.error('Failed to fetch system metrics:', error);
      }
    };

    // Initial fetch
    fetchSystemMetrics();

    // Set up polling every 5 seconds
    const interval = setInterval(fetchSystemMetrics, 5000);

    return () => clearInterval(interval);
  }, []);

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'healthy':
        return 'text-green-600 bg-green-100';
      case 'warning':
        return 'text-yellow-600 bg-yellow-100';
      case 'error':
        return 'text-red-600 bg-red-100';
      default:
        return 'text-gray-600 bg-gray-100';
    }
  };

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'healthy':
        return <CheckCircle className="w-4 h-4" />;
      case 'warning':
        return <AlertTriangle className="w-4 h-4" />;
      case 'error':
        return <AlertTriangle className="w-4 h-4" />;
      default:
        return <Clock className="w-4 h-4" />;
    }
  };

  const getMetricColor = (value: number, thresholds: { warning: number; error: number }) => {
    if (value >= thresholds.error) return 'text-red-600';
    if (value >= thresholds.warning) return 'text-yellow-600';
    return 'text-green-600';
  };

  const formatUptime = (seconds: number) => {
    const days = Math.floor(seconds / 86400);
    const hours = Math.floor((seconds % 86400) / 3600);
    const minutes = Math.floor((seconds % 3600) / 60);

    if (days > 0) return `${days}d ${hours}h ${minutes}m`;
    if (hours > 0) return `${hours}h ${minutes}m`;
    return `${minutes}m`;
  };

  return (
    <div className={`bg-white rounded-lg shadow-lg ${className}`}>
      <div className="p-6 border-b">
        <div className="flex items-center">
          <Monitor className="w-6 h-6 mr-2 text-blue-600" />
          <h2 className="text-xl font-bold">System Monitoring</h2>
          <div className="ml-auto flex items-center text-sm text-gray-600">
            <Activity className="w-4 h-4 mr-1" />
            <span>Uptime: {formatUptime(metrics.uptime)}</span>
          </div>
        </div>
      </div>

      <div className="p-6">
        {/* System Metrics */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 mb-6">
          <div className="bg-gray-50 p-4 rounded-lg">
            <div className="flex items-center justify-between">
              <div className="flex items-center">
                <Cpu className="w-5 h-5 mr-2 text-blue-600" />
                <span className="text-sm font-medium">CPU Usage</span>
              </div>
              <span className={`text-lg font-bold ${getMetricColor(metrics.cpu_usage, { warning: 70, error: 90 })}`}>
                {metrics.cpu_usage.toFixed(1)}%
              </span>
            </div>
            <div className="mt-2 bg-gray-200 rounded-full h-2">
              <div
                className={`h-2 rounded-full transition-all duration-300 ${
                  metrics.cpu_usage >= 90 ? 'bg-red-500' :
                  metrics.cpu_usage >= 70 ? 'bg-yellow-500' : 'bg-green-500'
                }`}
                style={{ width: `${Math.min(metrics.cpu_usage, 100)}%` }}
              />
            </div>
          </div>

          <div className="bg-gray-50 p-4 rounded-lg">
            <div className="flex items-center justify-between">
              <div className="flex items-center">
                <Database className="w-5 h-5 mr-2 text-green-600" />
                <span className="text-sm font-medium">Memory</span>
              </div>
              <span className={`text-lg font-bold ${getMetricColor(metrics.memory_usage, { warning: 80, error: 95 })}`}>
                {metrics.memory_usage.toFixed(1)}%
              </span>
            </div>
            <div className="mt-2 bg-gray-200 rounded-full h-2">
              <div
                className={`h-2 rounded-full transition-all duration-300 ${
                  metrics.memory_usage >= 95 ? 'bg-red-500' :
                  metrics.memory_usage >= 80 ? 'bg-yellow-500' : 'bg-green-500'
                }`}
                style={{ width: `${Math.min(metrics.memory_usage, 100)}%` }}
              />
            </div>
          </div>

          <div className="bg-gray-50 p-4 rounded-lg">
            <div className="flex items-center justify-between">
              <div className="flex items-center">
                <Wifi className="w-5 h-5 mr-2 text-purple-600" />
                <span className="text-sm font-medium">API Latency</span>
              </div>
              <span className={`text-lg font-bold ${getMetricColor(metrics.api_response_time, { warning: 500, error: 1000 })}`}>
                {metrics.api_response_time.toFixed(0)}ms
              </span>
            </div>
          </div>

          <div className="bg-gray-50 p-4 rounded-lg">
            <div className="flex items-center justify-between">
              <div className="flex items-center">
                <TrendingUp className="w-5 h-5 mr-2 text-orange-600" />
                <span className="text-sm font-medium">Error Rate</span>
              </div>
              <span className={`text-lg font-bold ${getMetricColor(metrics.error_rate, { warning: 1, error: 5 })}`}>
                {metrics.error_rate.toFixed(2)}%
              </span>
            </div>
          </div>
        </div>

        {/* Service Status */}
        <div className="mb-6">
          <h3 className="text-lg font-semibold mb-4 flex items-center">
            <Shield className="w-5 h-5 mr-2" />
            Service Status
          </h3>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {services.map((service) => (
              <motion.div
                key={service.name}
                className="border rounded-lg p-4 hover:shadow-md transition-shadow"
                whileHover={{ scale: 1.02 }}
              >
                <div className="flex items-center justify-between mb-2">
                  <span className="font-medium">{service.name}</span>
                  <div className={`px-2 py-1 rounded-full text-xs flex items-center ${getStatusColor(service.status)}`}>
                    {getStatusIcon(service.status)}
                    <span className="ml-1 capitalize">{service.status}</span>
                  </div>
                </div>
                <div className="text-sm text-gray-600">
                  <div className="flex justify-between">
                    <span>Response Time:</span>
                    <span>{service.response_time}ms</span>
                  </div>
                  {service.message && (
                    <div className="mt-1 text-xs">{service.message}</div>
                  )}
                </div>
              </motion.div>
            ))}
          </div>
        </div>

        {/* Recent Alerts */}
        {alerts.length > 0 && (
          <div>
            <h3 className="text-lg font-semibold mb-4 flex items-center">
              <AlertTriangle className="w-5 h-5 mr-2" />
              Recent Alerts
            </h3>
            <div className="space-y-2 max-h-48 overflow-y-auto">
              {alerts.slice(0, 10).map((alert) => (
                <div
                  key={alert.id}
                  className={`p-3 rounded-lg border-l-4 ${
                    alert.level === 'error' ? 'border-red-500 bg-red-50' :
                    alert.level === 'warning' ? 'border-yellow-500 bg-yellow-50' :
                    'border-blue-500 bg-blue-50'
                  }`}
                >
                  <div className="flex items-center justify-between">
                    <span className="text-sm font-medium">{alert.message}</span>
                    <span className="text-xs text-gray-500">
                      {new Date(alert.timestamp).toLocaleTimeString()}
                    </span>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );
};
