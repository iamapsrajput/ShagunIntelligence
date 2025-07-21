import React, { useState, useEffect } from 'react';
import {
  Box,
  Typography,
  List,
  ListItem,
  ListItemIcon,
  ListItemText,
  ListItemSecondaryAction,
  IconButton,
  Chip,
  Button,
  Badge,
  Divider,
  TextField,
  MenuItem,
  Paper
} from '@mui/material';
import {
  Error as ErrorIcon,
  Warning,
  Info,
  CheckCircle,
  Clear,
  FilterList,
  Refresh,
  NotificationsActive
} from '@mui/icons-material';
import { formatDistanceToNow } from 'date-fns';
import { api } from '../../services/api';

interface QualityAlert {
  id: string;
  severity: 'critical' | 'error' | 'warning' | 'info';
  type: string;
  message: string;
  source: string;
  symbol?: string;
  timestamp: string;
  acknowledged: boolean;
  acknowledgedBy?: string;
  acknowledgedAt?: string;
  details?: any;
}

interface QualityAlertsPanelProps {
  showFull?: boolean;
}

const QualityAlertsPanel: React.FC<QualityAlertsPanelProps> = ({ showFull = false }) => {
  const [alerts, setAlerts] = useState<QualityAlert[]>([]);
  const [loading, setLoading] = useState(true);
  const [filter, setFilter] = useState<'all' | 'active' | 'acknowledged'>('active');
  const [severityFilter, setSeverityFilter] = useState<string>('all');

  useEffect(() => {
    loadAlerts();
    // Refresh alerts every 30 seconds
    const interval = setInterval(loadAlerts, 30000);
    return () => clearInterval(interval);
  }, []);

  const loadAlerts = async () => {
    try {
      const response = await api.get('/data-quality/quality-alerts', {
        params: {
          active_only: filter === 'active',
          severity: severityFilter !== 'all' ? severityFilter : undefined,
          limit: showFull ? 200 : 50
        }
      });
      setAlerts(response.data);
    } catch (error) {
      console.error('Error loading alerts:', error);
    } finally {
      setLoading(false);
    }
  };

  const handleAcknowledge = async (alertId: string) => {
    try {
      await api.post(`/data-quality/quality-alerts/${alertId}/acknowledge`);
      setAlerts(alerts.map(alert => 
        alert.id === alertId 
          ? { ...alert, acknowledged: true, acknowledgedAt: new Date().toISOString() }
          : alert
      ));
    } catch (error) {
      console.error('Error acknowledging alert:', error);
    }
  };

  const getSeverityIcon = (severity: string) => {
    switch (severity) {
      case 'critical':
      case 'error':
        return <ErrorIcon color="error" />;
      case 'warning':
        return <Warning color="warning" />;
      case 'info':
        return <Info color="info" />;
      default:
        return <CheckCircle color="success" />;
    }
  };

  const getSeverityColor = (severity: string): any => {
    switch (severity) {
      case 'critical':
      case 'error':
        return 'error';
      case 'warning':
        return 'warning';
      case 'info':
        return 'info';
      default:
        return 'default';
    }
  };

  const getFilteredAlerts = () => {
    return alerts.filter(alert => {
      if (filter === 'active' && alert.acknowledged) return false;
      if (filter === 'acknowledged' && !alert.acknowledged) return false;
      if (severityFilter !== 'all' && alert.severity !== severityFilter) return false;
      return true;
    });
  };

  const activeAlertCount = alerts.filter(a => !a.acknowledged).length;
  const criticalCount = alerts.filter(a => !a.acknowledged && (a.severity === 'critical' || a.severity === 'error')).length;

  return (
    <Box>
      <Box display="flex" justifyContent="space-between" alignItems="center" mb={2}>
        <Box display="flex" alignItems="center" gap={1}>
          <Typography variant={showFull ? "h5" : "h6"} component="h2">
            Quality Alerts
          </Typography>
          {activeAlertCount > 0 && (
            <Badge badgeContent={activeAlertCount} color={criticalCount > 0 ? "error" : "warning"}>
              <NotificationsActive color={criticalCount > 0 ? "error" : "warning"} />
            </Badge>
          )}
        </Box>
        <Box display="flex" gap={1}>
          {showFull && (
            <>
              <TextField
                select
                size="small"
                value={filter}
                onChange={(e) => setFilter(e.target.value as any)}
                sx={{ minWidth: 120 }}
              >
                <MenuItem value="all">All</MenuItem>
                <MenuItem value="active">Active</MenuItem>
                <MenuItem value="acknowledged">Acknowledged</MenuItem>
              </TextField>
              <TextField
                select
                size="small"
                value={severityFilter}
                onChange={(e) => setSeverityFilter(e.target.value)}
                sx={{ minWidth: 100 }}
              >
                <MenuItem value="all">All</MenuItem>
                <MenuItem value="critical">Critical</MenuItem>
                <MenuItem value="error">Error</MenuItem>
                <MenuItem value="warning">Warning</MenuItem>
                <MenuItem value="info">Info</MenuItem>
              </TextField>
            </>
          )}
          <IconButton size="small" onClick={loadAlerts}>
            <Refresh />
          </IconButton>
        </Box>
      </Box>

      <Paper elevation={0} sx={{ maxHeight: showFull ? 'none' : 400, overflow: 'auto' }}>
        <List>
          {getFilteredAlerts().length === 0 ? (
            <ListItem>
              <ListItemText
                primary={
                  <Typography variant="body2" color="text.secondary" textAlign="center">
                    No alerts to display
                  </Typography>
                }
              />
            </ListItem>
          ) : (
            getFilteredAlerts().map((alert, index) => (
              <React.Fragment key={alert.id}>
                {index > 0 && <Divider />}
                <ListItem
                  sx={{
                    opacity: alert.acknowledged ? 0.6 : 1,
                    backgroundColor: alert.acknowledged ? 'transparent' : 
                      alert.severity === 'critical' || alert.severity === 'error' 
                        ? 'error.lighter' 
                        : 'transparent'
                  }}
                >
                  <ListItemIcon>
                    {getSeverityIcon(alert.severity)}
                  </ListItemIcon>
                  <ListItemText
                    primary={
                      <Box display="flex" alignItems="center" gap={1}>
                        <Typography variant="body2" fontWeight="medium">
                          {alert.message}
                        </Typography>
                        <Chip
                          label={alert.severity}
                          size="small"
                          color={getSeverityColor(alert.severity)}
                        />
                        {alert.source && (
                          <Chip
                            label={alert.source}
                            size="small"
                            variant="outlined"
                          />
                        )}
                        {alert.symbol && (
                          <Chip
                            label={alert.symbol}
                            size="small"
                            variant="outlined"
                            color="primary"
                          />
                        )}
                      </Box>
                    }
                    secondary={
                      <Box>
                        <Typography variant="caption" color="text.secondary">
                          {formatDistanceToNow(new Date(alert.timestamp), { addSuffix: true })}
                          {alert.acknowledged && alert.acknowledgedAt && (
                            <> • Acknowledged {formatDistanceToNow(new Date(alert.acknowledgedAt), { addSuffix: true })}</>
                          )}
                        </Typography>
                        {alert.details && (
                          <Typography variant="caption" display="block" sx={{ mt: 0.5 }}>
                            {JSON.stringify(alert.details).substring(0, 100)}...
                          </Typography>
                        )}
                      </Box>
                    }
                  />
                  <ListItemSecondaryAction>
                    {!alert.acknowledged && (
                      <IconButton
                        edge="end"
                        size="small"
                        onClick={() => handleAcknowledge(alert.id)}
                        title="Acknowledge alert"
                      >
                        <CheckCircle />
                      </IconButton>
                    )}
                  </ListItemSecondaryAction>
                </ListItem>
              </React.Fragment>
            ))
          )}
        </List>
      </Paper>

      {!showFull && alerts.length > 5 && (
        <Box mt={2} textAlign="center">
          <Button
            size="small"
            color="primary"
            onClick={() => {/* Navigate to full alerts view */}}
          >
            View All Alerts ({alerts.length})
          </Button>
        </Box>
      )}
    </Box>
  );
};

export default QualityAlertsPanel;