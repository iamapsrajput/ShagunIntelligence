import React from 'react';
import {
  Box,
  Typography,
  Grid,
  Chip,
  LinearProgress,
  Tooltip,
  IconButton,
  TextField,
  MenuItem
} from '@mui/material';
import {
  CheckCircle,
  Warning,
  Error as ErrorIcon,
  Refresh,
  TrendingUp,
  TrendingDown,
  Remove
} from '@mui/icons-material';

interface DataQualityOverviewProps {
  symbol: string;
  data: any;
  onSymbolChange: (symbol: string) => void;
}

const DataQualityOverview: React.FC<DataQualityOverviewProps> = ({
  symbol,
  data,
  onSymbolChange
}) => {
  const popularSymbols = [
    'RELIANCE.NS',
    'TCS.NS',
    'INFY.NS',
    'HDFC.NS',
    'ICICIBANK.NS',
    'SBIN.NS',
    'BHARTIARTL.NS',
    'HDFCBANK.NS'
  ];

  const getQualityIcon = (level: string) => {
    switch (level) {
      case 'HIGH':
        return <CheckCircle color="success" />;
      case 'MEDIUM':
        return <Warning color="warning" />;
      case 'LOW':
        return <ErrorIcon color="error" />;
      default:
        return <Remove color="disabled" />;
    }
  };

  const getQualityColor = (score: number): string => {
    if (score >= 0.8) return '#4caf50';
    if (score >= 0.6) return '#ff9800';
    return '#f44336';
  };

  const formatLatency = (ms: number): string => {
    if (ms < 1000) return `${ms.toFixed(0)}ms`;
    return `${(ms / 1000).toFixed(2)}s`;
  };

  const calculateTrend = (current: number, history: any[]): number => {
    if (!history || history.length < 2) return 0;
    const previous = history[history.length - 2]?.score || current;
    return ((current - previous) / previous) * 100;
  };

  if (!data) {
    return (
      <Box p={3} textAlign="center">
        <CircularProgress />
      </Box>
    );
  }

  const trend = calculateTrend(
    data.overall_quality_score,
    data.quality_history || []
  );

  return (
    <Box>
      <Box display="flex" justifyContent="space-between" alignItems="center" mb={3}>
        <Typography variant="h5" component="h2">
          Data Quality Overview
        </Typography>
        <Box display="flex" alignItems="center" gap={2}>
          <TextField
            select
            size="small"
            value={symbol}
            onChange={(e) => onSymbolChange(e.target.value)}
            sx={{ minWidth: 150 }}
          >
            {popularSymbols.map((sym) => (
              <MenuItem key={sym} value={sym}>
                {sym}
              </MenuItem>
            ))}
          </TextField>
          <IconButton size="small" onClick={() => window.location.reload()}>
            <Refresh />
          </IconButton>
        </Box>
      </Box>

      <Grid container spacing={3}>
        {/* Overall Quality Score */}
        <Grid item xs={12} md={4}>
          <Box
            sx={{
              p: 2,
              border: '1px solid',
              borderColor: 'divider',
              borderRadius: 2,
              position: 'relative'
            }}
          >
            <Typography variant="subtitle2" color="text.secondary" gutterBottom>
              Overall Quality Score
            </Typography>
            <Box display="flex" alignItems="center" gap={2}>
              <Typography variant="h3" component="div">
                {(data.overall_quality_score * 100).toFixed(1)}%
              </Typography>
              {getQualityIcon(data.overall_quality)}
              {trend !== 0 && (
                <Box display="flex" alignItems="center">
                  {trend > 0 ? (
                    <TrendingUp color="success" />
                  ) : (
                    <TrendingDown color="error" />
                  )}
                  <Typography
                    variant="body2"
                    color={trend > 0 ? 'success.main' : 'error.main'}
                  >
                    {Math.abs(trend).toFixed(1)}%
                  </Typography>
                </Box>
              )}
            </Box>
            <LinearProgress
              variant="determinate"
              value={data.overall_quality_score * 100}
              sx={{
                mt: 1,
                height: 8,
                borderRadius: 4,
                backgroundColor: 'grey.300',
                '& .MuiLinearProgress-bar': {
                  backgroundColor: getQualityColor(data.overall_quality_score),
                  borderRadius: 4
                }
              }}
            />
          </Box>
        </Grid>

        {/* Aggregated Metrics */}
        <Grid item xs={12} md={8}>
          <Box
            sx={{
              p: 2,
              border: '1px solid',
              borderColor: 'divider',
              borderRadius: 2
            }}
          >
            <Typography variant="subtitle2" color="text.secondary" gutterBottom>
              Aggregated Metrics
            </Typography>
            <Grid container spacing={2}>
              <Grid item xs={6} sm={3}>
                <Typography variant="body2" color="text.secondary">
                  Total Sources
                </Typography>
                <Typography variant="h6">
                  {data.aggregated_metrics?.total_sources || 0}
                </Typography>
              </Grid>
              <Grid item xs={6} sm={3}>
                <Typography variant="body2" color="text.secondary">
                  High Quality
                </Typography>
                <Typography variant="h6" color="success.main">
                  {data.aggregated_metrics?.high_quality_sources || 0}
                </Typography>
              </Grid>
              <Grid item xs={6} sm={3}>
                <Typography variant="body2" color="text.secondary">
                  Avg Latency
                </Typography>
                <Typography variant="h6">
                  {formatLatency(data.aggregated_metrics?.average_latency_ms || 0)}
                </Typography>
              </Grid>
              <Grid item xs={6} sm={3}>
                <Typography variant="body2" color="text.secondary">
                  Failovers
                </Typography>
                <Typography variant="h6" color="warning.main">
                  {data.aggregated_metrics?.failover_count || 0}
                </Typography>
              </Grid>
            </Grid>
          </Box>
        </Grid>

        {/* Source Quality Breakdown */}
        <Grid item xs={12}>
          <Box
            sx={{
              p: 2,
              border: '1px solid',
              borderColor: 'divider',
              borderRadius: 2
            }}
          >
            <Typography variant="subtitle2" color="text.secondary" gutterBottom>
              Source Quality Breakdown
            </Typography>
            <Grid container spacing={2} sx={{ mt: 1 }}>
              {Object.entries(data.sources || {}).map(([source, sourceData]: [string, any]) => (
                <Grid item xs={12} sm={6} md={4} key={source}>
                  <Box
                    sx={{
                      p: 2,
                      backgroundColor: 'background.default',
                      borderRadius: 1,
                      position: 'relative'
                    }}
                  >
                    <Box display="flex" justifyContent="space-between" alignItems="center">
                      <Typography variant="body1" fontWeight="medium">
                        {source}
                      </Typography>
                      <Chip
                        size="small"
                        label={sourceData.quality_level}
                        color={
                          sourceData.quality_level === 'HIGH' ? 'success' :
                          sourceData.quality_level === 'MEDIUM' ? 'warning' : 'error'
                        }
                      />
                    </Box>
                    <Box mt={1}>
                      <Box display="flex" justifyContent="space-between">
                        <Typography variant="body2" color="text.secondary">
                          Score
                        </Typography>
                        <Typography variant="body2">
                          {(sourceData.quality_score * 100).toFixed(1)}%
                        </Typography>
                      </Box>
                      <Box display="flex" justifyContent="space-between">
                        <Typography variant="body2" color="text.secondary">
                          Latency
                        </Typography>
                        <Typography variant="body2">
                          {formatLatency(sourceData.latency_ms || 0)}
                        </Typography>
                      </Box>
                      <Box display="flex" justifyContent="space-between">
                        <Typography variant="body2" color="text.secondary">
                          Status
                        </Typography>
                        <Typography
                          variant="body2"
                          color={sourceData.is_healthy ? 'success.main' : 'error.main'}
                        >
                          {sourceData.is_healthy ? 'Healthy' : 'Unhealthy'}
                        </Typography>
                      </Box>
                    </Box>
                    <LinearProgress
                      variant="determinate"
                      value={sourceData.quality_score * 100}
                      sx={{
                        mt: 1,
                        height: 4,
                        borderRadius: 2,
                        backgroundColor: 'grey.300',
                        '& .MuiLinearProgress-bar': {
                          backgroundColor: getQualityColor(sourceData.quality_score),
                          borderRadius: 2
                        }
                      }}
                    />
                  </Box>
                </Grid>
              ))}
            </Grid>
          </Box>
        </Grid>

        {/* Recent Failovers */}
        {data.recent_failovers && data.recent_failovers.length > 0 && (
          <Grid item xs={12}>
            <Box
              sx={{
                p: 2,
                border: '1px solid',
                borderColor: 'warning.main',
                borderRadius: 2,
                backgroundColor: 'warning.lighter'
              }}
            >
              <Typography variant="subtitle2" color="warning.dark" gutterBottom>
                Recent Failover Events
              </Typography>
              {data.recent_failovers.map((event: any, index: number) => (
                <Box key={index} sx={{ mt: 1 }}>
                  <Typography variant="body2">
                    {new Date(event.timestamp).toLocaleString()}:
                    Failed from <strong>{event.from_source}</strong> to <strong>{event.to_source}</strong>
                    {event.reason && ` - ${event.reason}`}
                  </Typography>
                </Box>
              ))}
            </Box>
          </Grid>
        )}
      </Grid>
    </Box>
  );
};

export default DataQualityOverview;
