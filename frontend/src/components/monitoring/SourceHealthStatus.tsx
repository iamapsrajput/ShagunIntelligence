import React from 'react';
import {
  Box,
  Typography,
  Grid,
  Paper,
  Chip,
  LinearProgress,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  IconButton,
  Tooltip
} from '@mui/material';
import {
  CheckCircle,
  Warning,
  Error as ErrorIcon,
  InfoOutlined,
  Refresh
} from '@mui/icons-material';
import { PieChart, Pie, Cell, ResponsiveContainer, Tooltip as RechartsTooltip } from 'recharts';

interface SourceHealthStatusProps {
  data: any;
}

const SourceHealthStatus: React.FC<SourceHealthStatusProps> = ({ data }) => {
  const getHealthIcon = (status: string) => {
    switch (status) {
      case 'healthy':
        return <CheckCircle color="success" fontSize="small" />;
      case 'degraded':
        return <Warning color="warning" fontSize="small" />;
      case 'unhealthy':
        return <ErrorIcon color="error" fontSize="small" />;
      default:
        return <InfoOutlined color="disabled" fontSize="small" />;
    }
  };

  const getHealthColor = (status: string): string => {
    switch (status) {
      case 'healthy':
        return '#4caf50';
      case 'degraded':
        return '#ff9800';
      case 'unhealthy':
        return '#f44336';
      default:
        return '#9e9e9e';
    }
  };

  const formatUptime = (uptime: number): string => {
    return `${(uptime * 100).toFixed(1)}%`;
  };

  const formatCost = (cost: number): string => {
    return `$${cost.toFixed(2)}`;
  };

  if (!data) {
    return (
      <Box p={3} textAlign="center">
        <Typography>Loading health status...</Typography>
      </Box>
    );
  }

  const pieData = [
    { name: 'Healthy', value: data.summary?.healthy_sources || 0, color: '#4caf50' },
    { name: 'Degraded', value: data.summary?.degraded_sources || 0, color: '#ff9800' },
    { name: 'Unhealthy', value: data.summary?.unhealthy_sources || 0, color: '#f44336' }
  ];

  return (
    <Box>
      <Box display="flex" justifyContent="space-between" alignItems="center" mb={3}>
        <Typography variant="h5" component="h2">
          Data Source Health Status
        </Typography>
        <IconButton size="small" onClick={() => window.location.reload()}>
          <Refresh />
        </IconButton>
      </Box>

      <Grid container spacing={3}>
        {/* Summary Cards */}
        <Grid item xs={12} md={8}>
          <Grid container spacing={2}>
            <Grid item xs={6} sm={3}>
              <Paper sx={{ p: 2, textAlign: 'center' }}>
                <Typography variant="h4" color="primary">
                  {data.summary?.total_sources || 0}
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  Total Sources
                </Typography>
              </Paper>
            </Grid>
            <Grid item xs={6} sm={3}>
              <Paper sx={{ p: 2, textAlign: 'center' }}>
                <Typography variant="h4" color="success.main">
                  {data.summary?.healthy_sources || 0}
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  Healthy
                </Typography>
              </Paper>
            </Grid>
            <Grid item xs={6} sm={3}>
              <Paper sx={{ p: 2, textAlign: 'center' }}>
                <Typography variant="h4" color="warning.main">
                  {data.summary?.degraded_sources || 0}
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  Degraded
                </Typography>
              </Paper>
            </Grid>
            <Grid item xs={6} sm={3}>
              <Paper sx={{ p: 2, textAlign: 'center' }}>
                <Typography variant="h4" color="error.main">
                  {data.summary?.unhealthy_sources || 0}
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  Unhealthy
                </Typography>
              </Paper>
            </Grid>
          </Grid>
        </Grid>

        {/* Health Score Chart */}
        <Grid item xs={12} md={4}>
          <Paper sx={{ p: 2 }}>
            <Typography variant="subtitle1" gutterBottom>
              Overall Health Score
            </Typography>
            <Box display="flex" alignItems="center" justifyContent="center">
              <ResponsiveContainer width={150} height={150}>
                <PieChart>
                  <Pie
                    data={pieData}
                    cx="50%"
                    cy="50%"
                    innerRadius={40}
                    outerRadius={60}
                    paddingAngle={5}
                    dataKey="value"
                  >
                    {pieData.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={entry.color} />
                    ))}
                  </Pie>
                  <RechartsTooltip />
                </PieChart>
              </ResponsiveContainer>
              <Typography variant="h3" component="div" sx={{ ml: 2 }}>
                {data.summary?.overall_health_score?.toFixed(0) || 0}%
              </Typography>
            </Box>
          </Paper>
        </Grid>

        {/* Detailed Source Table */}
        <Grid item xs={12}>
          <Paper sx={{ p: 2 }}>
            <Typography variant="h6" gutterBottom>
              Source Details
            </Typography>
            <TableContainer>
              <Table size="small">
                <TableHead>
                  <TableRow>
                    <TableCell>Source</TableCell>
                    <TableCell>Status</TableCell>
                    <TableCell>Uptime</TableCell>
                    <TableCell>Response Time</TableCell>
                    <TableCell>Error Rate</TableCell>
                    <TableCell>Quality Score</TableCell>
                    <TableCell>Monthly Cost</TableCell>
                    <TableCell>Last Check</TableCell>
                  </TableRow>
                </TableHead>
                <TableBody>
                  {Object.entries(data.sources || {}).map(([source, sourceData]: [string, any]) => (
                    <TableRow key={source}>
                      <TableCell>
                        <Typography variant="body2" fontWeight="medium">
                          {source}
                        </Typography>
                      </TableCell>
                      <TableCell>
                        <Box display="flex" alignItems="center" gap={1}>
                          {getHealthIcon(sourceData.status)}
                          <Chip
                            label={sourceData.status}
                            size="small"
                            color={
                              sourceData.status === 'healthy' ? 'success' :
                              sourceData.status === 'degraded' ? 'warning' : 'error'
                            }
                          />
                        </Box>
                      </TableCell>
                      <TableCell>{formatUptime(sourceData.uptime || 0)}</TableCell>
                      <TableCell>{sourceData.avg_response_time?.toFixed(0) || 0}ms</TableCell>
                      <TableCell>
                        <Typography
                          variant="body2"
                          color={sourceData.error_rate > 0.1 ? 'error' : 'text.primary'}
                        >
                          {(sourceData.error_rate * 100).toFixed(1)}%
                        </Typography>
                      </TableCell>
                      <TableCell>
                        <Box display="flex" alignItems="center" gap={1}>
                          <LinearProgress
                            variant="determinate"
                            value={sourceData.quality_score * 100}
                            sx={{
                              width: 60,
                              height: 6,
                              borderRadius: 3,
                              backgroundColor: 'grey.300',
                              '& .MuiLinearProgress-bar': {
                                backgroundColor: getHealthColor(sourceData.status),
                                borderRadius: 3
                              }
                            }}
                          />
                          <Typography variant="body2">
                            {(sourceData.quality_score * 100).toFixed(0)}%
                          </Typography>
                        </Box>
                      </TableCell>
                      <TableCell>{formatCost(sourceData.monthly_cost || 0)}</TableCell>
                      <TableCell>
                        <Tooltip title={new Date(sourceData.last_check).toLocaleString()}>
                          <Typography variant="body2" color="text.secondary">
                            {new Date(sourceData.last_check).toLocaleTimeString()}
                          </Typography>
                        </Tooltip>
                      </TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </TableContainer>
          </Paper>
        </Grid>

        {/* API Usage Summary */}
        {data.api_usage && (
          <Grid item xs={12}>
            <Paper sx={{ p: 2 }}>
              <Typography variant="h6" gutterBottom>
                API Usage Summary
              </Typography>
              <Grid container spacing={2}>
                <Grid item xs={12} sm={6} md={3}>
                  <Box>
                    <Typography variant="body2" color="text.secondary">
                      Total API Calls (24h)
                    </Typography>
                    <Typography variant="h6">
                      {data.api_usage.total_calls_24h || 0}
                    </Typography>
                  </Box>
                </Grid>
                <Grid item xs={12} sm={6} md={3}>
                  <Box>
                    <Typography variant="body2" color="text.secondary">
                      Total Cost (Month)
                    </Typography>
                    <Typography variant="h6">
                      {formatCost(data.api_usage.total_monthly_cost || 0)}
                    </Typography>
                  </Box>
                </Grid>
                <Grid item xs={12} sm={6} md={3}>
                  <Box>
                    <Typography variant="body2" color="text.secondary">
                      Average Response Time
                    </Typography>
                    <Typography variant="h6">
                      {data.api_usage.avg_response_time?.toFixed(0) || 0}ms
                    </Typography>
                  </Box>
                </Grid>
                <Grid item xs={12} sm={6} md={3}>
                  <Box>
                    <Typography variant="body2" color="text.secondary">
                      Budget Utilization
                    </Typography>
                    <Typography variant="h6">
                      {data.api_usage.budget_utilization?.toFixed(1) || 0}%
                    </Typography>
                    <LinearProgress
                      variant="determinate"
                      value={data.api_usage.budget_utilization || 0}
                      sx={{
                        mt: 1,
                        height: 6,
                        borderRadius: 3,
                        backgroundColor: 'grey.300',
                        '& .MuiLinearProgress-bar': {
                          backgroundColor:
                            (data.api_usage.budget_utilization || 0) > 80 ? '#f44336' :
                            (data.api_usage.budget_utilization || 0) > 60 ? '#ff9800' : '#4caf50',
                          borderRadius: 3
                        }
                      }}
                    />
                  </Box>
                </Grid>
              </Grid>
            </Paper>
          </Grid>
        )}
      </Grid>
    </Box>
  );
};

export default SourceHealthStatus;
