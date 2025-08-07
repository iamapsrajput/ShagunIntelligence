import React, { useState, useEffect } from 'react';
import {
  Grid,
  Paper,
  Typography,
  Box,
  Tab,
  Tabs,
  CircularProgress,
  Alert
} from '@mui/material';
import { styled } from '@mui/material/styles';
import DataQualityOverview from '../components/monitoring/DataQualityOverview';
import SourceHealthStatus from '../components/monitoring/SourceHealthStatus';
import SentimentFusionChart from '../components/monitoring/SentimentFusionChart';
import FailoverLogs from '../components/monitoring/FailoverLogs';
import APICostMonitor from '../components/monitoring/APICostMonitor';
import QualityAlertsPanel from '../components/monitoring/QualityAlertsPanel';
import RealTimeQualityChart from '../components/monitoring/RealTimeQualityChart';
import { useWebSocket } from '../hooks/useWebSocket';
import { api } from '../services/api';

const StyledPaper = styled(Paper)(({ theme }) => ({
  padding: theme.spacing(3),
  height: '100%',
  position: 'relative',
  backgroundColor: theme.palette.background.paper,
  borderRadius: theme.shape.borderRadius,
  boxShadow: theme.shadows[2],
}));

interface TabPanelProps {
  children?: React.ReactNode;
  index: number;
  value: number;
}

function TabPanel(props: TabPanelProps) {
  const { children, value, index, ...other } = props;
  return (
    <div
      role="tabpanel"
      hidden={value !== index}
      id={`monitoring-tabpanel-${index}`}
      aria-labelledby={`monitoring-tab-${index}`}
      {...other}
    >
      {value === index && <Box sx={{ py: 3 }}>{children}</Box>}
    </div>
  );
}

const MonitoringDashboard: React.FC = () => {
  const [tabValue, setTabValue] = useState(0);
  const [selectedSymbol, setSelectedSymbol] = useState<string>('RELIANCE.NS');
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [qualityData, setQualityData] = useState<any>(null);
  const [sourceHealth, setSourceHealth] = useState<any>(null);
  const [sentimentData, setSentimentData] = useState<any>(null);

  // WebSocket connection for real-time quality updates
  const {
    data: wsData,
    isConnected,
    subscribe,
    unsubscribe
  } = useWebSocket('/api/v1/data-quality/ws/quality-monitor');

  // Load initial data
  useEffect(() => {
    loadDashboardData();
  }, [selectedSymbol]);

  // Handle WebSocket data
  useEffect(() => {
    if (wsData && wsData.type === 'quality_update' && wsData.symbol === selectedSymbol) {
      setQualityData(wsData.data);
    }
  }, [wsData, selectedSymbol]);

  // Subscribe to quality updates for selected symbol
  useEffect(() => {
    if (isConnected && selectedSymbol) {
      subscribe({ type: 'subscribe', symbol: selectedSymbol });
      return () => {
        unsubscribe({ type: 'unsubscribe', symbol: selectedSymbol });
      };
    }
  }, [isConnected, selectedSymbol, subscribe, unsubscribe]);

  const loadDashboardData = async () => {
    try {
      setLoading(true);
      setError(null);

      // Load all dashboard data in parallel
      const [quality, health, sentiment] = await Promise.all([
        api.get(`/data-quality/metrics/${selectedSymbol}`),
        api.get('/data-quality/source-health-multi'),
        api.get(`/data-quality/sentiment-fusion/${selectedSymbol}`)
      ]);

      setQualityData(quality.data);
      setSourceHealth(health.data);
      setSentimentData(sentiment.data);
    } catch (err) {
      console.error('Error loading dashboard data:', err);
      setError('Failed to load dashboard data');
    } finally {
      setLoading(false);
    }
  };

  const handleTabChange = (event: React.SyntheticEvent, newValue: number) => {
    setTabValue(newValue);
  };

  const handleSymbolChange = (newSymbol: string) => {
    setSelectedSymbol(newSymbol);
  };

  if (loading) {
    return (
      <Box display="flex" justifyContent="center" alignItems="center" minHeight="80vh">
        <CircularProgress />
      </Box>
    );
  }

  if (error) {
    return (
      <Box p={3}>
        <Alert severity="error">{error}</Alert>
      </Box>
    );
  }

  return (
    <Box sx={{ flexGrow: 1, p: 3 }}>
      <Typography variant="h4" gutterBottom>
        Multi-Source Monitoring Dashboard
      </Typography>

      <Paper sx={{ mb: 3 }}>
        <Tabs
          value={tabValue}
          onChange={handleTabChange}
          indicatorColor="primary"
          textColor="primary"
          variant="scrollable"
          scrollButtons="auto"
        >
          <Tab label="Quality Overview" />
          <Tab label="Source Health" />
          <Tab label="Sentiment Analysis" />
          <Tab label="API Costs" />
          <Tab label="Alerts & Logs" />
        </Tabs>
      </Paper>

      <TabPanel value={tabValue} index={0}>
        <Grid container spacing={3}>
          <Grid item xs={12}>
            <StyledPaper>
              <DataQualityOverview
                symbol={selectedSymbol}
                data={qualityData}
                onSymbolChange={handleSymbolChange}
              />
            </StyledPaper>
          </Grid>
          <Grid item xs={12} lg={8}>
            <StyledPaper>
              <RealTimeQualityChart
                symbol={selectedSymbol}
                data={qualityData}
              />
            </StyledPaper>
          </Grid>
          <Grid item xs={12} lg={4}>
            <StyledPaper>
              <QualityAlertsPanel />
            </StyledPaper>
          </Grid>
        </Grid>
      </TabPanel>

      <TabPanel value={tabValue} index={1}>
        <Grid container spacing={3}>
          <Grid item xs={12}>
            <StyledPaper>
              <SourceHealthStatus data={sourceHealth} />
            </StyledPaper>
          </Grid>
          <Grid item xs={12}>
            <StyledPaper>
              <FailoverLogs />
            </StyledPaper>
          </Grid>
        </Grid>
      </TabPanel>

      <TabPanel value={tabValue} index={2}>
        <Grid container spacing={3}>
          <Grid item xs={12}>
            <StyledPaper>
              <SentimentFusionChart
                symbol={selectedSymbol}
                data={sentimentData}
                onSymbolChange={handleSymbolChange}
              />
            </StyledPaper>
          </Grid>
        </Grid>
      </TabPanel>

      <TabPanel value={tabValue} index={3}>
        <Grid container spacing={3}>
          <Grid item xs={12}>
            <StyledPaper>
              <APICostMonitor />
            </StyledPaper>
          </Grid>
        </Grid>
      </TabPanel>

      <TabPanel value={tabValue} index={4}>
        <Grid container spacing={3}>
          <Grid item xs={12} md={6}>
            <StyledPaper>
              <QualityAlertsPanel showFull />
            </StyledPaper>
          </Grid>
          <Grid item xs={12} md={6}>
            <StyledPaper>
              <FailoverLogs showFull />
            </StyledPaper>
          </Grid>
        </Grid>
      </TabPanel>
    </Box>
  );
};

export default MonitoringDashboard;
