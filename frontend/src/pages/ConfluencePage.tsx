import React, { useState } from 'react';
import {
  Container,
  Typography,
  Paper,
  Box,
  TextField,
  Button,
  Grid,
  Card,
  CardContent,
  Switch,
  FormControlLabel,
  Alert,
  CircularProgress,
  MenuItem,
} from '@mui/material';
import {
  Cloud,
  Settings,
  Save,
  Sync,
  CheckCircle,
  Error as ErrorIcon,
} from '@mui/icons-material';
import { confluenceAPI } from '../services/api';
import toast from 'react-hot-toast';

interface ConfluenceConfig {
  url: string;
  username: string;
  token: string;
  space_key: string;
  auto_sync: boolean;
  auth_type: 'pat' | 'basic';
}

const ConfluencePage: React.FC = () => {
  const [config, setConfig] = useState<ConfluenceConfig>({
    url: '',
    username: '',
    token: '',
    space_key: '',
    auto_sync: false,
    auth_type: 'pat',
  });

  const [connectionStatus, setConnectionStatus] = useState<{
    status: 'not_tested' | 'testing' | 'connected' | 'error';
    message: string;
    user?: string;
    space?: string;
  }>({
    status: 'not_tested',
    message: 'Not tested yet',
  });

  const [saving, setSaving] = useState(false);
  const [testing, setTesting] = useState(false);

  const handleInputChange = (field: keyof ConfluenceConfig) => (
    event: React.ChangeEvent<HTMLInputElement>
  ) => {
    if (field === 'auto_sync') {
      setConfig(prev => ({ ...prev, [field]: event.target.checked }));
    } else {
      setConfig(prev => ({ ...prev, [field]: event.target.value }));
    }
    // Reset connection status when config changes
    if (connectionStatus.status !== 'not_tested') {
      setConnectionStatus({
        status: 'not_tested',
        message: 'Configuration changed - please test connection again',
      });
    }
  };

  const handleTestConnection = async () => {
    if (!config.url || !config.token) {
      toast.error('Please fill in URL and API token');
      return;
    }

    if (config.auth_type === 'basic' && !config.username) {
      toast.error('Username is required for Basic authentication');
      return;
    }

    setTesting(true);
    setConnectionStatus({
      status: 'testing',
      message: 'Testing connection...',
    });

    try {
      const result = await confluenceAPI.testConnection({
        url: config.url,
        token: config.token,
        auth_type: config.auth_type,
        ...(config.username && { username: config.username }),
        ...(config.space_key && { space_key: config.space_key }),
      });

      if (result.success) {
        setConnectionStatus({
          status: 'connected',
          message: result.message,
          user: result.user,
          space: result.space,
        });
        toast.success('Connection successful!');
      } else {
        setConnectionStatus({
          status: 'error',
          message: result.message,
        });
        toast.error(result.message);
      }
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Unknown error occurred';
      setConnectionStatus({
        status: 'error',
        message: `Connection failed: ${errorMessage}`,
      });
      toast.error('Failed to test connection');
    } finally {
      setTesting(false);
    }
  };

  const handleSaveConfiguration = async () => {
    if (connectionStatus.status !== 'connected') {
      toast.error('Please test the connection successfully before saving');
      return;
    }

    setSaving(true);
    try {
      // For now, just simulate saving (you can implement actual API call later)
      await new Promise(resolve => setTimeout(resolve, 1000));
      toast.success('Configuration saved successfully!');
    } catch (error) {
      toast.error('Failed to save configuration');
    } finally {
      setSaving(false);
    }
  };

  const getStatusIcon = () => {
    switch (connectionStatus.status) {
      case 'connected':
        return <CheckCircle sx={{ fontSize: 48, color: 'success.main', mb: 2 }} />;
      case 'error':
        return <ErrorIcon sx={{ fontSize: 48, color: 'error.main', mb: 2 }} />;
      case 'testing':
        return <CircularProgress size={48} sx={{ mb: 2 }} />;
      default:
        return <Cloud sx={{ fontSize: 48, color: 'text.secondary', mb: 2 }} />;
    }
  };

  const getStatusColor = () => {
    switch (connectionStatus.status) {
      case 'connected':
        return 'success.main';
      case 'error':
        return 'error.main';
      case 'testing':
        return 'warning.main';
      default:
        return 'text.secondary';
    }
  };

  const getStatusText = () => {
    switch (connectionStatus.status) {
      case 'connected':
        return 'Connected';
      case 'error':
        return 'Connection Failed';
      case 'testing':
        return 'Testing...';
      default:
        return 'Not Connected';
    }
  };

  return (
    <Container maxWidth="xl" sx={{ py: 4 }}>
      <Box sx={{ mb: 6 }}>
        <Typography variant="h1" component="h1" sx={{ mb: 2 }}>
          Confluence Integration
        </Typography>
        <Typography variant="h6" color="text.secondary">
          Connect and sync with your Confluence workspace
        </Typography>
      </Box>

      <Grid container spacing={4}>
        {/* Configuration */}
        <Grid size={{ xs: 12, lg: 8 }}>
          <Paper elevation={2} sx={{ p: 4 }}>
            <Typography variant="h4" component="h2" sx={{ mb: 4 }}>
              Connection Settings
            </Typography>
            
            <Alert severity="info" sx={{ mb: 4 }}>
              Configure your Confluence settings to enable automatic synchronization and content generation.
              Choose "Personal Access Token" for modern Confluence instances (recommended) or "Basic Auth" for older setups.
            </Alert>

            {connectionStatus.status === 'error' && (
              <Alert severity="error" sx={{ mb: 4 }}>
                {connectionStatus.message}
              </Alert>
            )}

            {connectionStatus.status === 'connected' && (
              <Alert severity="success" sx={{ mb: 4 }}>
                {connectionStatus.message}
                {connectionStatus.user && (
                  <><br />User: {connectionStatus.user}</>
                )}
                {connectionStatus.space && (
                  <><br />Space: {connectionStatus.space}</>
                )}
              </Alert>
            )}

            <Box component="form" sx={{ '& .MuiTextField-root': { mb: 3 } }}>
              <TextField
                fullWidth
                select
                label="Authentication Type"
                value={config.auth_type}
                onChange={(e) => setConfig(prev => ({ ...prev, auth_type: e.target.value as 'pat' | 'basic' }))}
                disabled={testing || saving}
                sx={{ mb: 3 }}
                helperText="PAT (Personal Access Token) is recommended for modern Confluence instances"
              >
                <MenuItem value="pat">Personal Access Token (PAT)</MenuItem>
                <MenuItem value="basic">Basic Authentication</MenuItem>
              </TextField>

              <TextField
                fullWidth
                label="Confluence URL"
                placeholder="https://your-domain.atlassian.net or https://wiki.yourcompany.com"
                variant="outlined"
                value={config.url}
                onChange={handleInputChange('url')}
                disabled={testing || saving}
              />
              
              {config.auth_type === 'basic' && (
                <TextField
                  fullWidth
                  label="Username/Email"
                  placeholder="your-email@domain.com"
                  variant="outlined"
                  value={config.username}
                  onChange={handleInputChange('username')}
                  disabled={testing || saving}
                />
              )}
              
              <TextField
                fullWidth
                label={config.auth_type === 'pat' ? 'Personal Access Token' : 'API Token/Password'}
                type="password"
                placeholder={config.auth_type === 'pat' ? 'Your Personal Access Token' : 'Your API Token or Password'}
                variant="outlined"
                value={config.token}
                onChange={handleInputChange('token')}
                disabled={testing || saving}
                helperText={
                  config.auth_type === 'pat' 
                    ? "Generate a PAT from your Confluence account settings under Security > API tokens"
                    : "Use an API token or password for Basic authentication"
                }
              />
              
              <TextField
                fullWidth
                label="Space Key (Optional)"
                placeholder="DOCS"
                variant="outlined"
                value={config.space_key}
                onChange={handleInputChange('space_key')}
                disabled={testing || saving}
                helperText="The space key where documents will be created (leave empty to test without space access)"
              />

              <FormControlLabel
                control={
                  <Switch 
                    checked={config.auto_sync}
                    onChange={handleInputChange('auto_sync')}
                    disabled={testing || saving}
                  />
                }
                label="Enable automatic synchronization"
                sx={{ mb: 3 }}
              />

              <Box sx={{ display: 'flex', gap: 2 }}>
                <Button 
                  variant="contained" 
                  startIcon={saving ? <CircularProgress size={20} /> : <Save />} 
                  size="large"
                  onClick={handleSaveConfiguration}
                  disabled={testing || saving || connectionStatus.status !== 'connected'}
                >
                  {saving ? 'Saving...' : 'Save Configuration'}
                </Button>
                <Button 
                  variant="outlined" 
                  startIcon={testing ? <CircularProgress size={20} /> : <Sync />} 
                  size="large"
                  onClick={handleTestConnection}
                  disabled={testing || saving}
                >
                  {testing ? 'Testing...' : 'Test Connection'}
                </Button>
              </Box>
            </Box>
          </Paper>
        </Grid>

        {/* Status */}
        <Grid size={{ xs: 12, lg: 4 }}>
          <Card sx={{ mb: 3 }}>
            <CardContent sx={{ textAlign: 'center' }}>
              {getStatusIcon()}
              <Typography variant="h6" sx={{ mb: 1, color: getStatusColor() }}>
                {getStatusText()}
              </Typography>
              <Typography variant="body2" color="text.secondary">
                {connectionStatus.message}
              </Typography>
            </CardContent>
          </Card>

          <Card>
            <CardContent>
              <Typography variant="h6" sx={{ mb: 2 }}>
                Quick Actions
              </Typography>
              <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
                <Button 
                  variant="outlined" 
                  fullWidth 
                  disabled={connectionStatus.status !== 'connected'}
                >
                  Sync Documents
                </Button>
                <Button 
                  variant="outlined" 
                  fullWidth 
                  disabled={connectionStatus.status !== 'connected'}
                >
                  Create Page
                </Button>
                <Button 
                  variant="outlined" 
                  fullWidth 
                  disabled={connectionStatus.status !== 'connected'}
                >
                  View Spaces
                </Button>
              </Box>
            </CardContent>
          </Card>
        </Grid>
      </Grid>
    </Container>
  );
};

export default ConfluencePage; 