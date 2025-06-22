import React, { useState, useEffect } from 'react';
import {
  Box,
  TextField,
  Button,
  Switch,
  FormControlLabel,
  Alert,
  MenuItem,
  CircularProgress,
  Typography,
  Card,
  CardContent,
  FormControl,
  FormLabel,
  Radio,
  RadioGroup,
} from '@mui/material';
import { Settings, CheckCircle, Error as ErrorIcon, Save, Check } from '@mui/icons-material';
import { confluenceAPI } from '../../services/api';
import toast from 'react-hot-toast';

export interface ConfluenceConfig {
  url: string;
  username: string;
  token: string;
  space_key: string;
  auto_sync: boolean;
  auth_type: 'pat' | 'basic';
}

interface ConnectionStatus {
  status: 'not_tested' | 'testing' | 'connected' | 'error';
  message: string;
  user?: string;
  space?: string;
}

interface ConfluenceSettingsProps {
  onConfigured: () => void;
}

export const ConfluenceSettings: React.FC<ConfluenceSettingsProps> = ({ onConfigured }) => {
  const [config, setConfig] = useState<ConfluenceConfig>({
    url: '',
    username: '',
    token: '',
    space_key: '',
    auto_sync: false,
    auth_type: 'pat',
  });

  const [connectionStatus, setConnectionStatus] = useState<ConnectionStatus>({
    status: 'not_tested',
    message: 'Not tested yet',
  });

  const [saving, setSaving] = useState(false);
  const [testing, setTesting] = useState(false);
  const [configSaved, setConfigSaved] = useState(false);
  const [configLoaded, setConfigLoaded] = useState(false);
  const [testResult, setTestResult] = useState<{
    success: boolean;
    message: string;
  } | null>(null);

  useEffect(() => {
    loadSavedConfig();
  }, []);

  const loadSavedConfig = async () => {
    try {
      const savedConfig = localStorage.getItem('confluence_config');
      if (savedConfig) {
        const parsedConfig = JSON.parse(savedConfig);
        setConfig(prev => ({
          ...prev,
          url: parsedConfig.url || '',
          username: parsedConfig.username || '',
          token: parsedConfig.token || '',
          space_key: parsedConfig.space_key || '',
          auth_type: parsedConfig.auth_type || 'pat',
          auto_sync: parsedConfig.auto_sync || false,
        }));
        
        setConfigLoaded(true);
        
        const hasValidCredentials = parsedConfig.url && parsedConfig.token;
        if (hasValidCredentials) {
          setConnectionStatus({
            status: 'connected',
            message: 'Using saved credentials (test connection to verify)',
          });
          onConfigured();
        }
      }
    } catch (error) {
      console.error('Error loading saved config from localStorage:', error);
    }
  };

  const handleInputChange = (field: keyof ConfluenceConfig) => (
    event: React.ChangeEvent<HTMLInputElement>
  ) => {
    if (field === 'auto_sync') {
      setConfig(prev => ({ ...prev, [field]: event.target.checked }));
    } else {
      setConfig(prev => ({ ...prev, [field]: event.target.value }));
    }
    if (connectionStatus.status !== 'not_tested') {
      setConnectionStatus({
        status: 'not_tested',
        message: 'Configuration changed - please test connection again',
      });
    }
    setConfigSaved(false);
    setTestResult(null);
  };

  const handleSaveConfiguration = async () => {
    if (connectionStatus.status !== 'connected') {
      toast.error('Please test the connection successfully before saving');
      return;
    }

    setSaving(true);
    try {
      const configToSave = {
        url: config.url,
        username: config.username,
        token: config.token,
        space_key: config.space_key,
        auth_type: config.auth_type,
        auto_sync: config.auto_sync,
        saved_at: new Date().toISOString()
      };
      
      localStorage.setItem('confluence_config', JSON.stringify(configToSave));
      
      toast.success('Configuration saved successfully');
      setConfigSaved(true);
      onConfigured();
    } catch (error) {
      toast.error('Failed to save configuration');
      console.error('Save error:', error);
    } finally {
      setSaving(false);
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
        setTestResult(result);
      } else {
        setConnectionStatus({
          status: 'error',
          message: result.message,
        });
        toast.error(result.message);
        setTestResult(result);
      }
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Unknown error occurred';
      setConnectionStatus({
        status: 'error',
        message: `Connection failed: ${errorMessage}`,
      });
      toast.error('Failed to test connection');
      setTestResult({
        success: false,
        message: `Connection failed: ${errorMessage}`,
      });
    } finally {
      setTesting(false);
    }
  };

  const handleClearConfiguration = async () => {
    try {
      localStorage.removeItem('confluence_config');
      
      setConfig({
        url: '',
        username: '',
        token: '',
        space_key: '',
        auto_sync: false,
        auth_type: 'pat',
      });
      
      setConnectionStatus({
        status: 'not_tested',
        message: 'Configuration cleared',
      });
      setConfigSaved(false);
      setConfigLoaded(false);
      
      toast.success('Configuration cleared successfully');
    } catch (error) {
      toast.error('Failed to clear configuration');
      console.error('Clear error:', error);
    }
  };

  const getStatusIcon = () => {
    switch (connectionStatus.status) {
      case 'connected':
        return <CheckCircle color="success" sx={{ fontSize: 48, mb: 1 }} />;
      case 'error':
        return <ErrorIcon color="error" sx={{ fontSize: 48, mb: 1 }} />;
      case 'testing':
        return <CircularProgress size={48} sx={{ mb: 1 }} />;
      default:
        return <Settings color="disabled" sx={{ fontSize: 48, mb: 1 }} />;
    }
  };

  const getStatusText = () => {
    switch (connectionStatus.status) {
      case 'connected':
        return 'Connected';
      case 'error':
        return 'Connection Error';
      case 'testing':
        return 'Testing Connection';
      default:
        return 'Not Connected';
    }
  };

  const getStatusColor = () => {
    switch (connectionStatus.status) {
      case 'connected':
        return 'success.main';
      case 'error':
        return 'error.main';
      case 'testing':
        return 'info.main';
      default:
        return 'text.secondary';
    }
  };

  return (
    <Box>
      <Typography variant="h6" gutterBottom>
        Confluence Integration
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

      <Box sx={{ display: 'flex', gap: 4 }}>
        <Box sx={{ flex: 1 }}>
          <Box component="form" sx={{ '& .MuiTextField-root': { mb: 3 } }}>
            <FormControl component="fieldset" sx={{ mb: 4 }}>
              <FormLabel component="legend">Authentication Type</FormLabel>
              <RadioGroup
                row
                value={config.auth_type}
                onChange={handleInputChange('auth_type')}
              >
                <FormControlLabel
                  value="pat"
                  control={<Radio />}
                  label="Personal Access Token"
                />
                <FormControlLabel
                  value="basic"
                  control={<Radio />}
                  label="Basic Auth"
                />
              </RadioGroup>
            </FormControl>

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

            <Box sx={{ display: 'flex', gap: 2, mt: 2 }}>
              <Button
                variant="contained"
                onClick={handleTestConnection}
                disabled={testing || !config.url || !config.token || (config.auth_type === 'basic' && !config.username)}
              >
                {testing ? 'Testing...' : 'Test Connection'}
              </Button>
              
              <Button
                variant="contained"
                color="primary"
                onClick={handleSaveConfiguration}
                disabled={saving || connectionStatus.status !== 'connected'}
              >
                {saving ? 'Saving...' : 'Save Configuration'}
              </Button>

              <Button
                variant="outlined"
                color="error"
                onClick={handleClearConfiguration}
                disabled={saving || testing}
              >
                Clear Configuration
              </Button>
            </Box>
          </Box>
        </Box>

        <Box sx={{ width: 300 }}>
          <Card>
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
        </Box>
      </Box>

      {testResult && (
        <Alert
          severity={testResult.success ? 'success' : 'error'}
          sx={{ mb: 3 }}
        >
          {testResult.message}
        </Alert>
      )}
    </Box>
  );
}; 