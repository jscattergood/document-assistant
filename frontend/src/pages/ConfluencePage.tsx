import React, { useState, useEffect } from 'react';
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
  Tabs,
  Tab,
  List,
  ListItem,
  ListItemText,
  ListItemSecondaryAction,
  IconButton,
  Chip,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Divider,
} from '@mui/material';
import {
  Cloud,
  Settings,
  Save,
  Sync,
  CheckCircle,
  Error as ErrorIcon,
  Add,
  Delete,
  PlayArrow,
  Link,
  Schedule,
  ContentCopy,
  Clear,
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

interface SyncPage {
  id: string;
  web_url: string;
  page_id: string;
  space_key: string;
  title: string;
  api_url: string;
  last_synced: string | null;
  sync_enabled: boolean;
  created_at: string;
}

interface TabPanelProps {
  children?: React.ReactNode;
  index: number;
  value: number;
}

const TabPanel: React.FC<TabPanelProps> = ({ children, value, index, ...other }) => (
  <div
    role="tabpanel"
    hidden={value !== index}
    id={`tabpanel-${index}`}
    aria-labelledby={`tab-${index}`}
    {...other}
  >
    {value === index && <Box sx={{ p: 4 }}>{children}</Box>}
  </div>
);

const ConfluencePage: React.FC = () => {
  const [tabValue, setTabValue] = useState(0);
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
  const [configSaved, setConfigSaved] = useState(false);
  const [configLoaded, setConfigLoaded] = useState(false);
  
  // Sync pages state
  const [syncPages, setSyncPages] = useState<SyncPage[]>([]);
  const [loadingSyncPages, setLoadingSyncPages] = useState(false);
  const [syncing, setSyncing] = useState(false);
  const [addUrlsDialog, setAddUrlsDialog] = useState(false);
  const [urlsToAdd, setUrlsToAdd] = useState('');

  useEffect(() => {
    if (connectionStatus.status === 'connected') {
      loadSyncPages();
    }
  }, [connectionStatus.status]);

  useEffect(() => {
    // Load saved configuration on component mount
    loadSavedConfig();
  }, []);

  const loadSavedConfig = async () => {
    try {
      // Load ALL configuration from localStorage (client-side only for security)
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
        
        // If we have complete credentials, enable tabs even without testing
        const hasValidCredentials = parsedConfig.url && parsedConfig.token;
        if (hasValidCredentials) {
          setConnectionStatus({
            status: 'connected', // Assume valid since we have saved credentials
            message: 'Using saved credentials (test connection to verify)',
          });
        }        
      }
    } catch (error) {
      console.error('Error loading saved config from localStorage:', error);
    }
  };

  const loadSyncPages = async () => {
    setLoadingSyncPages(true);
    try {
      const result = await confluenceAPI.listSyncPages();
      if (result.success) {
        setSyncPages(result.pages);
      }
    } catch (error) {
      console.error('Error loading sync pages:', error);
    } finally {
      setLoadingSyncPages(false);
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
    // Reset connection status when config changes
    if (connectionStatus.status !== 'not_tested') {
      setConnectionStatus({
        status: 'not_tested',
        message: 'Configuration changed - please test connection again',
      });
    }
    // Reset saved status when config changes
    setConfigSaved(false);
  };

  const handleSaveConfiguration = async () => {
    if (connectionStatus.status !== 'connected') {
      toast.error('Please test the connection successfully before saving');
      return;
    }

    setSaving(true);
    try {
      // Save ALL configuration to localStorage (client-side only)
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
      
      toast.success('Configuration saved successfully (client-side)');
      setConfigSaved(true);
    } catch (error) {
      toast.error('Failed to save configuration');
      console.error('Save error:', error);
    } finally {
      setSaving(false);
    }
  };

  const handleClearConfiguration = async () => {
    try {
      // Clear localStorage (client-side only)
      localStorage.removeItem('confluence_config');
      
      // Reset form
      setConfig({
        url: '',
        username: '',
        token: '',
        space_key: '',
        auto_sync: false,
        auth_type: 'pat',
      });
      
      // Reset status
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

  const handleAddUrls = async () => {
    if (!urlsToAdd.trim()) return;
    
    const urls = urlsToAdd.split('\n').filter(url => url.trim());
    if (urls.length === 0) return;

    try {
      const credentials = {
        url: config.url,
        username: config.username,
        api_token: config.token,
        auth_type: config.auth_type,
      };
      
      const result = await confluenceAPI.addPagesToSync(credentials, urls);
      if (result.success) {
        toast.success(result.message);
        setUrlsToAdd('');
        setAddUrlsDialog(false);
        loadSyncPages();
      } else {
        toast.error(result.message);
        if (result.errors && result.errors.length > 0) {
          result.errors.forEach(error => toast.error(error));
        }
      }
    } catch (error) {
      toast.error('Failed to add URLs to sync');
    }
  };

  const handleRunSync = async (pageIds?: string[]) => {
    setSyncing(true);
    try {
      const credentials = {
        url: config.url,
        username: config.username,
        api_token: config.token,
        auth_type: config.auth_type,
      };
      
      const result = await confluenceAPI.runSync(credentials, pageIds);
      if (result.success) {
        toast.success(result.message);
        loadSyncPages();
      } else {
        toast.error(result.message);
        if (result.errors && result.errors.length > 0) {
          result.errors.forEach(error => toast.error(error));
        }
      }
    } catch (error) {
      toast.error('Failed to run sync');
    } finally {
      setSyncing(false);
    }
  };

  const handleRemoveFromSync = async (pageId: string) => {
    try {
      const result = await confluenceAPI.removeFromSync(pageId);
      if (result.success) {
        toast.success(result.message);
        loadSyncPages();
      }
    } catch (error) {
      toast.error('Failed to remove page from sync');
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

  const formatDate = (dateString: string) => {
    return new Date(dateString).toLocaleString();
  };

  return (
    <Container maxWidth="xl" sx={{ py: 4 }}>
      <Box sx={{ mb: 6 }}>
        <Typography variant="h1" component="h1" sx={{ mb: 2 }}>
          Confluence Integration
        </Typography>
        <Typography variant="h6" color="text.secondary">
          Connect, sync, and work with your Confluence workspace
        </Typography>
      </Box>

      <Paper sx={{ mb: 4 }}>
        <Tabs
          value={tabValue}
          onChange={(_, newValue) => setTabValue(newValue)}
          sx={{ borderBottom: 1, borderColor: 'divider' }}
        >
          <Tab label="Connection" icon={<Settings />} />
          <Tab label="Page Sync" icon={<Sync />} disabled={connectionStatus.status !== 'connected'} />
        </Tabs>

        <TabPanel value={tabValue} index={0}>
          <Grid container spacing={4}>
            {/* Configuration */}
            <Grid size={{ xs: 12, lg: 8 }}>
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

              {configSaved && (
                <Alert severity="info" sx={{ mb: 4 }}>
                  Configuration saved successfully! Your Confluence settings have been stored.
                </Alert>
              )}

              {configLoaded && (
                <Alert severity="info" sx={{ mb: 4 }}>
                  Configuration loaded from browser storage. All Confluence settings are stored locally for security.
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

                <Box sx={{ display: 'flex', gap: 2, flexWrap: 'wrap' }}>
                  <Button 
                    variant="contained" 
                    startIcon={saving ? <CircularProgress size={20} /> : (configSaved ? <CheckCircle /> : <Save />)} 
                    size="large"
                    onClick={handleSaveConfiguration}
                    disabled={testing || saving || connectionStatus.status !== 'connected'}
                    color={configSaved ? 'success' : 'primary'}
                  >
                    {saving ? 'Saving...' : (configSaved ? 'Saved ✓' : 'Save Configuration')}
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
                  <Button 
                    variant="outlined" 
                    startIcon={<Clear />} 
                    size="large"
                    onClick={handleClearConfiguration}
                    disabled={testing || saving}
                    color="error"
                  >
                    Clear Configuration
                  </Button>
                </Box>
              </Box>
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
                      onClick={() => setTabValue(1)}
                    >
                      Manage Page Sync
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
        </TabPanel>

        <TabPanel value={tabValue} index={1}>
          <Box sx={{ mb: 4 }}>
            <Typography variant="h4" component="h2" sx={{ mb: 2 }}>
              Page Synchronization
            </Typography>
            <Typography color="text.secondary" sx={{ mb: 4 }}>
              Manage pages that are automatically synced to your document index. Paste Confluence page URLs to add them to the sync list.
            </Typography>

            <Box sx={{ display: 'flex', gap: 2, mb: 4 }}>
              <Button
                variant="contained"
                startIcon={<Add />}
                onClick={() => setAddUrlsDialog(true)}
              >
                Add Pages
              </Button>
              <Button
                variant="outlined"
                startIcon={syncing ? <CircularProgress size={20} /> : <Sync />}
                onClick={() => handleRunSync()}
                disabled={syncing || syncPages.length === 0}
              >
                {syncing ? 'Syncing...' : 'Sync All'}
              </Button>
              <Button
                variant="outlined"
                startIcon={<ContentCopy />}
                onClick={loadSyncPages}
                disabled={loadingSyncPages}
              >
                Refresh
              </Button>
            </Box>

            {loadingSyncPages ? (
              <Box sx={{ display: 'flex', justifyContent: 'center', py: 4 }}>
                <CircularProgress />
              </Box>
            ) : syncPages.length === 0 ? (
              <Alert severity="info">
                No pages added to sync yet. Click "Add Pages" to get started.
              </Alert>
            ) : (
              <Paper variant="outlined">
                <List>
                  {syncPages.map((page, index) => (
                    <React.Fragment key={page.id}>
                      <ListItem>
                        <ListItemText
                          primary={page.title}
                          secondary={
                            <Box>
                              <Typography variant="body2" color="text.secondary">
                                Space: {page.space_key} • Page ID: {page.page_id}
                              </Typography>
                              <Typography variant="body2" color="text.secondary">
                                {page.last_synced ? `Last synced: ${formatDate(page.last_synced)}` : 'Never synced'}
                              </Typography>
                              <Box sx={{ mt: 1 }}>
                                <Chip
                                  size="small"
                                  label={page.sync_enabled ? 'Enabled' : 'Disabled'}
                                  color={page.sync_enabled ? 'success' : 'default'}
                                  sx={{ mr: 1 }}
                                />
                                <Chip
                                  size="small"
                                  label="Confluence"
                                  variant="outlined"
                                />
                              </Box>
                            </Box>
                          }
                        />
                        <ListItemSecondaryAction>
                          <Box sx={{ display: 'flex', gap: 1 }}>
                            <IconButton
                              size="small"
                              onClick={() => handleRunSync([page.id])}
                              disabled={syncing}
                              title="Sync this page"
                            >
                              <PlayArrow />
                            </IconButton>
                            <IconButton
                              size="small"
                              onClick={() => window.open(page.web_url, '_blank')}
                              title="Open in Confluence"
                            >
                              <Link />
                            </IconButton>
                            <IconButton
                              size="small"
                              onClick={() => handleRemoveFromSync(page.id)}
                              color="error"
                              title="Remove from sync"
                            >
                              <Delete />
                            </IconButton>
                          </Box>
                        </ListItemSecondaryAction>
                      </ListItem>
                      {index < syncPages.length - 1 && <Divider />}
                    </React.Fragment>
                  ))}
                </List>
              </Paper>
            )}
          </Box>
        </TabPanel>
      </Paper>

      {/* Add URLs Dialog */}
      <Dialog open={addUrlsDialog} onClose={() => setAddUrlsDialog(false)} maxWidth="md" fullWidth>
        <DialogTitle>Add Pages to Sync</DialogTitle>
        <DialogContent>
          <Alert severity="info" sx={{ mb: 3 }}>
            Paste Confluence page URLs, one per line. Supported formats include:
            <br />• https://wiki.domain.com/spaces/viewspace.action?key=SPACE&pageId=123456
            <br />• https://wiki.domain.com/pages/viewpage.action?spaceKey=SPACE&title=Page+Title
            <br />• https://wiki.domain.com/display/SPACE/Page+Title
            <br />• https://wiki.domain.com/spaces/SPACE/pages/123456/Page+Title
          </Alert>
          <TextField
            fullWidth
            multiline
            rows={8}
            label="Confluence Page URLs"
            placeholder="https://wiki.autodesk.com/spaces/viewspace.action?key=~scattej&pageId=123456"
            value={urlsToAdd}
            onChange={(e) => setUrlsToAdd(e.target.value)}
            helperText="Paste one URL per line"
          />
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setAddUrlsDialog(false)}>Cancel</Button>
          <Button onClick={handleAddUrls} variant="contained" disabled={!urlsToAdd.trim()}>
            Add Pages
          </Button>
        </DialogActions>
      </Dialog>
    </Container>
  );
};

export default ConfluencePage; 