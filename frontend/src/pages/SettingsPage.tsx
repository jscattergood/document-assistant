import React, { useState, useEffect, useCallback } from 'react';
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
  Select,
  MenuItem,
  FormControl,
  InputLabel,
  Divider,
  Slider,
  Alert,
  CircularProgress,
  Chip,
  LinearProgress,
  List,
  ListItem,
  ListItemText,
  ListItemSecondaryAction,
  IconButton,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Tooltip,
  Tabs,
  Tab,
  Link as MuiLink,
} from '@mui/material';
import { Link as RouterLink } from 'react-router-dom';
import {
  Settings,
  Save,
  Refresh,
  Storage,
  Psychology,
  Memory,
  Speed,
  CheckCircle,
  Error as ErrorIcon,
  Download,
  CloudUpload,
  Delete,
  PlayArrow,
  Info,
  Warning,
  Star,
  Computer,
  Folder,
  Assessment,
  Launch,
  Stop,
  RestartAlt,
  PowerSettingsNew,
  Cloud,
} from '@mui/icons-material';
import { useDropzone } from 'react-dropzone';
import { modelsAPI } from '../services/api';
import toast from 'react-hot-toast';
import { ConfluenceSettings } from '../components/settings/ConfluenceSettings';
import { TemplateManagement } from '../components/settings/TemplateManagement';
import { SyncManagement } from '../components/settings/SyncManagement';

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
      id={`settings-tabpanel-${index}`}
      aria-labelledby={`settings-tab-${index}`}
      {...other}
    >
      {value === index && <Box sx={{ pt: 3 }}>{children}</Box>}
    </div>
  );
}

const SettingsPage: React.FC = () => {
  const [currentTab, setCurrentTab] = useState(0);
  const [availableModels, setAvailableModels] = useState<Record<string, any>>({});
  const [currentModel, setCurrentModel] = useState<any>(null);
  const [systemStatus, setSystemStatus] = useState<any>(null);
  const [loadingModels, setLoadingModels] = useState(false);
  const [changingModel, setChangingModel] = useState(false);
  const [testingModel, setTestingModel] = useState(false);
  const [selectedModel, setSelectedModel] = useState<string>('');
  const [clearingStore, setClearingStore] = useState(false);
  
  // Storage-related state
  const [storageConfig, setStorageConfig] = useState<any>(null);
  const [storageStatus, setStorageStatus] = useState<any>(null);
  const [loadingStorage, setLoadingStorage] = useState(false);
  const [clearingCache, setClearingCache] = useState(false);

  // GPT4All model state
  const [gpt4allModels, setGpt4allModels] = useState<any[]>([]);
  const [downloadedModels, setDownloadedModels] = useState<any[]>([]);
  const [loadingGpt4allModels, setLoadingGpt4allModels] = useState(false);
  const [downloadingModels, setDownloadingModels] = useState<Set<string>>(new Set());
  const [uploadingModel, setUploadingModel] = useState(false);
  const [deleteDialog, setDeleteDialog] = useState<{ open: boolean; model?: any }>({ open: false });
  const [downloadProgress, setDownloadProgress] = useState<Record<string, number>>({});

  // App Settings State
  const [useDocumentContext, setUseDocumentContext] = useState(true);
  const [enableNotifications, setEnableNotifications] = useState(true);
  const [maxTokens, setMaxTokens] = useState(512);
  const [temperature, setTemperature] = useState(0.7);
  const [maxConversationHistory, setMaxConversationHistory] = useState(10);
  const [enableConversationContext, setEnableConversationContext] = useState(true);
  const [loadingSettings, setLoadingSettings] = useState(false);
  const [savingSettings, setSavingSettings] = useState(false);

  // Provider State
  const [currentProvider, setCurrentProvider] = useState<string>('gpt4all');
  const [preferredGpt4allModel, setPreferredGpt4allModel] = useState<string | null>(null);
  const [preferredOllamaModel, setPreferredOllamaModel] = useState<string>('llama3.2:3b');
  const [ollamaModels, setOllamaModels] = useState<string[]>([]);
  const [ollamaRunning, setOllamaRunning] = useState(false);
  const [loadingProvider, setLoadingProvider] = useState(false);
  const [switchingProvider, setSwitchingProvider] = useState(false);

  // Ollama Process Management State
  const [ollamaStatus, setOllamaStatus] = useState<any>(null);
  const [loadingOllamaStatus, setLoadingOllamaStatus] = useState(false);
  const [controllingOllama, setControllingOllama] = useState(false);
  const [autoStartOllama, setAutoStartOllama] = useState(false);

  // Loading states
  const [loadingEmbedding, setLoadingEmbedding] = useState(false);

  const [confluenceTab, setConfluenceTab] = useState(0);
  const [isConfigured, setIsConfigured] = useState(() => {
    try {
      const savedConfig = localStorage.getItem('confluence_config');
      if (savedConfig) {
        const parsedConfig = JSON.parse(savedConfig);
        return !!(parsedConfig.url && parsedConfig.token);
      }
      return false;
    } catch (error) {
      return false;
    }
  });

  useEffect(() => {
    loadModelInfo();
    loadStorageInfo();
    loadGpt4allModels();
    loadAppSettings();
    loadProviderInfo();
    loadOllamaStatus();
  }, []);

  const loadModelInfo = async () => {
    setLoadingModels(true);
    try {
      const [modelsResponse, currentResponse, statusResponse] = await Promise.all([
        modelsAPI.getAvailableModels(),
        modelsAPI.getCurrentModel(),
        modelsAPI.getSystemStatus(),
      ]);

      if (modelsResponse.success) {
        setAvailableModels(modelsResponse.models);
      }

      if (currentResponse.success) {
        setCurrentModel(currentResponse.model_info);
        
        // Find the key that matches the current model name
        const currentModelName = currentResponse.model_info?.model_name;
        let foundKey = '';
        
        if (currentModelName && modelsResponse.success) {
          // Look for a model key where the model_name matches the current model
          foundKey = Object.keys(modelsResponse.models).find(key => {
            const modelConfig = modelsResponse.models[key];
            return modelConfig.model_name === currentModelName;
          }) || '';
        }
        
        setSelectedModel(foundKey);
      }

      if (statusResponse.success) {
        setSystemStatus(statusResponse);
      }
    } catch (error) {
      console.error('Error loading model info:', error);
      toast.error('Failed to load model information');
    } finally {
      setLoadingModels(false);
    }
  };

  const loadStorageInfo = async () => {
    setLoadingStorage(true);
    try {
      const [configResponse, statusResponse] = await Promise.all([
        modelsAPI.getStorageConfig(),
        modelsAPI.getStorageStatus(),
      ]);

      if (configResponse.data) {
        setStorageConfig(configResponse.data);
      }

      if (statusResponse.data && statusResponse.data.success) {
        setStorageStatus(statusResponse.data);
      }
    } catch (error) {
      console.error('Error loading storage info:', error);
      toast.error('Failed to load storage information');
    } finally {
      setLoadingStorage(false);
    }
  };

  const loadGpt4allModels = async () => {
    setLoadingGpt4allModels(true);
    try {
      const [availableResponse, downloadedResponse] = await Promise.all([
        modelsAPI.getAvailableGPT4AllModels(),
        modelsAPI.getDownloadedGPT4AllModels(),
      ]);

      if (availableResponse.success) {
        setGpt4allModels(availableResponse.models);
      }

      if (downloadedResponse.success) {
        setDownloadedModels(downloadedResponse.models);
      }
    } catch (error) {
      console.error('Error loading GPT4All models:', error);
      toast.error('Failed to load GPT4All models');
    } finally {
      setLoadingGpt4allModels(false);
    }
  };

  const loadAppSettings = async () => {
    setLoadingSettings(true);
    try {
      const response = await modelsAPI.getSettings();
      if (response.success) {
        setMaxTokens(response.max_tokens);
        setTemperature(response.temperature);
        setUseDocumentContext(response.use_document_context);
        setEnableNotifications(response.enable_notifications);
        setMaxConversationHistory(response.max_conversation_history);
        setEnableConversationContext(response.enable_conversation_context);
      }
    } catch (error) {
      console.error('Error loading settings:', error);
      toast.error('Failed to load settings');
    } finally {
      setLoadingSettings(false);
    }
  };

  const loadProviderInfo = async () => {
    setLoadingProvider(true);
    try {
      const [providerResponse, ollamaResponse] = await Promise.all([
        modelsAPI.getCurrentProvider(),
        modelsAPI.getOllamaModels(),
      ]);

      if (providerResponse.success) {
        setCurrentProvider(providerResponse.provider);
        setPreferredGpt4allModel(providerResponse.gpt4all_model || null);
        setPreferredOllamaModel(providerResponse.ollama_model || 'llama3.2:3b');
      }

      if (ollamaResponse.success) {
        setOllamaModels(ollamaResponse.models);
        setOllamaRunning(ollamaResponse.ollama_running);
      } else {
        setOllamaModels([]);
        setOllamaRunning(false);
      }
    } catch (error) {
      console.error('Error loading provider info:', error);
      toast.error('Failed to load provider information');
    } finally {
      setLoadingProvider(false);
    }
  };

  const handleProviderSwitch = async (provider: string, model?: string) => {
    setSwitchingProvider(true);
    try {
      const response = await modelsAPI.setProvider(provider, model);
      if (response.success) {
        setCurrentProvider(provider);
        if (provider === 'ollama' && model) {
          setPreferredOllamaModel(model);
        } else if (provider === 'gpt4all' && model) {
          setPreferredGpt4allModel(model);
        }
        toast.success(response.message);
        // Reload model info to reflect changes
        loadGpt4allModels();
      } else {
        toast.error('Failed to switch provider');
      }
    } catch (error: any) {
      console.error('Error switching provider:', error);
      toast.error(error.response?.data?.detail || 'Failed to switch provider');
    } finally {
      setSwitchingProvider(false);
    }
  };

  const loadOllamaStatus = async () => {
    setLoadingOllamaStatus(true);
    try {
      const response = await modelsAPI.getOllamaStatus();
      if (response.success) {
        setOllamaStatus(response);
        // For UI purposes, consider Ollama "running" if it's responding
        // This will show the correct buttons based on actual functionality
        setOllamaRunning(response.responding);
        setOllamaModels(response.models);
      } else {
        console.error('Ollama status API returned success: false', response);
      }
    } catch (error) {
      console.error('Error loading Ollama status:', error);
      // Reset status on error
      setOllamaStatus(null);
      setOllamaRunning(false);
      setOllamaModels([]);
    } finally {
      setLoadingOllamaStatus(false);
    }
  };

  const handleStartOllama = async () => {
    setControllingOllama(true);
    try {
      const response = await modelsAPI.startOllama();
      if (response.success) {
        toast.success(response.message);
        // Reload status and provider info
        await loadOllamaStatus();
        await loadProviderInfo();
      } else {
        toast.error(response.message);
        if (response.install_url) {
          toast('Please install Ollama first');
        }
      }
    } catch (error: any) {
      console.error('Error starting Ollama:', error);
      toast.error(error.response?.data?.detail || 'Failed to start Ollama');
    } finally {
      setControllingOllama(false);
    }
  };

  const handleStopOllama = async () => {
    setControllingOllama(true);
    try {
      const response = await modelsAPI.stopOllama();
      if (response.success) {
        toast.success(response.message);
        // Reload status and provider info
        await loadOllamaStatus();
        await loadProviderInfo();
      } else {
        toast.error(response.message);
      }
    } catch (error: any) {
      console.error('Error stopping Ollama:', error);
      toast.error(error.response?.data?.detail || 'Failed to stop Ollama');
    } finally {
      setControllingOllama(false);
    }
  };

  const handleRestartOllama = async () => {
    setControllingOllama(true);
    try {
      const response = await modelsAPI.restartOllama();
      if (response.success) {
        toast.success(response.message);
        // Reload status and provider info
        await loadOllamaStatus();
        await loadProviderInfo();
      } else {
        toast.error(response.message);
      }
    } catch (error: any) {
      console.error('Error restarting Ollama:', error);
      toast.error(error.response?.data?.detail || 'Failed to restart Ollama');
    } finally {
      setControllingOllama(false);
    }
  };

  const saveAppSettings = async () => {
    setSavingSettings(true);
    try {
      const response = await modelsAPI.updateSettings({
        max_tokens: maxTokens,
        temperature: temperature,
        use_document_context: useDocumentContext,
        enable_notifications: enableNotifications,
        max_conversation_history: maxConversationHistory,
        enable_conversation_context: enableConversationContext,
      });
      
      if (response.success) {
        toast.success('Settings saved successfully!');
      } else {
        toast.error('Failed to save settings');
      }
    } catch (error) {
      console.error('Error saving settings:', error);
      toast.error('Failed to save settings');
    } finally {
      setSavingSettings(false);
    }
  };

  const handleModelChange = async (modelKey: string) => {
    if (!modelKey) return;

    setChangingModel(true);
    try {
      const response = await modelsAPI.setEmbeddingModel(modelKey, systemStatus?.gpu_info?.cuda_available);
      
      if (response.success) {
        toast.success(response.message);
        await loadModelInfo(); // Reload model info
      } else {
        toast.error(response.message);
      }
    } catch (error) {
      console.error('Error changing model:', error);
      toast.error('Failed to change embedding model');
    } finally {
      setChangingModel(false);
    }
  };

  const handleTestModel = async () => {
    setTestingModel(true);
    try {
      const response = await modelsAPI.testEmbeddingModel();
      
      if (response.success) {
        toast.success(`Model test successful! Generated ${response.embedding_length}-dimensional embedding`);
      } else {
        toast.error('Model test failed');
      }
    } catch (error) {
      console.error('Error testing model:', error);
      toast.error('Failed to test embedding model');
    } finally {
      setTestingModel(false);
    }
  };

  const handleClearVectorStore = async () => {
    setClearingStore(true);
    try {
      const response = await modelsAPI.clearVectorStore();
      
      if (response.success) {
        toast.success('Vector store cleared successfully! You may need to re-upload documents.');
        await loadModelInfo(); // Reload info
      } else {
        toast.error(response.message || 'Failed to clear vector store');
      }
    } catch (error) {
      console.error('Error clearing vector store:', error);
      toast.error('Failed to clear vector store');
    } finally {
      setClearingStore(false);
    }
  };

  const handleDownloadModel = async (model: any) => {
    if (downloadingModels.has(model.filename)) return;

    setDownloadingModels(prev => new Set([...prev, model.filename]));
    
    try {
      const response = await modelsAPI.downloadGPT4AllModel(model.filename, model.download_url);
      
      if (response.success) {
        toast.success(response.message);
        
        // Start polling download progress
        pollDownloadProgress(model.filename);
      } else {
        toast.error(response.message);
        setDownloadingModels(prev => {
          const newSet = new Set(prev);
          newSet.delete(model.filename);
          return newSet;
        });
      }
    } catch (error) {
      console.error('Error downloading model:', error);
      toast.error('Failed to start model download');
      setDownloadingModels(prev => {
        const newSet = new Set(prev);
        newSet.delete(model.filename);
        return newSet;
      });
    }
  };

  const pollDownloadProgress = useCallback((filename: string) => {
    const interval = setInterval(async () => {
      try {
        const status = await modelsAPI.getDownloadStatus(filename);
        
        if (status.success) {
          setDownloadProgress(prev => ({
            ...prev,
            [filename]: status.progress
          }));
          
          if (status.is_complete) {
            clearInterval(interval);
            setDownloadingModels(prev => {
              const newSet = new Set(prev);
              newSet.delete(filename);
              return newSet;
            });
            setDownloadProgress(prev => {
              const { [filename]: _, ...rest } = prev;
              return rest;
            });
            toast.success(`Model ${filename} downloaded successfully!`);
            loadGpt4allModels(); // Refresh model list
          }
        }
      } catch (error) {
        clearInterval(interval);
        setDownloadingModels(prev => {
          const newSet = new Set(prev);
          newSet.delete(filename);
          return newSet;
        });
      }
    }, 2000); // Poll every 2 seconds

    // Clean up after 10 minutes
    setTimeout(() => {
      clearInterval(interval);
      setDownloadingModels(prev => {
        const newSet = new Set(prev);
        newSet.delete(filename);
        return newSet;
      });
    }, 600000);
  }, []);

  const handleUploadModel = useCallback(async (acceptedFiles: File[]) => {
    const file = acceptedFiles[0];
    if (!file) return;

    if (!file.name.endsWith('.gguf') && !file.name.endsWith('.bin')) {
      toast.error('Only .gguf and .bin files are supported');
      return;
    }

    setUploadingModel(true);
    try {
      const response = await modelsAPI.uploadGPT4AllModel(file);
      
      if (response.success) {
        toast.success(response.message);
        loadGpt4allModels(); // Refresh model list
      } else {
        toast.error(response.message);
      }
    } catch (error) {
      console.error('Error uploading model:', error);
      toast.error('Failed to upload model');
    } finally {
      setUploadingModel(false);
    }
  }, []);

  const { getRootProps, getInputProps } = useDropzone({
    onDrop: handleUploadModel,
    accept: {
      'application/octet-stream': ['.gguf', '.bin'],
    },
    multiple: false,
    disabled: uploadingModel,
  });

  const handleDeleteModel = async (model: any) => {
    try {
      const response = await modelsAPI.deleteGPT4AllModel(model.filename);
      
      if (response.success) {
        toast.success(response.message);
        loadGpt4allModels(); // Refresh model list
      } else {
        toast.error(response.message);
      }
    } catch (error) {
      console.error('Error deleting model:', error);
      toast.error('Failed to delete model');
    }
    
    setDeleteDialog({ open: false });
  };

  const handleSetActiveModel = async (model: any) => {
    try {
      const response = await modelsAPI.setActiveGPT4AllModel(model.filename);
      
      if (response.success) {
        toast.success(response.message);
        loadGpt4allModels(); // Refresh model list
      } else {
        toast.error(response.message);
      }
    } catch (error) {
      console.error('Error setting active model:', error);
      toast.error('Failed to set active model');
    }
  };

  const handleClearCache = async () => {
    setClearingCache(true);
    try {
      const response = await modelsAPI.clearStorageCache();
      
      if (response.data.success) {
        toast.success('Storage cache cleared successfully!');
        await loadStorageInfo(); // Reload storage info
      } else {
        toast.error('Failed to clear storage cache');
      }
    } catch (error) {
      console.error('Error clearing cache:', error);
      toast.error('Failed to clear storage cache');
    } finally {
      setClearingCache(false);
    }
  };

  const formatBytes = (bytes: number) => {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  };

  const handleTabChange = (_: React.SyntheticEvent, newValue: number) => {
    setCurrentTab(newValue);
  };

  const handleConfluenceTabChange = (_: React.SyntheticEvent, newValue: number) => {
    setConfluenceTab(newValue);
  };

  return (
    <>
      <Container maxWidth="xl" sx={{ py: 4 }}>
        <Box sx={{ mb: 4 }}>
          <Typography variant="h1" component="h1" sx={{ mb: 2 }}>
            Settings
          </Typography>
          <Typography variant="h6" color="text.secondary">
            Configure your Document Assistant preferences
          </Typography>
        </Box>

        {/* Navigation Tabs */}
        <Paper elevation={1} sx={{ mb: 4 }}>
          <Tabs
            value={currentTab}
            onChange={handleTabChange}
            variant="fullWidth"
            sx={{ borderBottom: 1, borderColor: 'divider' }}
          >
            <Tab icon={<Memory />} label="Embedding Models" />
            <Tab icon={<Psychology />} label="Language Models" />
            <Tab icon={<Storage />} label="Storage & System" />
            <Tab icon={<Settings />} label="Application" />
            <Tab icon={<Cloud />} label="Integrations" />
          </Tabs>
        </Paper>

        {/* Tab Panels */}
        <TabPanel value={currentTab} index={0}>
          {/* BERT Embedding Models */}
          <Paper elevation={2} sx={{ p: 4 }}>
            <Box sx={{ display: 'flex', alignItems: 'center', mb: 3 }}>
              <Memory sx={{ mr: 2, color: 'primary.main' }} />
              <Typography variant="h4" component="h2">
                BERT Embedding Models
              </Typography>
            </Box>

            {loadingModels ? (
              <Box sx={{ display: 'flex', justifyContent: 'center', py: 4 }}>
                <CircularProgress />
              </Box>
            ) : (
              <>
                {/* System Status Overview */}
                {systemStatus && (
                  <Card variant="outlined" sx={{ mb: 3 }}>
                    <CardContent>
                      <Typography variant="h6" sx={{ mb: 2 }}>System Overview</Typography>
                      <Grid container spacing={2}>
                        <Grid size={{ xs: 12, md: 4 }}>
                          <Box sx={{ textAlign: 'center' }}>
                            <Computer sx={{ fontSize: 40, color: 'primary.main', mb: 1 }} />
                            <Typography variant="h6">{systemStatus.document_count}</Typography>
                            <Typography variant="body2" color="text.secondary">Documents</Typography>
                          </Box>
                        </Grid>
                        <Grid size={{ xs: 12, md: 4 }}>
                          <Box sx={{ textAlign: 'center' }}>
                            <CheckCircle sx={{ 
                              fontSize: 40, 
                              color: systemStatus.gpu_info.cuda_available ? 'success.main' : 'text.secondary',
                              mb: 1 
                            }} />
                            <Typography variant="h6">
                              {systemStatus.gpu_info.cuda_available ? 'Available' : 'Not Available'}
                            </Typography>
                            <Typography variant="body2" color="text.secondary">CUDA</Typography>
                          </Box>
                        </Grid>
                        <Grid size={{ xs: 12, md: 4 }}>
                          <Box sx={{ textAlign: 'center' }}>
                            <Memory sx={{ fontSize: 40, color: 'secondary.main', mb: 1 }} />
                            <Typography variant="h6">
                              {currentModel?.config?.dimensions || 'N/A'}D
                            </Typography>
                            <Typography variant="body2" color="text.secondary">Embeddings</Typography>
                          </Box>
                        </Grid>
                      </Grid>
                      {systemStatus.gpu_info.cuda_device_name && (
                        <Alert severity="info" sx={{ mt: 2 }}>
                          GPU: {systemStatus.gpu_info.cuda_device_name}
                        </Alert>
                      )}
                    </CardContent>
                  </Card>
                )}

                {/* Current Model Info */}
                {currentModel && (
                  <Alert severity="info" sx={{ mb: 3 }}>
                    <Typography variant="subtitle2">Current Model:</Typography>
                    <Typography variant="body2">
                      <strong>{currentModel.model_name}</strong> ({currentModel.device})
                    </Typography>
                    {currentModel.config?.description && (
                      <Typography variant="body2" color="text.secondary">
                        {currentModel.config.description}
                      </Typography>
                    )}
                  </Alert>
                )}

                {/* Dimension Mismatch Warning */}
                {currentModel && selectedModel && availableModels[selectedModel] && 
                 currentModel.config?.dimensions !== availableModels[selectedModel].dimensions && (
                  <Alert severity="warning" sx={{ mb: 3 }}>
                    <Typography variant="subtitle2">Dimension Mismatch Detected</Typography>
                    <Typography variant="body2">
                      Current model uses {currentModel.config.dimensions}D embeddings, but selected model uses {availableModels[selectedModel].dimensions}D. 
                      The system will automatically handle this when you apply the new model, but existing documents may need to be re-uploaded.
                    </Typography>
                  </Alert>
                )}

                <Grid container spacing={3}>
                  <Grid size={{ xs: 12, md: 6 }}>
                    {/* Model Selection */}
                    <FormControl fullWidth sx={{ mb: 3 }}>
                      <InputLabel>Select Embedding Model</InputLabel>
                      <Select
                        value={selectedModel}
                        label="Select Embedding Model"
                        onChange={(e) => setSelectedModel(e.target.value)}
                        disabled={changingModel}
                      >
                        {Object.entries(availableModels).map(([key, model]) => (
                          <MenuItem key={key} value={key}>
                            <Box>
                              <Typography variant="body1">{key}</Typography>
                              <Typography variant="body2" color="text.secondary">
                                {model.description} • {model.dimensions}D
                              </Typography>
                            </Box>
                          </MenuItem>
                        ))}
                      </Select>
                    </FormControl>

                    {changingModel && (
                      <Box sx={{ mb: 3 }}>
                        <Typography variant="body2" sx={{ mb: 1 }}>Changing model...</Typography>
                        <LinearProgress />
                      </Box>
                    )}
                  </Grid>

                  <Grid size={{ xs: 12, md: 6 }}>
                    {/* Model Details */}
                    {selectedModel && availableModels[selectedModel] && (
                      <Card variant="outlined">
                        <CardContent>
                          <Typography variant="subtitle2" sx={{ mb: 1 }}>
                            {selectedModel} Details:
                          </Typography>
                          <Typography variant="body2" color="text.secondary">
                            Model: {availableModels[selectedModel].model_name}
                          </Typography>
                          <Typography variant="body2" color="text.secondary">
                            Dimensions: {availableModels[selectedModel].dimensions}
                          </Typography>
                          <Typography variant="body2" color="text.secondary">
                            {availableModels[selectedModel].description}
                          </Typography>
                        </CardContent>
                      </Card>
                    )}
                  </Grid>
                </Grid>

                {/* Action Buttons */}
                <Box sx={{ display: 'flex', gap: 2, flexWrap: 'wrap', mt: 3 }}>
                  <Button
                    variant="contained"
                    onClick={() => handleModelChange(selectedModel)}
                    disabled={!selectedModel || changingModel || (availableModels[selectedModel]?.model_name === currentModel?.model_name)}
                    startIcon={changingModel ? <CircularProgress size={20} /> : <Save />}
                  >
                    {changingModel ? 'Applying...' : 'Apply Model'}
                  </Button>
                  <Button
                    variant="outlined"
                    onClick={handleTestModel}
                    disabled={testingModel || !currentModel}
                    startIcon={testingModel ? <CircularProgress size={20} /> : <Speed />}
                  >
                    {testingModel ? 'Testing...' : 'Test Model'}
                  </Button>
                  <Button
                    variant="outlined"
                    onClick={handleClearVectorStore}
                    disabled={clearingStore}
                    startIcon={clearingStore ? <CircularProgress size={20} /> : <Refresh />}
                  >
                    {clearingStore ? 'Clearing...' : 'Clear Vector Store'}
                  </Button>
                  <Button
                    variant="outlined"
                    onClick={loadModelInfo}
                    disabled={loadingModels}
                    startIcon={<Refresh />}
                  >
                    Refresh
                  </Button>
                </Box>
              </>
            )}
          </Paper>
        </TabPanel>

        <TabPanel value={currentTab} index={1}>
          {/* LLM Provider Selection */}
          <Paper elevation={2} sx={{ p: 4, mb: 3 }}>
            <Box sx={{ display: 'flex', alignItems: 'center', mb: 3 }}>
              <Psychology sx={{ mr: 2, color: 'primary.main' }} />
              <Typography variant="h4" component="h2">
                Language Model Provider
              </Typography>
            </Box>

            {loadingProvider ? (
              <Box sx={{ display: 'flex', justifyContent: 'center', py: 4 }}>
                <CircularProgress />
              </Box>
            ) : (
              <>
                <Grid container spacing={3}>
                  {/* GPT4All Option */}
                  <Grid size={{ xs: 12, md: 6 }}>
                    <Card 
                      variant="outlined" 
                      sx={{ 
                        cursor: 'pointer',
                        border: currentProvider === 'gpt4all' ? 2 : 1,
                        borderColor: currentProvider === 'gpt4all' ? 'primary.main' : 'divider',
                        '&:hover': { borderColor: 'primary.main' }
                      }}
                      onClick={() => !switchingProvider && handleProviderSwitch('gpt4all')}
                    >
                      <CardContent>
                        <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                          <Computer sx={{ mr: 2, color: 'primary.main' }} />
                          <Typography variant="h6">GPT4All (Local)</Typography>
                          {currentProvider === 'gpt4all' && (
                            <Chip
                              icon={<CheckCircle />}
                              label="Active"
                              size="small"
                              color="success"
                              sx={{ ml: 'auto' }}
                            />
                          )}
                        </Box>
                        <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
                          Local models that run entirely on your machine. Good privacy, works offline.
                        </Typography>
                        <Typography variant="body2" color="text.secondary">
                          <strong>Current model:</strong> {preferredGpt4allModel || 'Auto-selected'}
                        </Typography>
                      </CardContent>
                    </Card>
                  </Grid>

                  {/* Ollama Option */}
                  <Grid size={{ xs: 12, md: 6 }}>
                    <Card 
                      variant="outlined" 
                      sx={{ 
                        cursor: 'pointer',
                        border: currentProvider === 'ollama' ? 2 : 1,
                        borderColor: currentProvider === 'ollama' ? 'primary.main' : 'divider',
                        '&:hover': { borderColor: 'primary.main' }
                      }}
                      onClick={() => !switchingProvider && ollamaRunning && handleProviderSwitch('ollama')}
                    >
                      <CardContent>
                        <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                          <Speed sx={{ mr: 2, color: ollamaRunning ? 'success.main' : 'text.disabled' }} />
                          <Typography variant="h6" color={ollamaRunning ? 'inherit' : 'text.disabled'}>
                            Ollama (Optimized)
                          </Typography>
                          {currentProvider === 'ollama' && (
                            <Chip
                              icon={<CheckCircle />}
                              label="Active"
                              size="small"
                              color="success"
                              sx={{ ml: 'auto' }}
                            />
                          )}
                        </Box>
                        <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
                          Optimized local inference engine. 3-5x faster than GPT4All.
                        </Typography>
                        {ollamaRunning ? (
                          <>
                            <Typography variant="body2" color="text.secondary">
                              <strong>Available models:</strong> {ollamaModels.length}
                            </Typography>
                            <Typography variant="body2" color="text.secondary">
                              <strong>Current model:</strong> {preferredOllamaModel}
                            </Typography>
                          </>
                        ) : (
                          <Alert severity="warning" sx={{ mt: 1 }}>
                            Ollama is not running. Install and start Ollama to enable.
                          </Alert>
                        )}
                      </CardContent>
                    </Card>
                  </Grid>
                </Grid>

                {/* Ollama model management moved to dedicated section below */}

                {/* Switch Provider Status */}
                {switchingProvider && (
                  <Box sx={{ mt: 3 }}>
                    <Typography variant="body2" sx={{ mb: 1 }}>Switching provider...</Typography>
                    <LinearProgress />
                  </Box>
                )}
              </>
            )}
          </Paper>

          {/* GPT4All Language Models - Only show when GPT4All is selected */}
          {currentProvider === 'gpt4all' && (
            <Paper elevation={2} sx={{ p: 4 }}>
              <Box sx={{ display: 'flex', alignItems: 'center', mb: 3 }}>
                <Psychology sx={{ mr: 2, color: 'primary.main' }} />
                <Typography variant="h4" component="h2">
                  GPT4All Language Models
                </Typography>
              </Box>

            {loadingGpt4allModels ? (
              <Box sx={{ display: 'flex', justifyContent: 'center', py: 4 }}>
                <CircularProgress />
              </Box>
            ) : (
              <>
                {/* Model Upload Section */}
                <Card sx={{ mb: 3, border: '2px dashed #ccc' }}>
                  <CardContent>
                    <Box
                      {...getRootProps()}
                      sx={{
                        p: 3,
                        textAlign: 'center',
                        cursor: uploadingModel ? 'not-allowed' : 'pointer',
                        '&:hover': { bgcolor: 'action.hover' },
                      }}
                    >
                      <input {...getInputProps()} />
                      <CloudUpload sx={{ fontSize: 48, color: 'primary.main', mb: 2 }} />
                      <Typography variant="h6" sx={{ mb: 1 }}>
                        {uploadingModel ? 'Uploading...' : 'Upload GPT4All Model'}
                      </Typography>
                      <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
                        Drag and drop a .gguf or .bin file here, or click to select
                      </Typography>
                      {uploadingModel && <LinearProgress sx={{ mt: 2 }} />}
                    </Box>
                  </CardContent>
                </Card>

                {/* Available Models Table */}
                <Typography variant="h6" sx={{ mb: 2 }}>
                  Available Models for Download
                </Typography>
                <TableContainer component={Paper} variant="outlined" sx={{ mb: 3 }}>
                  <Table>
                    <TableHead>
                      <TableRow>
                        <TableCell>Model</TableCell>
                        <TableCell>Size</TableCell>
                        <TableCell>Status</TableCell>
                        <TableCell align="right">Actions</TableCell>
                      </TableRow>
                    </TableHead>
                    <TableBody>
                      {gpt4allModels.map((model) => (
                        <TableRow key={model.filename}>
                          <TableCell>
                            <Box>
                              <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                                <Typography variant="body1">
                                  {model.name}
                                </Typography>
                                {model.recommended && (
                                  <Chip
                                    icon={<Star />}
                                    label="Recommended"
                                    size="small"
                                    color="warning"
                                  />
                                )}
                                {model.is_active && (
                                  <Chip
                                    icon={<CheckCircle />}
                                    label="Active"
                                    size="small"
                                    color="success"
                                  />
                                )}
                              </Box>
                              <Typography variant="body2" color="text.secondary">
                                {model.description}
                              </Typography>
                            </Box>
                          </TableCell>
                          <TableCell>{model.size_human}</TableCell>
                          <TableCell>
                            {downloadingModels.has(model.filename) ? (
                              <Box>
                                <Typography variant="body2" color="primary">
                                  Downloading... {downloadProgress[model.filename]?.toFixed(1) || 0}%
                                </Typography>
                                <LinearProgress
                                  variant="determinate"
                                  value={downloadProgress[model.filename] || 0}
                                  sx={{ mt: 1 }}
                                />
                              </Box>
                            ) : model.is_downloaded ? (
                              <Chip
                                icon={<CheckCircle />}
                                label="Downloaded"
                                color="success"
                                size="small"
                              />
                            ) : (
                              <Chip
                                label="Not Downloaded"
                                color="default"
                                size="small"
                              />
                            )}
                          </TableCell>
                          <TableCell align="right">
                            <Box sx={{ display: 'flex', gap: 1, justifyContent: 'flex-end' }}>
                              {!model.is_downloaded && !downloadingModels.has(model.filename) && (
                                <Tooltip title="Download model">
                                  <IconButton
                                    size="small"
                                    onClick={() => handleDownloadModel(model)}
                                    disabled={downloadingModels.has(model.filename)}
                                  >
                                    <Download />
                                  </IconButton>
                                </Tooltip>
                              )}
                              {model.is_downloaded && !model.is_active && (
                                <Tooltip title="Set as active model">
                                  <IconButton
                                    size="small"
                                    onClick={() => handleSetActiveModel(model)}
                                  >
                                    <PlayArrow />
                                  </IconButton>
                                </Tooltip>
                              )}
                              {model.is_downloaded && !model.is_active && (
                                <Tooltip title="Delete model">
                                  <IconButton
                                    size="small"
                                    onClick={() => setDeleteDialog({ open: true, model })}
                                    color="error"
                                  >
                                    <Delete />
                                  </IconButton>
                                </Tooltip>
                              )}
                            </Box>
                          </TableCell>
                        </TableRow>
                      ))}
                    </TableBody>
                  </Table>
                </TableContainer>

                {/* Downloaded Models Summary */}
                {downloadedModels.length > 0 && (
                  <>
                    <Typography variant="h6" sx={{ mb: 2 }}>
                      Downloaded Models ({downloadedModels.length})
                    </Typography>
                    <List>
                      {downloadedModels.map((model) => (
                        <ListItem
                          key={model.filename}
                          sx={{
                            border: '1px solid',
                            borderColor: 'divider',
                            borderRadius: 1,
                            mb: 1,
                          }}
                        >
                          <ListItemText
                            primary={
                              <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                                <Typography variant="body1">{model.name}</Typography>
                                {model.is_active && (
                                  <Chip
                                    icon={<CheckCircle />}
                                    label="Active"
                                    size="small"
                                    color="success"
                                  />
                                )}
                              </Box>
                            }
                            secondary={`${model.size_human} • ${model.filename}`}
                          />
                          <ListItemSecondaryAction>
                            <Box sx={{ display: 'flex', gap: 1 }}>
                              {!model.is_active && (
                                <Tooltip title="Set as active">
                                  <IconButton
                                    size="small"
                                    onClick={() => handleSetActiveModel(model)}
                                  >
                                    <PlayArrow />
                                  </IconButton>
                                </Tooltip>
                              )}
                              {!model.is_active && (
                                <Tooltip title="Delete model">
                                  <IconButton
                                    size="small"
                                    onClick={() => setDeleteDialog({ open: true, model })}
                                    color="error"
                                  >
                                    <Delete />
                                  </IconButton>
                                </Tooltip>
                              )}
                            </Box>
                          </ListItemSecondaryAction>
                        </ListItem>
                      ))}
                    </List>
                  </>
                )}

                <Box sx={{ display: 'flex', gap: 2, mt: 3 }}>
                  <Button
                    variant="outlined"
                    onClick={loadGpt4allModels}
                    disabled={loadingGpt4allModels}
                    startIcon={<Refresh />}
                  >
                    Refresh Models
                  </Button>
                </Box>
              </>
            )}
          </Paper>
          )}

          {/* Ollama Language Models - Only show when Ollama is selected */}
          {currentProvider === 'ollama' && (
          <Paper elevation={2} sx={{ p: 4 }}>
            <Box sx={{ display: 'flex', alignItems: 'center', mb: 3 }}>
              <Speed sx={{ mr: 2, color: 'primary.main' }} />
              <Typography variant="h4" component="h2">
                Ollama Language Models
              </Typography>
            </Box>

            {loadingProvider || loadingOllamaStatus ? (
              <Box sx={{ display: 'flex', justifyContent: 'center', py: 4 }}>
                <CircularProgress />
              </Box>
            ) : (
              <>
                {/* Ollama Process Management Section */}
                <Card sx={{ mb: 3, border: '2px solid', borderColor: ollamaRunning ? 'success.main' : 'warning.main' }}>
                  <CardContent>
                    <Box sx={{ p: 2 }}>
                      <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', mb: 2 }}>
                        <Box sx={{ display: 'flex', alignItems: 'center' }}>
                          {ollamaRunning ? (
                            <CheckCircle sx={{ fontSize: 40, color: 'success.main', mr: 2 }} />
                          ) : (
                            <Warning sx={{ fontSize: 40, color: 'warning.main', mr: 2 }} />
                          )}
                          <Box>
                            <Typography variant="h6" color={ollamaRunning ? 'success.main' : 'warning.main'}>
                              Ollama {ollamaRunning ? 'Running' : 'Not Running'}
                            </Typography>
                            <Typography variant="body2" color="text.secondary">
                              {ollamaStatus ? (
                                <>
                                  {ollamaStatus.platform} • {ollamaStatus.version ? `v${ollamaStatus.version}` : 'Unknown version'}
                                  {ollamaStatus.process_id && ` • PID: ${ollamaStatus.process_id}`}
                                </>
                              ) : (
                                'Status unknown'
                              )}
                            </Typography>
                          </Box>
                        </Box>
                        
                        {/* Process Control Buttons */}
                        <Box sx={{ display: 'flex', gap: 1 }}>
                          {/* Always show control buttons, but disable if can't control */}
                          {!ollamaRunning ? (
                            <Button
                              variant="contained"
                              color="success"
                              startIcon={<PlayArrow />}
                              onClick={handleStartOllama}
                              disabled={controllingOllama || !ollamaStatus?.can_control}
                              size="small"
                            >
                              {controllingOllama ? 'Starting...' : (ollamaStatus?.running ? 'Restart' : 'Start')}
                            </Button>
                          ) : (
                            <>
                              <Button
                                variant="outlined"
                                color="warning"
                                startIcon={<RestartAlt />}
                                onClick={handleRestartOllama}
                                disabled={controllingOllama || !ollamaStatus?.can_control}
                                size="small"
                              >
                                {controllingOllama ? 'Restarting...' : 'Restart'}
                              </Button>
                              <Button
                                variant="outlined"
                                color="error"
                                startIcon={<Stop />}
                                onClick={handleStopOllama}
                                disabled={controllingOllama || !ollamaStatus?.can_control}
                                size="small"
                              >
                                {controllingOllama ? 'Stopping...' : 'Stop'}
                              </Button>
                            </>
                          )}
                          
                          <Button
                            variant="outlined"
                            startIcon={<Refresh />}
                            onClick={loadOllamaStatus}
                            disabled={loadingOllamaStatus}
                            size="small"
                          >
                            Refresh
                          </Button>
                        </Box>
                      </Box>
                      
                      {/* Status Details */}
                      {ollamaStatus && (
                        <Box sx={{ mt: 2, p: 2, bgcolor: 'grey.50', borderRadius: 1 }}>
                          <Box sx={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 2 }}>
                            <Box>
                              <Typography variant="caption" color="text.secondary">Process</Typography>
                              <Typography variant="body2">
                                {ollamaStatus.running ? 'Running' : 'Stopped'}
                              </Typography>
                            </Box>
                            <Box>
                              <Typography variant="caption" color="text.secondary">API</Typography>
                              <Typography variant="body2">
                                {ollamaStatus.responding ? 'Responding' : 'No Response'}
                              </Typography>
                            </Box>
                            <Box>
                              <Typography variant="caption" color="text.secondary">Models</Typography>
                              <Typography variant="body2">
                                {ollamaStatus.models_count}
                              </Typography>
                            </Box>
                            <Box>
                              <Typography variant="caption" color="text.secondary">Control</Typography>
                              <Typography variant="body2">
                                {ollamaStatus.can_control ? 'Available' : 'Limited'}
                              </Typography>
                            </Box>
                          </Box>
                        </Box>
                      )}
                      
                      {/* Installation Help */}
                      {!ollamaStatus?.running && (
                        <Alert severity="info" sx={{ mt: 2 }}>
                          <Typography variant="body2" sx={{ mb: 1 }}>
                            Ollama not found or not running.
                          </Typography>
                          <Button
                            variant="outlined"
                            href="https://ollama.ai"
                            target="_blank"
                            rel="noopener noreferrer"
                            startIcon={<Launch />}
                            size="small"
                          >
                            Install Ollama
                          </Button>
                        </Alert>
                      )}
                    </Box>
                  </CardContent>
                </Card>
                
                {/* Available Ollama Models Table */}
                {ollamaRunning && (
                  <>
                    <Typography variant="h6" sx={{ mb: 2 }}>
                      Available Ollama Models
                    </Typography>
                    {ollamaModels.length === 0 ? (
                      <Card sx={{ mb: 3, border: '2px dashed #ccc' }}>
                        <CardContent>
                          <Box sx={{ textAlign: 'center', p: 3 }}>
                            <Download sx={{ fontSize: 48, color: 'text.secondary', mb: 2 }} />
                            <Typography variant="h6" sx={{ mb: 1 }}>
                              No Models Downloaded
                            </Typography>
                            <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
                              Use the terminal to download Ollama models:
                            </Typography>
                            <Typography variant="body2" sx={{ fontFamily: 'monospace', p: 2, bgcolor: 'grey.100', borderRadius: 1 }}>
                              ollama pull llama3.2:3b
                            </Typography>
                          </Box>
                        </CardContent>
                      </Card>
                    ) : (
                      <TableContainer component={Paper} variant="outlined" sx={{ mb: 3 }}>
                        <Table>
                          <TableHead>
                            <TableRow>
                              <TableCell>Model</TableCell>
                              <TableCell>Status</TableCell>
                              <TableCell>Performance</TableCell>
                              <TableCell align="right">Actions</TableCell>
                            </TableRow>
                          </TableHead>
                          <TableBody>
                            {ollamaModels.map((model) => {
                              const isActive = preferredOllamaModel === model;
                              const is3B = model.includes('3b') || model.includes('3B');
                              const is8B = model.includes('8b') || model.includes('8B');
                              const performance = is3B ? 'Fast (3-8s)' : is8B ? 'Moderate (8-15s)' : 'Variable';
                              const performanceColor = is3B ? 'success.main' : is8B ? 'warning.main' : 'text.secondary';
                              
                              return (
                                <TableRow key={model}>
                                  <TableCell>
                                    <Box>
                                      <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                                        <Typography variant="body1">
                                          {model}
                                        </Typography>
                                        {is3B && (
                                          <Chip
                                            icon={<Speed />}
                                            label="Recommended"
                                            size="small"
                                            color="success"
                                          />
                                        )}
                                        {isActive && (
                                          <Chip
                                            icon={<CheckCircle />}
                                            label="Active"
                                            size="small"
                                            color="primary"
                                          />
                                        )}
                                      </Box>
                                      <Typography variant="body2" color="text.secondary">
                                        Optimized local inference with GPU acceleration
                                      </Typography>
                                    </Box>
                                  </TableCell>
                                  <TableCell>
                                    <Chip
                                      icon={<CheckCircle />}
                                      label="Available"
                                      color="success"
                                      size="small"
                                    />
                                  </TableCell>
                                  <TableCell>
                                    <Typography variant="body2" sx={{ color: performanceColor }}>
                                      {performance}
                                    </Typography>
                                  </TableCell>
                                  <TableCell align="right">
                                    <Box sx={{ display: 'flex', gap: 1, justifyContent: 'flex-end' }}>
                                      {!isActive && (
                                        <Tooltip title="Set as active model">
                                          <IconButton
                                            size="small"
                                            onClick={() => handleProviderSwitch('ollama', model)}
                                            disabled={switchingProvider}
                                          >
                                            <PlayArrow />
                                          </IconButton>
                                        </Tooltip>
                                      )}
                                    </Box>
                                  </TableCell>
                                </TableRow>
                              );
                            })}
                          </TableBody>
                        </Table>
                      </TableContainer>
                    )}

                    <Box sx={{ display: 'flex', gap: 2, mt: 3 }}>
                      <Button
                        variant="outlined"
                        onClick={loadProviderInfo}
                        disabled={loadingProvider}
                        startIcon={<Refresh />}
                      >
                        Refresh Models
                      </Button>
                      <Button
                        variant="outlined"
                        href="https://ollama.ai/library"
                        target="_blank"
                        rel="noopener noreferrer"
                        startIcon={<Launch />}
                      >
                        Browse Model Library
                      </Button>
                    </Box>
                  </>
                )}
              </>
            )}
          </Paper>
          )}
        </TabPanel>

        <TabPanel value={currentTab} index={2}>
          {/* Storage & System Settings */}
          <Grid container spacing={4}>
            <Grid size={{ xs: 12, md: 6 }}>
              {/* Storage Settings */}
              <Paper elevation={2} sx={{ p: 4 }}>
                <Box sx={{ display: 'flex', alignItems: 'center', mb: 3 }}>
                  <Storage sx={{ mr: 2, color: 'primary.main' }} />
                  <Typography variant="h4" component="h2">
                    Storage Settings
                  </Typography>
                </Box>
                
                {loadingStorage ? (
                  <Box sx={{ display: 'flex', justifyContent: 'center', py: 4 }}>
                    <CircularProgress />
                  </Box>
                ) : (
                  <Box sx={{ '& .MuiTextField-root': { mb: 3 } }}>
                    <TextField
                      fullWidth
                      label="Documents Directory"
                      value={storageConfig?.documents_directory || 'Loading...'}
                      variant="outlined"
                      helperText="Location where uploaded documents are stored"
                      InputProps={{ readOnly: true }}
                    />
                    
                    <TextField
                      fullWidth
                      label="Vector Database Path"
                      value={storageConfig?.vector_database_path || 'Loading...'}
                      variant="outlined"
                      helperText="Location of the vector database for document embeddings"
                      InputProps={{ readOnly: true }}
                    />

                    <TextField
                      fullWidth
                      label="Models Directory"
                      value={storageConfig?.models_directory || 'Loading...'}
                      variant="outlined"
                      helperText="Location where AI models are stored"
                      InputProps={{ readOnly: true }}
                    />

                    <Box sx={{ display: 'flex', gap: 2, mt: 3 }}>
                      <Button
                        variant="outlined"
                        onClick={handleClearCache}
                        disabled={clearingCache}
                        startIcon={clearingCache ? <CircularProgress size={20} /> : <Refresh />}
                      >
                        {clearingCache ? 'Clearing...' : 'Clear Cache'}
                      </Button>
                      <Button
                        variant="outlined"
                        onClick={loadStorageInfo}
                        disabled={loadingStorage}
                        startIcon={<Refresh />}
                      >
                        Refresh
                      </Button>
                    </Box>
                  </Box>
                )}
              </Paper>
            </Grid>

            <Grid size={{ xs: 12, md: 6 }}>
              {/* Storage Status */}
              <Paper elevation={2} sx={{ p: 4 }}>
                <Box sx={{ display: 'flex', alignItems: 'center', mb: 3 }}>
                  <Assessment sx={{ mr: 2, color: 'primary.main' }} />
                  <Typography variant="h4" component="h2">
                    Storage Status
                  </Typography>
                </Box>

                {storageStatus && (
                  <Grid container spacing={2}>
                    <Grid size={{ xs: 6 }}>
                      <Card variant="outlined">
                        <CardContent sx={{ textAlign: 'center', py: 3 }}>
                          <Folder sx={{ fontSize: 40, color: 'primary.main', mb: 1 }} />
                          <Typography variant="h4" color="primary">
                            {storageStatus.storage_status?.documents_directory?.file_count || 0}
                          </Typography>
                          <Typography variant="body2" color="text.secondary">
                            Documents
                          </Typography>
                        </CardContent>
                      </Card>
                    </Grid>
                    <Grid size={{ xs: 6 }}>
                      <Card variant="outlined">
                        <CardContent sx={{ textAlign: 'center', py: 3 }}>
                          <Memory sx={{ fontSize: 40, color: 'secondary.main', mb: 1 }} />
                          <Typography variant="h4" color="secondary">
                            {storageStatus.storage_status?.models_directory?.file_count || 0}
                          </Typography>
                          <Typography variant="body2" color="text.secondary">
                            Models
                          </Typography>
                        </CardContent>
                      </Card>
                    </Grid>
                    <Grid size={{ xs: 12 }}>
                      <Card variant="outlined">
                        <CardContent sx={{ textAlign: 'center', py: 2 }}>
                          <Typography variant="body2" color="text.secondary">
                            Vector Store: {storageStatus.storage_status?.vector_store_connected ? 'Connected' : 'Disconnected'}
                          </Typography>
                        </CardContent>
                      </Card>
                    </Grid>
                  </Grid>
                )}
              </Paper>
            </Grid>
          </Grid>
        </TabPanel>

        <TabPanel value={currentTab} index={3}>
          {/* Application Settings */}
          <Grid container spacing={4}>
            <Grid size={{ xs: 12, md: 6 }}>
              {/* AI Response Settings */}
              <Paper elevation={2} sx={{ p: 4, mb: 4 }}>
                <Box sx={{ display: 'flex', alignItems: 'center', mb: 3 }}>
                  <Psychology sx={{ mr: 2, color: 'primary.main' }} />
                  <Typography variant="h4" component="h2">
                    AI Response Settings
                  </Typography>
                </Box>
                
                <Box sx={{ '& .MuiFormControl-root': { mb: 3 } }}>
                  <Box sx={{ mb: 3 }}>
                    <Typography gutterBottom>Response Temperature</Typography>
                    <Slider
                      value={temperature}
                      onChange={(_, value) => setTemperature(value as number)}
                      min={0}
                      max={1}
                      step={0.1}
                      marks
                      valueLabelDisplay="auto"
                      disabled={loadingSettings}
                    />
                    <Typography variant="body2" color="text.secondary">
                      Controls creativity vs consistency in responses
                    </Typography>
                  </Box>

                  <Box sx={{ mb: 3 }}>
                    <Typography gutterBottom>Max Response Length</Typography>
                    <Slider
                      value={maxTokens}
                      onChange={(_, value) => setMaxTokens(value as number)}
                      min={128}
                      max={2048}
                      step={128}
                      marks={[
                        { value: 128, label: '128' },
                        { value: 512, label: '512' },
                        { value: 1024, label: '1K' },
                        { value: 1536, label: '1.5K' },
                        { value: 2048, label: '2K' }
                      ]}
                      valueLabelDisplay="auto"
                      disabled={loadingSettings}
                    />
                    <Typography variant="body2" color="text.secondary">
                      Maximum tokens for AI responses. Higher values may lead to repetitive content. (128-2048 tokens)
                    </Typography>
                  </Box>

                  <FormControlLabel
                    control={
                      <Switch 
                        checked={useDocumentContext}
                        onChange={(e) => setUseDocumentContext(e.target.checked)}
                        disabled={loadingSettings}
                      />
                    }
                    label="Use document context in responses"
                  />

                  <FormControlLabel
                    control={
                      <Switch 
                        checked={enableNotifications}
                        onChange={(e) => setEnableNotifications(e.target.checked)}
                        disabled={loadingSettings}
                      />
                    }
                    label="Enable browser notifications"
                  />

                  {/* Conversation Context Settings */}
                  <Divider sx={{ my: 3 }} />
                  <Typography variant="h6" gutterBottom>
                    Conversation Context
                  </Typography>

                  <FormControlLabel
                    control={
                      <Switch 
                        checked={enableConversationContext}
                        onChange={(e) => setEnableConversationContext(e.target.checked)}
                        disabled={loadingSettings}
                      />
                    }
                    label="Include conversation history in responses"
                  />
                  <Typography variant="body2" color="text.secondary" sx={{ mb: 3 }}>
                    When enabled, the AI remembers previous messages in the conversation for better context.
                  </Typography>

                  <Box sx={{ mb: 3 }}>
                    <Typography gutterBottom>Max Conversation History</Typography>
                    <Slider
                      value={maxConversationHistory}
                      onChange={(_, value) => setMaxConversationHistory(value as number)}
                      min={0}
                      max={50}
                      step={5}
                      marks={[
                        { value: 0, label: '0' },
                        { value: 10, label: '10' },
                        { value: 20, label: '20' },
                        { value: 30, label: '30' },
                        { value: 50, label: '50' }
                      ]}
                      valueLabelDisplay="auto"
                      disabled={loadingSettings || !enableConversationContext}
                    />
                    <Typography variant="body2" color="text.secondary">
                      Maximum number of previous messages to include for context. Set to 0 to disable history. (Higher values use more tokens)
                    </Typography>
                  </Box>

                  <Box sx={{ display: 'flex', gap: 2, mt: 3 }}>
                    <Button
                      variant="contained"
                      onClick={saveAppSettings}
                      disabled={savingSettings || loadingSettings}
                      startIcon={savingSettings ? <CircularProgress size={20} /> : <Save />}
                    >
                      {savingSettings ? 'Saving...' : 'Save Settings'}
                    </Button>
                    <Button
                      variant="outlined"
                      onClick={loadAppSettings}
                      disabled={loadingSettings}
                      startIcon={<Refresh />}
                    >
                      Refresh
                    </Button>
                  </Box>
                </Box>
              </Paper>

              {/* General Application Settings */}
              <Paper elevation={2} sx={{ p: 4 }}>
                <Box sx={{ display: 'flex', alignItems: 'center', mb: 3 }}>
                  <Settings sx={{ mr: 2, color: 'primary.main' }} />
                  <Typography variant="h4" component="h2">
                    General Settings
                  </Typography>
                </Box>
                
                <Box sx={{ '& .MuiFormControl-root': { mb: 3 } }}>
                  <FormControlLabel
                    control={
                      <Switch 
                        checked={enableNotifications}
                        onChange={(e) => setEnableNotifications(e.target.checked)}
                        disabled={loadingSettings}
                      />
                    }
                    label="Enable notifications"
                    sx={{ mb: 2 }}
                  />

                  <Alert severity="info" sx={{ mt: 2 }}>
                    <Typography variant="subtitle2">Privacy & Security:</Typography>
                    <Typography variant="body2">
                      This application is designed to work completely offline. All your data stays private and local to your device. 
                      No information is sent to external servers.
                    </Typography>
                  </Alert>
                </Box>
              </Paper>
            </Grid>

            <Grid size={{ xs: 12, md: 6 }}>
              {/* Data Management */}
              <Paper elevation={2} sx={{ p: 4 }}>
                <Box sx={{ display: 'flex', alignItems: 'center', mb: 3 }}>
                  <Computer sx={{ mr: 2, color: 'primary.main' }} />
                  <Typography variant="h4" component="h2">
                    Data Management
                  </Typography>
                </Box>
                
                <Box>
                  <Typography variant="body1" sx={{ mb: 3 }}>
                    Manage your application data and reset options.
                  </Typography>

                  <Divider sx={{ my: 3 }} />

                  <Box sx={{ display: 'flex', gap: 2, flexWrap: 'wrap' }}>
                    <Button 
                      variant="outlined" 
                      startIcon={<Folder />}
                      onClick={() => toast.success('Export functionality coming soon!')}
                    >
                      Export Data
                    </Button>
                    
                    <Button 
                      variant="outlined" 
                      color="error" 
                      startIcon={<Delete />}
                      onClick={() => toast.error('This would delete all data. Feature disabled for safety.')}
                    >
                      Reset All Data
                    </Button>
                  </Box>

                  <Alert severity="warning" sx={{ mt: 3 }}>
                    <Typography variant="subtitle2">Important:</Typography>
                    <Typography variant="body2">
                      All operations are performed locally. Your documents and settings never leave your device.
                    </Typography>
                  </Alert>
                </Box>
              </Paper>
            </Grid>
          </Grid>
        </TabPanel>

        <TabPanel value={currentTab} index={4}>
          {/* Integrations Settings */}
          <Grid container spacing={4}>
            <Grid size={{ xs: 12 }}>
              {/* Confluence Section */}
              <Paper elevation={2} sx={{ p: 4, mb: 4 }}>
                <Box sx={{ mb: 4 }}>
                  <Typography variant="h4" component="h2" sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                    <Cloud sx={{ mr: 2 }} /> Confluence Integration
                  </Typography>
                  <Typography color="text.secondary">
                    Connect and manage your Confluence workspace integration
                  </Typography>
                </Box>

                {/* Tabs for different Confluence sections */}
                <Tabs 
                  value={confluenceTab} 
                  onChange={handleConfluenceTabChange}
                  sx={{ borderBottom: 1, borderColor: 'divider', mb: 3 }}
                >
                  <Tab label="Configuration" />
                  <Tab label="Templates" disabled={!isConfigured} />
                  <Tab label="Sync Pages" disabled={!isConfigured} />
                </Tabs>

                {/* Configuration Tab */}
                <TabPanel value={confluenceTab} index={0}>
                  <ConfluenceSettings onConfigured={() => setIsConfigured(true)} />
                </TabPanel>

                {/* Templates Tab */}
                <TabPanel value={confluenceTab} index={1}>
                  <TemplateManagement />
                </TabPanel>

                {/* Sync Pages Tab */}
                <TabPanel value={confluenceTab} index={2}>
                  <SyncManagement />
                </TabPanel>
              </Paper>

              {/* Other Integrations */}
              <Paper elevation={2} sx={{ p: 4 }}>
                <Typography variant="h4" component="h2" sx={{ mb: 3 }}>
                  Other Integrations
                </Typography>
                <Alert severity="info">
                  More integrations coming soon...
                </Alert>
              </Paper>
            </Grid>
          </Grid>
        </TabPanel>
      </Container>

      {/* Delete Model Confirmation Dialog */}
      <Dialog
        open={deleteDialog.open}
        onClose={() => setDeleteDialog({ open: false })}
        maxWidth="sm"
        fullWidth
      >
        <DialogTitle>
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
            <Warning color="warning" />
            Delete Model
          </Box>
        </DialogTitle>
        <DialogContent>
          <Typography>
            Are you sure you want to delete the model <strong>{deleteDialog.model?.name}</strong>?
          </Typography>
          <Typography variant="body2" color="text.secondary" sx={{ mt: 1 }}>
            This action cannot be undone. You'll need to download the model again to use it.
          </Typography>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setDeleteDialog({ open: false })}>
            Cancel
          </Button>
          <Button
            onClick={() => handleDeleteModel(deleteDialog.model)}
            color="error"
            variant="contained"
          >
            Delete
          </Button>
        </DialogActions>
      </Dialog>
    </>
  );
};

export default SettingsPage; 