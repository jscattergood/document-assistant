import React, { useState, useEffect } from 'react';
import {
  Box,
  Typography,
  TextField,
  Button,
  List,
  ListItem,
  ListItemText,
  ListItemSecondaryAction,
  IconButton,
  CircularProgress,
  Alert,
  Divider,
  Card,
  CardContent,
  Chip,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
} from '@mui/material';
import { 
  AutoAwesome, 
  Delete, 
  Sync, 
  Add,
  InfoOutlined,
  Schedule,
  Link as LinkIcon
} from '@mui/icons-material';
import { templateAPI, confluenceAPI, type Template, type ConfluenceCredentials } from '../../services/api';
import toast from 'react-hot-toast';

export const TemplateManagement: React.FC = () => {
  const [templates, setTemplates] = useState<Template[]>([]);
  const [templateUrl, setTemplateUrl] = useState<string>('');
  const [loading, setLoading] = useState(false);
  const [syncing, setSyncing] = useState<string | null>(null);
  const [syncingAll, setSyncingAll] = useState(false);
  const [showAddDialog, setShowAddDialog] = useState(false);
  const [config, setConfig] = useState({
    url: '',
    username: '',
    api_token: '',
    space_key: '',
    auth_type: 'pat' as const,
  });

  useEffect(() => {
    loadSavedConfig();
    loadTemplates();
  }, []);

  const loadSavedConfig = () => {
    try {
      const savedConfig = localStorage.getItem('confluence_config');
      if (savedConfig) {
        const parsedConfig = JSON.parse(savedConfig);
        setConfig({
          url: parsedConfig.url || '',
          username: parsedConfig.username || '',
          api_token: parsedConfig.token || parsedConfig.api_token || '',
          space_key: parsedConfig.space_key || '',
          auth_type: parsedConfig.auth_type || 'pat',
        });
      }
    } catch (error) {
      console.error('Error loading saved config:', error);
    }
  };

  const loadTemplates = async () => {
    setLoading(true);
    try {
      const result = await templateAPI.listTemplates();
      if (result.success) {
        setTemplates(result.templates);
      } else {
        toast.error('Failed to load templates');
      }
    } catch (error) {
      console.error('Error loading templates:', error);
      toast.error('Failed to load templates');
    } finally {
      setLoading(false);
    }
  };

  const addTemplate = async () => {
    if (!templateUrl.trim()) {
      toast.error('Please enter a Confluence page URL');
      return;
    }

    if (!config.url || !config.api_token) {
      toast.error('Please configure Confluence credentials in settings first');
      return;
    }

    setLoading(true);
    try {
      const credentials: ConfluenceCredentials = {
        url: config.url,
        username: config.username,
        api_token: config.api_token,
        auth_type: config.auth_type,
      };

      const result = await templateAPI.createTemplate(templateUrl, credentials);
      if (result.success) {
        toast.success(`Template "${result.template?.name}" created successfully`);
        setTemplateUrl('');
        setShowAddDialog(false);
        await loadTemplates();
      } else {
        toast.error(result.message || 'Failed to create template');
      }
    } catch (error) {
      console.error('Error creating template:', error);
      toast.error('Failed to create template from URL');
    } finally {
      setLoading(false);
    }
  };

  const deleteTemplate = async (templateId: string, templateName: string) => {
    if (!confirm(`Are you sure you want to delete the template "${templateName}"?`)) {
      return;
    }

    try {
      const result = await templateAPI.deleteTemplate(templateId);
      if (result.success) {
        toast.success(`Template "${templateName}" deleted successfully`);
        await loadTemplates();
      } else {
        toast.error(result.message || 'Failed to delete template');
      }
    } catch (error) {
      console.error('Error deleting template:', error);
      toast.error('Failed to delete template');
    }
  };

  const syncTemplate = async (templateId: string, templateName: string) => {
    if (!config.url || !config.api_token) {
      toast.error('Please configure Confluence credentials in settings first');
      return;
    }

    setSyncing(templateId);
    try {
      const credentials: ConfluenceCredentials = {
        url: config.url,
        username: config.username,
        api_token: config.api_token,
        auth_type: config.auth_type,
      };

      const result = await templateAPI.syncTemplate(templateId, credentials);
      if (result.success) {
        toast.success(`Template "${templateName}" synced successfully`);
        await loadTemplates();
      } else {
        toast.error(result.message || 'Failed to sync template');
      }
    } catch (error) {
      console.error('Error syncing template:', error);
      toast.error('Failed to sync template');
    } finally {
      setSyncing(null);
    }
  };

  const syncAllTemplates = async () => {
    if (!config.url || !config.api_token) {
      toast.error('Please configure Confluence credentials in settings first');
      return;
    }

    const syncableTemplates = templates.filter(t => t.source_url && t.sync_enabled);
    if (syncableTemplates.length === 0) {
      toast.success('No templates available for syncing');
      return;
    }

    setSyncingAll(true);
    try {
      const credentials: ConfluenceCredentials = {
        url: config.url,
        username: config.username,
        api_token: config.api_token,
        auth_type: config.auth_type,
      };

      const result = await templateAPI.syncAllTemplates(credentials);
      if (result.success) {
        toast.success(result.message);
        await loadTemplates();
      } else {
        toast.error(result.message || 'Failed to sync templates');
      }
    } catch (error) {
      console.error('Error syncing all templates:', error);
      toast.error('Failed to sync templates');
    } finally {
      setSyncingAll(false);
    }
  };

  const formatDate = (dateString: string) => {
    return new Date(dateString).toLocaleDateString('en-US', {
      year: 'numeric',
      month: 'short',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit',
    });
  };

  const getTemplateTypeColor = (type: string) => {
    switch (type) {
      case 'confluence':
        return 'primary';
      case 'custom':
        return 'secondary';
      case 'builtin':
        return 'default';
      default:
        return 'default';
    }
  };

  return (
    <Box>
      {/* Header Section */}
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
        <Box>
          <Typography variant="h6" gutterBottom>
            Template Management
          </Typography>
          <Typography variant="body2" color="text.secondary">
            Manage your custom templates imported from Confluence pages.
          </Typography>
        </Box>
        <Box sx={{ display: 'flex', gap: 2 }}>
          <Button
            variant="outlined"
            onClick={syncAllTemplates}
            disabled={syncingAll || templates.filter(t => t.source_url && t.sync_enabled).length === 0}
            startIcon={syncingAll ? <CircularProgress size={20} /> : <Sync />}
          >
            {syncingAll ? 'Syncing...' : 'Sync All'}
          </Button>
          <Button
            variant="contained"
            onClick={() => setShowAddDialog(true)}
            startIcon={<Add />}
          >
            Add Template
          </Button>
        </Box>
      </Box>

      {/* Templates List */}
      <Box>
        {loading ? (
          <Box sx={{ display: 'flex', justifyContent: 'center', py: 4 }}>
            <CircularProgress />
          </Box>
        ) : templates.length === 0 ? (
          <Card>
            <CardContent sx={{ textAlign: 'center', py: 4 }}>
              <InfoOutlined sx={{ fontSize: 48, color: 'text.secondary', mb: 2 }} />
              <Typography variant="h6" gutterBottom>
                No Templates Found
              </Typography>
              <Typography variant="body2" color="text.secondary" sx={{ mb: 3 }}>
                Create your first template by importing from a Confluence page.
              </Typography>
              <Button
                variant="contained"
                onClick={() => setShowAddDialog(true)}
                startIcon={<Add />}
              >
                Add Your First Template
              </Button>
            </CardContent>
          </Card>
        ) : (
          <List>
            {templates.map((template, index) => (
              <React.Fragment key={template.id}>
                <ListItem sx={{ py: 2 }}>
                  <ListItemText
                    primary={
                      <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 1 }}>
                        <Typography variant="subtitle1" component="span">
                          {template.name}
                        </Typography>
                        <Chip
                          label={template.template_type}
                          size="small"
                          color={getTemplateTypeColor(template.template_type)}
                        />
                        {template.source_url && (
                          <Chip
                            icon={<LinkIcon />}
                            label="Synced"
                            size="small"
                            variant="outlined"
                          />
                        )}
                      </Box>
                    }
                    secondary={
                      <Box>
                        <Typography variant="body2" color="text.secondary" sx={{ mb: 1 }}>
                          {template.description}
                        </Typography>
                        <Box sx={{ display: 'flex', alignItems: 'center', gap: 2, flexWrap: 'wrap' }}>
                          <Typography variant="caption" color="text.secondary">
                            Created: {formatDate(template.created_at)}
                          </Typography>
                          {template.last_synced && (
                            <Typography variant="caption" color="text.secondary">
                              Last synced: {formatDate(template.last_synced)}
                            </Typography>
                          )}
                          {template.sections.length > 0 && (
                            <Typography variant="caption" color="text.secondary">
                              {template.sections.length} sections
                            </Typography>
                          )}
                        </Box>
                      </Box>
                    }
                  />
                  <ListItemSecondaryAction>
                    <Box sx={{ display: 'flex', gap: 1 }}>
                      {template.source_url && template.sync_enabled && (
                        <IconButton
                          edge="end"
                          onClick={() => syncTemplate(template.id, template.name)}
                          disabled={syncing === template.id}
                          title="Sync with Confluence"
                        >
                          {syncing === template.id ? (
                            <CircularProgress size={20} />
                          ) : (
                            <Sync />
                          )}
                        </IconButton>
                      )}
                      <IconButton
                        edge="end"
                        onClick={() => deleteTemplate(template.id, template.name)}
                        title="Delete template"
                        color="error"
                      >
                        <Delete />
                      </IconButton>
                    </Box>
                  </ListItemSecondaryAction>
                </ListItem>
                {index < templates.length - 1 && <Divider />}
              </React.Fragment>
            ))}
          </List>
        )}
      </Box>

      {/* Add Template Dialog */}
      <Dialog
        open={showAddDialog}
        onClose={() => setShowAddDialog(false)}
        maxWidth="sm"
        fullWidth
      >
        <DialogTitle>Add New Template</DialogTitle>
        <DialogContent>
          <Typography variant="body2" color="text.secondary" sx={{ mb: 3 }}>
            Import a template from an existing Confluence page. The system will analyze its structure and create a reusable template.
          </Typography>
          <TextField
            fullWidth
            label="Confluence Page URL"
            placeholder="https://your-wiki.atlassian.net/wiki/spaces/DOCS/pages/123456/Template"
            value={templateUrl}
            onChange={(e) => setTemplateUrl(e.target.value)}
            helperText="Enter the full URL of the Confluence page to import as a template"
            sx={{ mb: 2 }}
          />
          {(!config.url || !config.api_token) && (
            <Alert severity="warning" sx={{ mt: 2 }}>
              Please configure your Confluence credentials in the Confluence Settings section first.
            </Alert>
          )}
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setShowAddDialog(false)}>
            Cancel
          </Button>
          <Button
            onClick={addTemplate}
            variant="contained"
            disabled={loading || !templateUrl.trim() || !config.url || !config.api_token}
            startIcon={loading ? <CircularProgress size={20} /> : <Add />}
          >
            {loading ? 'Creating...' : 'Create Template'}
          </Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
}; 