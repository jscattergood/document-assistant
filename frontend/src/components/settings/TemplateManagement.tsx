import React, { useState, useEffect } from 'react';
import {
  Box,
  Typography,
  TextField,
  Button,
  List,
  ListItem,
  ListItemText,
  CircularProgress,
  Alert,
  Divider,
  Card,
  CardContent,
} from '@mui/material';
import { AutoAwesome } from '@mui/icons-material';
import { confluenceAPI } from '../../services/api';
import toast from 'react-hot-toast';

interface ContentTemplate {
  name: string;
  description: string;
  template: string;
}

export const TemplateManagement: React.FC = () => {
  const [templates, setTemplates] = useState<Record<string, ContentTemplate>>({});
  const [templatePageUrls, setTemplatePageUrls] = useState<string>('');
  const [loadingCustomTemplates, setLoadingCustomTemplates] = useState(false);
  const [config, setConfig] = useState({
    url: '',
    username: '',
    token: '',
    space_key: '',
    auth_type: 'pat' as const,
  });

  useEffect(() => {
    loadSavedConfig();
    loadContentTemplates();
  }, []);

  const loadSavedConfig = () => {
    try {
      const savedConfig = localStorage.getItem('confluence_config');
      if (savedConfig) {
        const parsedConfig = JSON.parse(savedConfig);
        setConfig({
          url: parsedConfig.url || '',
          username: parsedConfig.username || '',
          token: parsedConfig.token || '',
          space_key: parsedConfig.space_key || '',
          auth_type: parsedConfig.auth_type || 'pat',
        });
      }
    } catch (error) {
      console.error('Error loading saved config:', error);
    }
  };

  const loadContentTemplates = async () => {
    setLoadingCustomTemplates(true);
    try {
      const customTemplateUrls = templatePageUrls.split('\n').filter(url => url.trim());
      
      if (customTemplateUrls.length > 0) {
        const credentials = {
          url: config.url,
          username: config.username,
          api_token: config.token,
          auth_type: config.auth_type,
        };

        const confluenceResult = await confluenceAPI.getConfluenceTemplates(credentials, config.space_key, customTemplateUrls);
        
        if (confluenceResult.success && Object.keys(confluenceResult.templates).length > 0) {
          setTemplates(confluenceResult.templates);
          localStorage.setItem('custom_template_urls', JSON.stringify(customTemplateUrls));
          toast.success(`Created ${Object.keys(confluenceResult.templates).length} templates from your Confluence pages`);
          return;
        }
      }
      
      // Fallback to static templates
      const result = await confluenceAPI.getContentTemplates();
      if (result.success) {
        setTemplates(result.templates);
      }
    } catch (error) {
      console.error('Error loading templates:', error);
      toast.error('Failed to load templates');
    } finally {
      setLoadingCustomTemplates(false);
    }
  };

  return (
    <Box>
      {/* Custom Templates Section */}
      <Box sx={{ mb: 4 }}>
        <Typography variant="h6" gutterBottom>
          Custom Templates
        </Typography>
        <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
          Import templates from existing Confluence pages. The system will analyze their structure and create reusable templates.
        </Typography>

        <TextField
          fullWidth
          multiline
          rows={4}
          label="Template Page URLs"
          placeholder="https://your-wiki.atlassian.net/wiki/spaces/DOCS/pages/123456/Template"
          value={templatePageUrls}
          onChange={(e) => setTemplatePageUrls(e.target.value)}
          helperText="One URL per line. These pages will be analyzed to create new templates."
          sx={{ mb: 3 }}
        />

        <Box sx={{ display: 'flex', gap: 2, mb: 4 }}>
          <Button
            variant="contained"
            onClick={loadContentTemplates}
            disabled={loadingCustomTemplates || !templatePageUrls.trim()}
            startIcon={loadingCustomTemplates ? <CircularProgress size={20} /> : <AutoAwesome />}
          >
            {loadingCustomTemplates ? 'Loading...' : 'Create Templates'}
          </Button>
        </Box>
      </Box>

      {/* Available Templates Section */}
      <Box>
        <Typography variant="h6" gutterBottom>
          Available Templates
        </Typography>
        {Object.keys(templates).length === 0 ? (
          <Alert severity="info">
            No templates available. Add template URLs above to create custom templates.
          </Alert>
        ) : (
          <List>
            {Object.entries(templates).map(([key, template], index) => (
              <React.Fragment key={key}>
                <ListItem>
                  <ListItemText
                    primary={template.name}
                    secondary={template.description}
                  />
                </ListItem>
                {index < Object.entries(templates).length - 1 && <Divider />}
              </React.Fragment>
            ))}
          </List>
        )}
      </Box>
    </Box>
  );
}; 