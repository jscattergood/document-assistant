import React, { useState, useEffect } from 'react';
import {
  Box,
  Typography,
  Button,
  List,
  ListItem,
  ListItemText,
  ListItemSecondaryAction,
  IconButton,
  CircularProgress,
  Alert,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  TextField,
  Divider,
} from '@mui/material';
import {
  Add,
  Delete,
  PlayArrow,
  Link,
  Sync,
} from '@mui/icons-material';
import { confluenceAPI } from '../../services/api';
import toast from 'react-hot-toast';

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

export const SyncManagement: React.FC = () => {
  const [syncPages, setSyncPages] = useState<SyncPage[]>([]);
  const [loadingSyncPages, setLoadingSyncPages] = useState(false);
  const [syncing, setSyncing] = useState(false);
  const [addUrlsDialog, setAddUrlsDialog] = useState(false);
  const [urlsToAdd, setUrlsToAdd] = useState('');
  const [config, setConfig] = useState({
    url: '',
    username: '',
    token: '',
    space_key: '',
    auth_type: 'pat' as const,
  });

  useEffect(() => {
    loadSavedConfig();
    loadSyncPages();
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

  const loadSyncPages = async () => {
    setLoadingSyncPages(true);
    try {
      const result = await confluenceAPI.listSyncPages();
      if (result.success) {
        setSyncPages(result.pages);
      }
    } catch (error) {
      console.error('Error loading sync pages:', error);
      toast.error('Failed to load sync pages');
    } finally {
      setLoadingSyncPages(false);
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

  return (
    <Box>
      <Box sx={{ mb: 4 }}>
        <Typography variant="h6" gutterBottom>
          Synchronized Pages
        </Typography>
        <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
          Manage pages that are automatically synced to your document index.
        </Typography>

        <Box sx={{ display: 'flex', gap: 2, mb: 3 }}>
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
        </Box>

        {loadingSyncPages ? (
          <CircularProgress />
        ) : syncPages.length === 0 ? (
          <Alert severity="info">
            No pages are currently being synced. Add some pages to get started.
          </Alert>
        ) : (
          <List>
            {syncPages.map((page, index) => (
              <React.Fragment key={page.id}>
                <ListItem>
                  <ListItemText
                    primary={page.title}
                    secondary={
                      <>
                        Space: {page.space_key}
                        {page.last_synced && (
                          <> • Last synced: {new Date(page.last_synced).toLocaleString()}</>
                        )}
                      </>
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
        )}
      </Box>

      {/* Add URLs Dialog */}
      <Dialog
        open={addUrlsDialog}
        onClose={() => setAddUrlsDialog(false)}
        maxWidth="md"
        fullWidth
      >
        <DialogTitle>Add Pages to Sync</DialogTitle>
        <DialogContent>
          <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
            Enter the URLs of the Confluence pages you want to sync, one per line.
          </Typography>
          <TextField
            fullWidth
            multiline
            rows={4}
            value={urlsToAdd}
            onChange={(e) => setUrlsToAdd(e.target.value)}
            placeholder="https://your-wiki.atlassian.net/wiki/spaces/DOCS/pages/123456/Page1&#10;https://your-wiki.atlassian.net/wiki/spaces/DOCS/pages/123457/Page2"
          />
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setAddUrlsDialog(false)}>Cancel</Button>
          <Button
            variant="contained"
            onClick={handleAddUrls}
            disabled={!urlsToAdd.trim()}
          >
            Add Pages
          </Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
}; 