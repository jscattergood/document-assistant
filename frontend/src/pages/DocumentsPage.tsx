import React, { useState, useEffect, useCallback } from 'react';
import {
  Container,
  Typography,
  Paper,
  Box,
  Button,
  Card,
  CardContent,
  Grid,
  Chip,
  List,
  ListItem,
  ListItemText,
  ListItemIcon,
  ListItemSecondaryAction,
  IconButton,
  LinearProgress,
  Alert,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  TextField,
  Divider,
  Link as MuiLink,
} from '@mui/material';
import {
  CloudUpload,
  Description,
  Add,
  Delete,
  GetApp,
  PictureAsPdf,
  Article,
  Code,
  Link,
} from '@mui/icons-material';
import { useDropzone } from 'react-dropzone';
import { documentAPI, confluenceAPI } from '../services/api';
import type { Document } from '../services/api';
import toast from 'react-hot-toast';

const DocumentsPage: React.FC = () => {
  const [documents, setDocuments] = useState<Document[]>([]);
  const [loading, setLoading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState<{ [key: string]: number }>({});
  const [deleteDialog, setDeleteDialog] = useState<{ open: boolean; document?: Document }>({ open: false });
  const [confluenceUrl, setConfluenceUrl] = useState('');
  const [uploadingFromUrl, setUploadingFromUrl] = useState(false);

  // Fetch documents on component mount
  useEffect(() => {
    fetchDocuments();
  }, []);

  const fetchDocuments = async () => {
    try {
      setLoading(true);
      const response = await documentAPI.getDocuments();
      if (response.success && response.documents) {
        setDocuments(response.documents);
      }
    } catch (error) {
      console.error('Error fetching documents:', error);
      toast.error('Failed to load documents');
    } finally {
      setLoading(false);
    }
  };

  const handleFileUpload = async (files: File[]) => {
    for (const file of files) {
      const fileId = `${file.name}-${Date.now()}`;
      setUploadProgress(prev => ({ ...prev, [fileId]: 0 }));

      try {
        // Simulate upload progress (since we don't have real progress from axios)
        const progressInterval = setInterval(() => {
          setUploadProgress(prev => {
            const current = prev[fileId] || 0;
            if (current < 90) {
              return { ...prev, [fileId]: current + 10 };
            }
            return prev;
          });
        }, 200);

        const response = await documentAPI.uploadDocument(file);
        
        clearInterval(progressInterval);
        setUploadProgress(prev => ({ ...prev, [fileId]: 100 }));

        if (response.success) {
          toast.success(`Successfully uploaded ${file.name}`);
          fetchDocuments(); // Refresh the document list
        } else {
          toast.error(`Failed to upload ${file.name}: ${response.message}`);
        }
      } catch (error) {
        console.error('Error uploading file:', error);
        toast.error(`Error uploading ${file.name}`);
      } finally {
        // Clean up progress after a delay
        setTimeout(() => {
          setUploadProgress(prev => {
            const newProgress = { ...prev };
            delete newProgress[fileId];
            return newProgress;
          });
        }, 2000);
      }
    }
  };

  const handleDeleteDocument = async (document: Document) => {
    try {
      const response = await documentAPI.deleteDocument(document.id);
      if (response.success) {
        toast.success('Document deleted successfully');
        fetchDocuments(); // Refresh the list
      } else {
        toast.error('Failed to delete document');
      }
    } catch (error) {
      console.error('Error deleting document:', error);
      toast.error('Error deleting document');
    }
    setDeleteDialog({ open: false });
  };

  const handleConfluenceUrlUpload = async () => {
    if (!confluenceUrl.trim()) {
      toast.error('Please enter a Confluence page URL');
      return;
    }

    // Check if we have saved Confluence credentials
    const savedConfig = localStorage.getItem('confluence_config');
    if (!savedConfig) {
      toast.error('Please configure Confluence credentials first in the Confluence section');
      return;
    }

    const parsedConfig = JSON.parse(savedConfig);
    if (!parsedConfig.url || !parsedConfig.token) {
      toast.error('Incomplete Confluence credentials. Please reconfigure in the Confluence section');
      return;
    }

    setUploadingFromUrl(true);
    try {
      const credentials = {
        url: parsedConfig.url,
        username: parsedConfig.username,
        api_token: parsedConfig.token,
        auth_type: parsedConfig.auth_type,
      };

      // Use the new import-as-document endpoint to create a permanent document
      const response = await confluenceAPI.importPageAsDocument(credentials, confluenceUrl);
      
      if (response.success) {
        toast.success(`Successfully imported Confluence page: ${response.title}`);
        setConfluenceUrl('');
        fetchDocuments(); // Refresh the document list
      } else {
        // Show more specific error messages
        let errorMessage = response.message || 'Failed to import Confluence page';
        if (errorMessage.includes('No space with key')) {
          errorMessage = `Space not found or no access. Please check the space key in the URL and ensure your Confluence credentials have access to this space.`;
        } else if (errorMessage.includes('not found')) {
          errorMessage = `Page not found. Please check the URL and ensure the page exists and is accessible.`;
        } else if (errorMessage.includes('unauthorized') || errorMessage.includes('authentication')) {
          errorMessage = `Authentication failed. Please check your Confluence credentials in the Confluence settings.`;
        }
        toast.error(errorMessage);
      }
    } catch (error) {
      console.error('Error importing Confluence page:', error);
      toast.error('Error importing Confluence page');
    } finally {
      setUploadingFromUrl(false);
    }
  };

  const onDrop = useCallback((acceptedFiles: File[]) => {
    handleFileUpload(acceptedFiles);
  }, []);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'application/pdf': ['.pdf'],
      'application/vnd.openxmlformats-officedocument.wordprocessingml.document': ['.docx'],
      'application/msword': ['.doc'],
      'text/plain': ['.txt'],
      'text/markdown': ['.md', '.markdown'],
      'text/html': ['.html', '.htm'],
    },
    multiple: true,
  });

  const getDocumentIcon = (type: string) => {
    switch (type.toLowerCase()) {
      case 'pdf':
        return <PictureAsPdf color="error" />;
      case 'docx':
        return <Description color="primary" />;
      case 'txt':
      case 'markdown':
        return <Article color="info" />;
      case 'html':
        return <Code color="secondary" />;
      default:
        return <Description />;
    }
  };

  const formatFileSize = (bytes?: number) => {
    if (!bytes) return 'Unknown size';
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(1024));
    return Math.round(bytes / Math.pow(1024, i) * 100) / 100 + ' ' + sizes[i];
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

  return (
    <Container maxWidth="xl" sx={{ py: 4 }}>
      <Box sx={{ mb: 6 }}>
        <Typography variant="h1" component="h1" sx={{ mb: 2 }}>
          Documents
        </Typography>
        <Typography variant="h6" color="text.secondary">
          Manage and upload your document collection
        </Typography>
      </Box>

      {/* Upload Section */}
      <Paper elevation={2} sx={{ p: 4, mb: 4 }}>
        <Typography variant="h4" component="h2" sx={{ mb: 3 }}>
          Upload Documents
        </Typography>
        
        {/* File Upload */}
        <Box
          {...getRootProps()}
          sx={{
            textAlign: 'center',
            py: 6,
            border: '2px dashed',
            borderColor: isDragActive ? 'primary.main' : 'grey.300',
            borderRadius: 2,
            backgroundColor: isDragActive ? 'primary.light' : 'transparent',
            cursor: 'pointer',
            transition: 'all 0.3s ease',
            '&:hover': {
              borderColor: 'primary.main',
              backgroundColor: 'primary.light',
            },
          }}
        >
          <input {...getInputProps()} />
          <CloudUpload sx={{ fontSize: 64, color: 'primary.main', mb: 2 }} />
          <Typography variant="h5" component="h3" sx={{ mb: 2 }}>
            {isDragActive ? 'Drop files here' : 'Upload Files'}
          </Typography>
          <Typography variant="body1" color="text.secondary" sx={{ mb: 4, maxWidth: 600, mx: 'auto' }}>
            Drag and drop files here, or click to browse. Supported formats: PDF, DOCX, TXT, MD, HTML
          </Typography>
          <Button variant="contained" size="large" startIcon={<Add />}>
            Choose Files
          </Button>
        </Box>

        <Divider sx={{ my: 4 }}>
          <Typography variant="body2" color="text.secondary">
            OR
          </Typography>
        </Divider>

        {/* Confluence URL Upload */}
        <Box sx={{ textAlign: 'center' }}>
          <Box sx={{ mb: 2 }}>
            <Link sx={{ fontSize: 48, color: 'primary.main' }} />
          </Box>
          <Typography variant="h5" component="h3" sx={{ mb: 2 }}>
            Import from Confluence
          </Typography>
          <Typography variant="body1" color="text.secondary" sx={{ mb: 2 }}>
            Import a Confluence page directly by providing its URL
          </Typography>
          <Typography variant="body2" color="text.secondary" sx={{ mb: 3, fontStyle: 'italic' }}>
            Note: Confluence credentials must be configured first in the Confluence section
          </Typography>
          
          <Box sx={{ display: 'flex', gap: 2, maxWidth: 800, mx: 'auto' }}>
            <TextField
              fullWidth
              label="Confluence Page URL"
              placeholder="https://wiki.autodesk.com/pages/viewpage.action?pageId=123456"
              value={confluenceUrl}
              onChange={(e) => setConfluenceUrl(e.target.value)}
              variant="outlined"
              disabled={uploadingFromUrl}
            />
            <Button
              variant="contained"
              onClick={handleConfluenceUrlUpload}
              disabled={!confluenceUrl.trim() || uploadingFromUrl}
              sx={{ minWidth: 120 }}
            >
              {uploadingFromUrl ? 'Importing...' : 'Import'}
            </Button>
          </Box>
        </Box>
      </Paper>

      {/* Upload Progress */}
      {Object.entries(uploadProgress).length > 0 && (
        <Box sx={{ mt: 4 }}>
          <Typography variant="h6" sx={{ mb: 2 }}>
            Uploading...
          </Typography>
          {Object.entries(uploadProgress).map(([fileId, progress]) => (
            <Box key={fileId} sx={{ mb: 2 }}>
              <Typography variant="body2" sx={{ mb: 1 }}>
                {fileId.split('-')[0]} - {progress}%
              </Typography>
              <LinearProgress variant="determinate" value={progress} />
            </Box>
          ))}
        </Box>
      )}

      {/* Document List */}
      <Paper elevation={2} sx={{ p: 4 }}>
        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
          <Typography variant="h4" component="h2">
            Your Documents ({documents.length})
          </Typography>
          <Button onClick={fetchDocuments} disabled={loading}>
            Refresh
          </Button>
        </Box>

        {loading ? (
          <Box sx={{ textAlign: 'center', py: 4 }}>
            <LinearProgress sx={{ mb: 2 }} />
            <Typography>Loading documents...</Typography>
          </Box>
        ) : documents.length === 0 ? (
          <Box sx={{ textAlign: 'center', py: 6 }}>
            <Description sx={{ fontSize: 48, color: 'text.secondary', mb: 2 }} />
            <Typography variant="h6" color="text.secondary" sx={{ mb: 1 }}>
              No documents uploaded yet
            </Typography>
            <Typography variant="body2" color="text.secondary">
              Upload your first document to get started
            </Typography>
          </Box>
        ) : (
          <List>
            {documents.map((document) => (
              <ListItem
                key={document.id}
                sx={{
                  border: '1px solid',
                  borderColor: 'divider',
                  borderRadius: 2,
                  mb: 1,
                  '&:hover': {
                    backgroundColor: 'action.hover',
                  },
                }}
              >
                <ListItemIcon>
                  {getDocumentIcon(document.type)}
                </ListItemIcon>
                <ListItemText
                  primary={
                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                      <Typography variant="subtitle1">{document.title}</Typography>
                      <Chip
                        label={document.status}
                        size="small"
                        clickable={false}
                        onClick={(e) => e.stopPropagation()}
                        color={
                          document.status === 'indexed' ? 'success' :
                          document.status === 'processing' ? 'warning' :
                          document.status === 'error' ? 'error' : 'default'
                        }
                      />
                    </Box>
                  }
                  secondary={
                    <Typography variant="body2" color="text.secondary" component="span">
                      Type: {document.type.toUpperCase()} • Size: {formatFileSize(document.size_bytes)} • 
                      Uploaded: {formatDate(document.created_at)}
                    </Typography>
                  }
                />
                <ListItemSecondaryAction>
                  <IconButton
                    edge="end"
                    color="error"
                    onClick={() => setDeleteDialog({ open: true, document })}
                    sx={{ mr: 1 }}
                  >
                    <Delete />
                  </IconButton>
                </ListItemSecondaryAction>
              </ListItem>
            ))}
          </List>
        )}
      </Paper>

      {/* Delete Confirmation Dialog */}
      <Dialog
        open={deleteDialog.open}
        onClose={() => setDeleteDialog({ open: false })}
      >
        <DialogTitle>Delete Document</DialogTitle>
        <DialogContent>
          <Typography>
            Are you sure you want to delete "{deleteDialog.document?.title}"? This action cannot be undone.
          </Typography>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setDeleteDialog({ open: false })}>
            Cancel
          </Button>
          <Button
            onClick={() => deleteDialog.document && handleDeleteDocument(deleteDialog.document)}
            color="error"
            variant="contained"
          >
            Delete
          </Button>
        </DialogActions>
      </Dialog>
    </Container>
  );
};

export default DocumentsPage; 