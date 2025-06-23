import React, { useState, useEffect, useRef } from 'react';
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
  List,
  ListItem,
  ListItemText,
  ListItemIcon,
  Chip,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  MenuItem,
  FormControlLabel,
  Switch,
  CircularProgress,
  Alert,
  IconButton,
  Divider,
  Avatar,
  ToggleButton,
  ToggleButtonGroup,
} from '@mui/material';
import {
  Send,
  ContentCopy,
  Publish,
  Preview,
  Edit,
  AutoAwesome,
  Chat,
  Settings,
  Clear,
  Search,
  Code,
  Article,
} from '@mui/icons-material';
import { confluenceAPI, documentAPI, templateAPI } from '../services/api';
import toast from 'react-hot-toast';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import rehypeRaw from 'rehype-raw';
import rehypeHighlight from 'rehype-highlight';

interface ChatMessage {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  timestamp: Date;
  isGenerating?: boolean;
}

interface ContentTemplate {
  id: string;
  name: string;
  description: string;
  sections: string[];
  template_type?: string;
}

interface Document {
  id: string;
  title: string;
  type: string;
}

interface GeneratedContent {
  title: string;
  content: string; // HTML version for publishing
  markdown?: string; // Markdown version for preview
  preview: string;
  page_id?: string;
  web_url?: string;
}

const ContentGenerationPage: React.FC = () => {
  // Chat state
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [currentMessage, setCurrentMessage] = useState('');
  const [isGenerating, setIsGenerating] = useState(false);
  
  // Content state
  const [generatedContent, setGeneratedContent] = useState<GeneratedContent | null>(null);
  const [selectedTemplate, setSelectedTemplate] = useState('documentation');
  const [templates, setTemplates] = useState<Record<string, ContentTemplate>>({});
  
  // Document context state
  const [availableDocuments, setAvailableDocuments] = useState<Document[]>([]);
  const [selectedDocuments, setSelectedDocuments] = useState<string[]>([]);
  const [useDocumentContext, setUseDocumentContext] = useState(true);
  const [useChatContext, setUseChatContext] = useState(true);
  
  // Settings state
  const [settingsDialog, setSettingsDialog] = useState(false);
  const [previewDialog, setPreviewDialog] = useState(false);
  const [enhancingContent, setEnhancingContent] = useState(false);
  const [loadingTemplates, setLoadingTemplates] = useState(false);
  const [documentSearchTerm, setDocumentSearchTerm] = useState('');
  const [previewMode, setPreviewMode] = useState<'markdown' | 'html'>('markdown');
  
  // Confluence settings (loaded from localStorage)
  const [confluenceConfig, setConfluenceConfig] = useState({
    url: '',
    token: '',
    space_key: '',
    auth_type: 'pat' as const,
  });
  const [publishSpaceKey, setPublishSpaceKey] = useState('');
  
  const messagesEndRef = useRef<HTMLDivElement>(null);
  
  useEffect(() => {
    loadTemplates();
    loadAvailableDocuments();
    loadConfluenceConfig();
    loadTemplatePreference();
  }, []);
  
  // No longer need to reload templates when Confluence config changes
  // Templates are now managed independently through the template API
  
  useEffect(() => {
    scrollToBottom();
  }, [messages]);
  
  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };
  
  const loadConfluenceConfig = () => {
    try {
      const savedConfig = localStorage.getItem('confluence_config');
      if (savedConfig) {
        const config = JSON.parse(savedConfig);
        setConfluenceConfig({
          url: config.url || '',
          token: config.token || '',
          space_key: config.space_key || '',
          auth_type: config.auth_type || 'pat',
        });
      }
      
      // Load publish space key preference
      const savedPublishSpaceKey = localStorage.getItem('publish_space_key');
      if (savedPublishSpaceKey) {
        setPublishSpaceKey(savedPublishSpaceKey);
      } else if (savedConfig) {
        // Default to main config space key if no separate publish space key is set
        const config = JSON.parse(savedConfig);
        setPublishSpaceKey(config.space_key || '');
      }
    } catch (error) {
      console.error('Error loading Confluence config:', error);
    }
  };
  
  const loadTemplatePreference = () => {
    try {
      const savedTemplate = localStorage.getItem('preferred_template');
      if (savedTemplate) {
        setSelectedTemplate(savedTemplate);

      }
    } catch (error) {
      console.error('Error loading template preference:', error);
    }
  };
  
  const saveTemplatePreference = (templateKey: string) => {
    try {
      localStorage.setItem('preferred_template', templateKey);

    } catch (error) {
      console.error('Error saving template preference:', error);
    }
  };
  
  const savePublishSpaceKeyPreference = (spaceKey: string) => {
    try {
      localStorage.setItem('publish_space_key', spaceKey);

    } catch (error) {
      console.error('Error saving publish space key preference:', error);
    }
  };
  
  const loadTemplates = async () => {
    setLoadingTemplates(true);
    try {
      // Load both built-in and custom templates from the backend
      const [builtinResult, customResult] = await Promise.all([
        templateAPI.getBuiltinTemplates(),
        templateAPI.listTemplates()
      ]);
      
      let allTemplates: Record<string, ContentTemplate> = {};
      let builtinCount = 0;
      let customCount = 0;
      
      // Add built-in templates
      if (builtinResult.success) {
        builtinResult.templates.forEach(template => {
          allTemplates[template.id] = {
            id: template.id,
            name: template.name,
            description: template.description,
            sections: template.sections,
            template_type: 'builtin'
          };
        });
        builtinCount = builtinResult.templates.length;
      }
      
      // Add custom templates
      if (customResult.success) {
        customResult.templates.forEach(template => {
          allTemplates[template.id] = {
            id: template.id,
            name: template.name,
            description: template.description,
            sections: template.sections,
            template_type: template.template_type
          };
        });
        customCount = customResult.templates.length;
      }
      
      setTemplates(allTemplates);
      
      // Validate and restore saved template preference
      const savedTemplate = localStorage.getItem('preferred_template');
      if (savedTemplate && allTemplates[savedTemplate]) {
        setSelectedTemplate(savedTemplate);
      } else if (savedTemplate && !allTemplates[savedTemplate]) {
        // Clear invalid preference
        localStorage.removeItem('preferred_template');
        // Default to documentation template if available
        if (allTemplates['documentation']) {
          setSelectedTemplate('documentation');
        }
      }
      
      // Templates loaded successfully (no toast needed)
    } catch (error) {
      console.error('Error loading templates:', error);
      toast.error('Failed to load content templates');
    } finally {
      setLoadingTemplates(false);
    }
  };
  
  const loadAvailableDocuments = async () => {
    try {
      const response = await documentAPI.getDocuments();
      if (response.success && response.documents) {
        const docs = response.documents.map((doc: any) => ({
          id: doc.id,
          title: doc.title,
          type: doc.type || 'unknown'
        }));
        setAvailableDocuments(docs);
      }
    } catch (error) {
      console.error('Error loading documents:', error);
    }
  };
  
  const handleSendMessage = async () => {
    if (!currentMessage.trim() || isGenerating) return;
    
    const userMessage: ChatMessage = {
      id: Date.now().toString(),
      role: 'user',
      content: currentMessage,
      timestamp: new Date(),
    };
    
    const assistantMessage: ChatMessage = {
      id: (Date.now() + 1).toString(),
      role: 'assistant',
      content: '',
      timestamp: new Date(),
      isGenerating: true,
    };
    
    setMessages(prev => [...prev, userMessage, assistantMessage]);
    setCurrentMessage('');
    setIsGenerating(true);
    
    try {
      // Prepare document IDs for generation
      let documentIdsToUse;
      if (!useDocumentContext) {
        // Don't use any documents
        documentIdsToUse = [];
      } else if (selectedDocuments.length === 0) {
        // Use all documents (send null/undefined)
        documentIdsToUse = null;
      } else {
        // Use specific selected documents
        documentIdsToUse = selectedDocuments;
      }
      
      
      // Generate content using the chat approach
      const result = await confluenceAPI.generateContent({
        credentials: {
          url: confluenceConfig.url,
          username: '',
          api_token: confluenceConfig.token,
          auth_type: confluenceConfig.auth_type,
        },
        content_type: selectedTemplate,
        topic: currentMessage,
        document_ids: documentIdsToUse,
        space_key: publishSpaceKey || confluenceConfig.space_key, // Use publish space key if set, fallback to config space key
        publish_directly: false,
        additional_context: useChatContext ? messages.map(m => `${m.role}: ${m.content}`).join('\n') : undefined,
      });
      
      if (result.success) {
        // Convert HTML to approximate markdown for preview
        // This is a simple conversion - in a real app you might want proper HTML-to-markdown conversion
        const htmlToMarkdown = (html: string) => {
          return html
            .replace(/<h1[^>]*>(.*?)<\/h1>/gi, '# $1\n')
            .replace(/<h2[^>]*>(.*?)<\/h2>/gi, '## $1\n')
            .replace(/<h3[^>]*>(.*?)<\/h3>/gi, '### $1\n')
            .replace(/<h4[^>]*>(.*?)<\/h4>/gi, '#### $1\n')
            .replace(/<p[^>]*>(.*?)<\/p>/gi, '$1\n\n')
            .replace(/<strong[^>]*>(.*?)<\/strong>/gi, '**$1**')
            .replace(/<b[^>]*>(.*?)<\/b>/gi, '**$1**')
            .replace(/<em[^>]*>(.*?)<\/em>/gi, '*$1*')
            .replace(/<i[^>]*>(.*?)<\/i>/gi, '*$1*')
            .replace(/<ul[^>]*>(.*?)<\/ul>/gis, '$1')
            .replace(/<ol[^>]*>(.*?)<\/ol>/gis, '$1')
            .replace(/<li[^>]*>(.*?)<\/li>/gi, '- $1\n')
            .replace(/<br\s*\/?>/gi, '\n')
            .replace(/<[^>]*>/g, '') // Remove remaining HTML tags
            .replace(/\n\s*\n\s*\n/g, '\n\n') // Clean up extra newlines
            .trim();
        };
        
        const contentWithMarkdown = {
          ...result,
          markdown: htmlToMarkdown(result.content)
        };
        
        setGeneratedContent(contentWithMarkdown);
        
        // Update the assistant message
        setMessages(prev => prev.map(msg => 
          msg.id === assistantMessage.id 
            ? { ...msg, content: `Generated content: "${result.title}"`, isGenerating: false }
            : msg
        ));
      } else {
        setMessages(prev => prev.map(msg => 
          msg.id === assistantMessage.id 
            ? { ...msg, content: `Error: ${result.message}`, isGenerating: false }
            : msg
        ));
        toast.error(result.message);
      }
    } catch (error) {
      setMessages(prev => prev.map(msg => 
        msg.id === assistantMessage.id 
          ? { ...msg, content: 'Error generating content', isGenerating: false }
          : msg
      ));
      toast.error('Failed to generate content');
    } finally {
      setIsGenerating(false);
    }
  };
  
  const handleEnhanceContent = async () => {
    if (!generatedContent) return;
    
    setEnhancingContent(true);
    try {
      const result = await confluenceAPI.enhanceContent(generatedContent.content, 'general');
      if (result.success) {
        setGeneratedContent(prev => prev ? {
          ...prev,
          content: result.enhanced_content,
          preview: result.enhanced_content.replace(/<[^>]*>/g, '').substring(0, 500) + '...'
        } : null);
      } else {
        toast.error('Failed to enhance content');
      }
    } catch (error) {
      toast.error('Failed to enhance content');
    } finally {
      setEnhancingContent(false);
    }
  };
  
    const handleCopyContent = () => {
    if (generatedContent) {
      // Copy based on current preview mode in full preview, or HTML by default for main preview
      const contentToCopy = previewMode === 'markdown' && generatedContent.markdown 
        ? generatedContent.markdown 
        : generatedContent.content;
      
      navigator.clipboard.writeText(contentToCopy);
    }
  };

  const handlePublishContent = async () => {
    if (!generatedContent) return;
    
    // Validate Confluence configuration
    if (!confluenceConfig.url || !confluenceConfig.token) {
      toast.error('Please configure Confluence settings first');
      return;
    }
    
    const targetSpaceKey = publishSpaceKey || confluenceConfig.space_key;
    if (!targetSpaceKey) {
      toast.error('Please specify a Confluence space key for publishing');
      return;
    }
    
    try {
      setIsGenerating(true);
      
      // Publish the existing content directly without regenerating
      const result = await confluenceAPI.publishContent({
        credentials: {
          url: confluenceConfig.url,
          username: '',
          api_token: confluenceConfig.token,
          auth_type: confluenceConfig.auth_type,
        },
        space_key: targetSpaceKey,
        title: generatedContent.title,
        content: generatedContent.content, // Use the existing generated content
      });
      
      if (result.success && result.page_id) {
        // Update the generated content with published page info
        setGeneratedContent(prev => prev ? {
          ...prev,
          page_id: result.page_id,
          web_url: result.web_url
        } : null);
        
        // Add success message to chat
        const action = result.action || 'published';
        const actionText = action === 'updated' ? 'updated' : 'created';
        const successMessage: ChatMessage = {
          id: Date.now().toString(),
          role: 'assistant',
          content: `✅ Content ${actionText} successfully in Confluence!\n\n**Page ID:** ${result.page_id}\n**Space:** ${targetSpaceKey}\n**URL:** ${result.web_url || 'Check your Confluence space'}\n**Action:** ${result.message}`,
          timestamp: new Date(),
        };
        setMessages(prev => [...prev, successMessage]);
      } else {
        throw new Error(result.message || 'Publishing failed');
      }
      
    } catch (error: any) {
      console.error('Error publishing content:', error);
      
      // Add error message to chat
      const errorMessage: ChatMessage = {
        id: Date.now().toString(),
        role: 'assistant',
        content: `❌ Failed to publish content: ${error.message || 'Unknown error'}\n\nPlease check your Confluence settings and try again.`,
        timestamp: new Date(),
      };
      setMessages(prev => [...prev, errorMessage]);
      
      toast.error(`Failed to publish: ${error.message || 'Unknown error'}`);
    } finally {
      setIsGenerating(false);
    }
  };

  const handleClearChat = () => {
    setMessages([]);
    setGeneratedContent(null);
  };
  
  const getFilteredDocuments = () => {
    if (!documentSearchTerm.trim()) {
      return availableDocuments;
    }
    
    const searchLower = documentSearchTerm.toLowerCase();
    return availableDocuments.filter(doc => 
      doc.title.toLowerCase().includes(searchLower) ||
      doc.type.toLowerCase().includes(searchLower)
    );
  };
  
  const renderMessage = (message: ChatMessage) => (
    <Box key={message.id} sx={{ mb: 2, display: 'flex', alignItems: 'flex-start' }}>
      <Avatar sx={{ mr: 1, bgcolor: message.role === 'user' ? 'primary.main' : 'secondary.main' }}>
        {message.role === 'user' ? <Chat /> : <AutoAwesome />}
      </Avatar>
      <Paper sx={{ p: 2, maxWidth: '80%', flexGrow: 1 }}>
        <Typography variant="body2" color="text.secondary" sx={{ fontSize: '0.75rem', mb: 0.5 }}>
          {message.role === 'user' ? 'You' : 'Assistant'} • {message.timestamp.toLocaleTimeString()}
        </Typography>
        <Typography variant="body1">
          {message.isGenerating ? (
            <Box sx={{ display: 'flex', alignItems: 'center' }}>
              <CircularProgress size={16} sx={{ mr: 1 }} />
              Generating content...
            </Box>
          ) : (
            message.content
          )}
        </Typography>
      </Paper>
    </Box>
  );
  
  return (
    <Container maxWidth="xl" sx={{ py: 4 }}>
      <Typography variant="h3" component="h1" gutterBottom>
        Content Generation
      </Typography>
      <Typography variant="body1" color="text.secondary" sx={{ mb: 4 }}>
        AI-powered content generation with chat interface and real-time preview
      </Typography>
      
      <Grid container spacing={3} sx={{ height: 'calc(100vh - 200px)' }}>
        {/* Left Panel - Chat Interface */}
        <Grid size={{ xs: 12, md: 6 }}>
          <Card sx={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
            <CardContent sx={{ flexGrow: 1, display: 'flex', flexDirection: 'column' }}>
              <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
                <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, flexWrap: 'wrap' }}>
                  <Typography variant="h6">Chat & Generate</Typography>
                  {selectedTemplate && templates[selectedTemplate] && (
                    <Chip 
                      label={templates[selectedTemplate].name} 
                      size="small" 
                      variant="outlined"
                      color="primary"
                    />
                  )}
                  {useDocumentContext && selectedDocuments.length > 0 && (
                    <Chip 
                      label={`${selectedDocuments.length} docs`}
                      size="small" 
                      variant="outlined"
                      color="secondary"
                    />
                  )}
                  {!useDocumentContext && (
                    <Chip 
                      label="No docs"
                      size="small" 
                      variant="outlined"
                      color="default"
                    />
                  )}
                </Box>
                <Box>
                  <IconButton onClick={() => setSettingsDialog(true)} size="small">
                    <Settings />
                  </IconButton>
                  <IconButton onClick={handleClearChat} size="small">
                    <Clear />
                  </IconButton>
                </Box>
              </Box>
              
              {/* Messages */}
              <Box sx={{ flexGrow: 1, overflowY: 'auto', mb: 2, pr: 1 }}>
                {messages.length === 0 ? (
                  <Alert severity="info">
                    Start a conversation to generate content. Ask for documentation, meeting notes, or any other content type.
                  </Alert>
                ) : (
                  messages.map(renderMessage)
                )}
                <div ref={messagesEndRef} />
              </Box>
              
              {/* Input */}
              <Box sx={{ display: 'flex', gap: 1 }}>
                <Box sx={{ flexGrow: 1 }}>
                  <TextField
                    fullWidth
                    placeholder="What content would you like to generate?"
                    value={currentMessage}
                    onChange={(e) => setCurrentMessage(e.target.value)}
                    onKeyPress={(e) => e.key === 'Enter' && !e.shiftKey && handleSendMessage()}
                    multiline
                    maxRows={3}
                    disabled={isGenerating}
                  />
                  <Typography variant="caption" color="text.secondary" sx={{ mt: 0.5, display: 'block' }}>
                    {useDocumentContext 
                      ? selectedDocuments.length > 0 
                        ? `Using ${selectedDocuments.length} document(s) for context`
                        : 'Using all documents for context'
                      : 'Document context disabled - using AI knowledge only'
                    }
                  </Typography>
                </Box>
                <Button
                  variant="contained"
                  onClick={handleSendMessage}
                  disabled={!currentMessage.trim() || isGenerating}
                  sx={{ minWidth: 'auto', px: 2 }}
                >
                  <Send />
                </Button>
              </Box>
            </CardContent>
          </Card>
        </Grid>
        
        {/* Right Panel - Generated Content */}
        <Grid size={{ xs: 12, md: 6 }}>
          <Card sx={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
            <CardContent sx={{ flexGrow: 1, display: 'flex', flexDirection: 'column' }}>
              <Typography variant="h6" gutterBottom>Generated Content</Typography>
              
              {generatedContent ? (
                <>
                  <Typography variant="h6" gutterBottom>
                    {generatedContent.title}
                  </Typography>
                  
                  <Box sx={{ 
                    flexGrow: 1,
                    p: 3, 
                    bgcolor: 'white', 
                    borderRadius: 1, 
                    mb: 2,
                    overflow: 'auto',
                    border: '1px solid',
                    borderColor: 'grey.300',
                    textAlign: 'left',
                    '& h1': {
                      fontSize: '1.75rem',
                      fontWeight: 600,
                      color: 'text.primary',
                      marginBottom: '1rem',
                      marginTop: '1.5rem',
                      lineHeight: 1.2,
                      textAlign: 'left',
                      '&:first-of-type': { marginTop: 0 }
                    },
                    '& h2': {
                      fontSize: '1.5rem',
                      fontWeight: 600,
                      color: 'text.primary',
                      marginBottom: '0.75rem',
                      marginTop: '1.25rem',
                      lineHeight: 1.3,
                      textAlign: 'left'
                    },
                    '& h3': {
                      fontSize: '1.25rem',
                      fontWeight: 600,
                      color: 'text.primary',
                      marginBottom: '0.5rem',
                      marginTop: '1rem',
                      lineHeight: 1.4,
                      textAlign: 'left'
                    },
                    '& p': {
                      fontSize: '1rem',
                      lineHeight: 1.6,
                      color: 'text.primary',
                      marginBottom: '1rem',
                      textAlign: 'left'
                    },
                    '& ul, & ol': {
                      marginBottom: '1rem',
                      paddingLeft: '1.5rem'
                    },
                    '& li': {
                      fontSize: '1rem',
                      lineHeight: 1.6,
                      color: 'text.primary',
                      marginBottom: '0.25rem'
                    },
                    '& strong': {
                      fontWeight: 600
                    },
                    '& code': {
                      backgroundColor: 'grey.100',
                      padding: '0.2rem 0.4rem',
                      borderRadius: '3px',
                      fontFamily: 'Consolas, Monaco, monospace',
                      fontSize: '0.9rem'
                    },
                    '& pre': {
                      backgroundColor: 'grey.100',
                      padding: '1rem',
                      borderRadius: '4px',
                      overflow: 'auto',
                      marginBottom: '1rem',
                      fontFamily: 'Consolas, Monaco, monospace',
                      fontSize: '0.9rem'
                    },
                    '& blockquote': {
                      borderLeft: '4px solid',
                      borderColor: 'primary.main',
                      paddingLeft: '1rem',
                      marginLeft: 0,
                      marginBottom: '1rem',
                      fontStyle: 'italic',
                      color: 'text.secondary'
                    },
                    '& table': {
                      width: '100%',
                      borderCollapse: 'collapse',
                      marginBottom: '1rem'
                    },
                    '& th, & td': {
                      border: '1px solid',
                      borderColor: 'grey.300',
                      padding: '0.75rem',
                      textAlign: 'left'
                    },
                    '& th': {
                      backgroundColor: 'grey.100',
                      fontWeight: 600
                    }
                  }}>
                    <ReactMarkdown 
                      remarkPlugins={[remarkGfm]}
                      rehypePlugins={[rehypeRaw]}
                    >
                      {generatedContent.markdown || generatedContent.content}
                    </ReactMarkdown>
                  </Box>
                  
                  <Box sx={{ display: 'flex', gap: 1, flexWrap: 'wrap' }}>
                    <Button
                      variant="outlined"
                      size="small"
                      onClick={() => setPreviewDialog(true)}
                      startIcon={<Preview />}
                    >
                      Full Preview
                    </Button>
                    
                    <Button
                      variant="outlined"
                      size="small"
                      onClick={handleCopyContent}
                      startIcon={<ContentCopy />}
                    >
                      Copy
                    </Button>

                    <Button
                      variant="outlined"
                      size="small"
                      onClick={handleEnhanceContent}
                      disabled={enhancingContent}
                      startIcon={enhancingContent ? <CircularProgress size={16} /> : <Edit />}
                    >
                      Enhance
                    </Button>

                    <Button
                      variant="contained"
                      size="small"
                      startIcon={isGenerating ? <CircularProgress size={16} /> : <Publish />}
                      color="success"
                      title={`Publish to ${publishSpaceKey || confluenceConfig.space_key || 'Confluence'} space`}
                      onClick={handlePublishContent}
                      disabled={isGenerating}
                    >
                      {isGenerating ? 'Publishing...' : `Publish to ${publishSpaceKey || confluenceConfig.space_key || 'Confluence'}`}
                    </Button>
                  </Box>
                </>
              ) : (
                <Alert severity="info">
                  Generated content will appear here once you start chatting and generating content.
                </Alert>
              )}
            </CardContent>
          </Card>
        </Grid>
      </Grid>
      
      {/* Settings Dialog */}
      <Dialog 
        open={settingsDialog} 
        onClose={() => {
          setSettingsDialog(false);
          setDocumentSearchTerm(''); // Reset search when closing
        }} 
        maxWidth="md" 
        fullWidth
      >
        <DialogTitle>Content Generation Settings</DialogTitle>
        <DialogContent>
          <Box sx={{ mb: 3 }}>
            <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', mb: 1 }}>
              <Typography variant="subtitle2">Template</Typography>
              <Button
                size="small"
                onClick={loadTemplates}
                disabled={loadingTemplates}
                startIcon={loadingTemplates ? <CircularProgress size={16} /> : <AutoAwesome />}
                variant="outlined"
              >
                {loadingTemplates ? 'Loading...' : 'Refresh Templates'}
              </Button>
            </Box>
            <TextField
              fullWidth
              select
              value={selectedTemplate}
              onChange={(e) => {
                const newTemplate = e.target.value;
                setSelectedTemplate(newTemplate);
                saveTemplatePreference(newTemplate);
              }}
            >
              {Object.entries(templates).map(([key, template]) => (
                <MenuItem key={key} value={key}>
                  <Box sx={{ width: '100%' }}>
                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 0.5 }}>
                      <Typography>{template.name}</Typography>
                      {template.template_type && (
                        <Chip 
                          label={template.template_type} 
                          size="small" 
                          color={template.template_type === 'builtin' ? 'default' : 'primary'}
                          variant="outlined"
                        />
                      )}
                    </Box>
                    {template.description && (
                      <Typography variant="caption" color="text.secondary" sx={{ display: 'block' }}>
                        {template.description}
                      </Typography>
                    )}
                  </Box>
                </MenuItem>
              ))}
            </TextField>
            <Typography variant="body2" color="text.secondary" sx={{ mt: 1 }}>
              To add custom templates, go to the Confluence → Templates tab and import from Confluence pages.
              {selectedTemplate && (
                <><br />Current template will be saved as your preference.</>
              )}
            </Typography>
          </Box>
          
          <Box sx={{ mb: 3 }}>
            <FormControlLabel
              control={
                <Switch
                  checked={useDocumentContext}
                  onChange={(e) => setUseDocumentContext(e.target.checked)}
                />
              }
              label="Use Document Context"
            />
            <Typography variant="body2" color="text.secondary">
              Include selected documents as context for content generation
            </Typography>
          </Box>
          
          {useDocumentContext && (
            <Box sx={{ mb: 3 }}>
              <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', mb: 1 }}>
                <Typography variant="subtitle2">Selected Documents</Typography>
                <Typography variant="caption" color="text.secondary">
                  {getFilteredDocuments().length} of {availableDocuments.length} documents
                </Typography>
              </Box>
              
              {/* Search Bar */}
              <TextField
                fullWidth
                size="small"
                placeholder="Search documents by title or type..."
                value={documentSearchTerm}
                onChange={(e) => setDocumentSearchTerm(e.target.value)}
                InputProps={{
                  startAdornment: <Search sx={{ color: 'text.secondary', mr: 1 }} />,
                  endAdornment: documentSearchTerm && (
                    <IconButton
                      size="small"
                      onClick={() => setDocumentSearchTerm('')}
                      sx={{ p: 0.5 }}
                    >
                      <Clear fontSize="small" />
                    </IconButton>
                  )
                }}
                sx={{ mb: 1 }}
              />
              
              <Box sx={{ 
                maxHeight: 200, 
                overflow: 'auto',
                border: '1px solid',
                borderColor: 'grey.300',
                borderRadius: 1,
                bgcolor: 'grey.50'
              }}>
                {(() => {
                  const filteredDocuments = getFilteredDocuments();
                  
                  if (availableDocuments.length === 0) {
                    return (
                      <Box sx={{ p: 2, textAlign: 'center' }}>
                        <Typography variant="body2" color="text.secondary">
                          No documents available
                        </Typography>
                      </Box>
                    );
                  }
                  
                  if (filteredDocuments.length === 0) {
                    return (
                      <Box sx={{ p: 2, textAlign: 'center' }}>
                        <Typography variant="body2" color="text.secondary">
                          No documents match "{documentSearchTerm}"
                        </Typography>
                      </Box>
                    );
                  }
                  
                  return filteredDocuments.map((doc) => (
                    <Box 
                      key={doc.id} 
                      sx={{ 
                        display: 'flex', 
                        alignItems: 'center', 
                        p: 1.5,
                        borderBottom: '1px solid',
                        borderColor: 'grey.200',
                        '&:last-child': {
                          borderBottom: 'none'
                        }
                      }}
                    >
                      <Switch
                        checked={selectedDocuments.includes(doc.id)}
                        onChange={(e) => {
                          if (e.target.checked) {
                            setSelectedDocuments(prev => [...prev, doc.id]);
                          } else {
                            setSelectedDocuments(prev => prev.filter(id => id !== doc.id));
                          }
                        }}
                        size="small"
                      />
                      <Box sx={{ ml: 2, flexGrow: 1 }}>
                        <Typography variant="body2" sx={{ fontWeight: 500 }}>
                          {doc.title}
                        </Typography>
                        <Box sx={{ display: 'flex', alignItems: 'center', mt: 0.5 }}>
                          <Chip label={doc.type} size="small" variant="outlined" />
                        </Box>
                      </Box>
                    </Box>
                  ));
                })()}
              </Box>
            </Box>
          )}
          
          <Box sx={{ mb: 3 }}>
            <FormControlLabel
              control={
                <Switch
                  checked={useChatContext}
                  onChange={(e) => setUseChatContext(e.target.checked)}
                />
              }
              label="Use Chat Context"
            />
            <Typography variant="body2" color="text.secondary">
              Include previous chat messages as context for content generation
            </Typography>
          </Box>
          
          <Box sx={{ mb: 3 }}>
            <Typography variant="subtitle2" gutterBottom>Publishing Settings</Typography>
            <TextField
              fullWidth
              label="Confluence Space Key for Publishing"
              placeholder="Enter space key (e.g., MYSPACE)"
              value={publishSpaceKey}
              onChange={(e) => {
                const newSpaceKey = e.target.value;
                setPublishSpaceKey(newSpaceKey);
                savePublishSpaceKeyPreference(newSpaceKey);
              }}
              size="small"
              sx={{ mb: 1 }}
            />
            <Typography variant="body2" color="text.secondary">
              Specify which Confluence space to publish content to. This can be different from your main configuration space.
            </Typography>
          </Box>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => {
            setSettingsDialog(false);
            setDocumentSearchTerm(''); // Reset search when closing
          }}>
            Close
          </Button>
        </DialogActions>
      </Dialog>
      
      {/* Preview Dialog */}
      <Dialog 
        open={previewDialog} 
        onClose={() => setPreviewDialog(false)}
        maxWidth="md"
        fullWidth
      >
        <DialogTitle>
          <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <Box>
              Content Preview
              {generatedContent?.title && (
                <Typography variant="subtitle2" color="text.secondary">
                  {generatedContent.title}
                </Typography>
              )}
            </Box>
            <ToggleButtonGroup
              value={previewMode}
              exclusive
              onChange={(e, newMode) => newMode && setPreviewMode(newMode)}
              size="small"
            >
              <ToggleButton value="markdown" aria-label="markdown view">
                <Article />
              </ToggleButton>
              <ToggleButton value="html" aria-label="html view">
                <Code />
              </ToggleButton>
            </ToggleButtonGroup>
          </Box>
        </DialogTitle>
        <DialogContent>
          {previewMode === 'markdown' ? (
            <Box sx={{ 
              p: 3, 
              bgcolor: 'white', 
              borderRadius: 1,
              maxHeight: 500,
              overflow: 'auto',
              border: '1px solid',
              borderColor: 'grey.300',
              '& h1': {
                fontSize: '1.75rem',
                fontWeight: 600,
                color: 'text.primary',
                marginBottom: '1rem',
                marginTop: '1.5rem',
                lineHeight: 1.2,
                '&:first-of-type': { marginTop: 0 }
              },
              '& h2': {
                fontSize: '1.5rem',
                fontWeight: 600,
                color: 'text.primary',
                marginBottom: '0.75rem',
                marginTop: '1.25rem',
                lineHeight: 1.3
              },
              '& h3': {
                fontSize: '1.25rem',
                fontWeight: 600,
                color: 'text.primary',
                marginBottom: '0.5rem',
                marginTop: '1rem',
                lineHeight: 1.4
              },
              '& p': {
                fontSize: '1rem',
                lineHeight: 1.6,
                color: 'text.primary',
                marginBottom: '1rem',
                textAlign: 'left'
              },
              '& ul, & ol': {
                marginBottom: '1rem',
                paddingLeft: '1.5rem'
              },
              '& li': {
                fontSize: '1rem',
                lineHeight: 1.6,
                color: 'text.primary',
                marginBottom: '0.25rem'
              },
              '& strong': {
                fontWeight: 600
              },
              '& code': {
                backgroundColor: 'grey.100',
                padding: '0.2rem 0.4rem',
                borderRadius: '3px',
                fontFamily: 'Consolas, Monaco, monospace',
                fontSize: '0.9rem'
              },
              '& pre': {
                backgroundColor: 'grey.100',
                padding: '1rem',
                borderRadius: '4px',
                overflow: 'auto',
                marginBottom: '1rem',
                fontFamily: 'Consolas, Monaco, monospace',
                fontSize: '0.9rem'
              },
              '& blockquote': {
                borderLeft: '4px solid',
                borderColor: 'primary.main',
                paddingLeft: '1rem',
                marginLeft: 0,
                marginBottom: '1rem',
                fontStyle: 'italic',
                color: 'text.secondary'
              },
              '& table': {
                width: '100%',
                borderCollapse: 'collapse',
                marginBottom: '1rem'
              },
              '& th, & td': {
                border: '1px solid',
                borderColor: 'grey.300',
                padding: '0.75rem',
                textAlign: 'left'
              },
              '& th': {
                backgroundColor: 'grey.100',
                fontWeight: 600
              }
            }}>
              {generatedContent?.markdown ? (
                <ReactMarkdown 
                  remarkPlugins={[remarkGfm]}
                  rehypePlugins={[rehypeRaw]}
                >
                  {generatedContent.markdown}
                </ReactMarkdown>
              ) : (
                <Typography color="text.secondary">No markdown content available</Typography>
              )}
            </Box>
          ) : (
            <Box sx={{ 
              p: 2, 
              bgcolor: 'grey.50', 
              borderRadius: 1,
              maxHeight: 500,
              overflow: 'auto',
              border: '1px solid',
              borderColor: 'grey.300'
            }}>
              {generatedContent?.content ? (
                <Box
                  component="pre"
                  sx={{
                    fontFamily: 'Consolas, Monaco, "Courier New", monospace',
                    fontSize: '0.875rem',
                    lineHeight: 1.5,
                    margin: 0,
                    whiteSpace: 'pre-wrap',
                    wordBreak: 'break-word',
                    color: 'text.primary'
                  }}
                >
                  {generatedContent.content}
                </Box>
              ) : (
                <Typography color="text.secondary">No HTML content available</Typography>
              )}
            </Box>
          )}
        </DialogContent>
        <DialogActions>
          <Box sx={{ flexGrow: 1 }}>
            <Typography variant="caption" color="text.secondary">
              Note: Publishing to Confluence always uses the HTML version
            </Typography>
          </Box>
          <Button onClick={handleCopyContent} startIcon={<ContentCopy />}>
            Copy {previewMode === 'markdown' ? 'Markdown' : 'HTML'}
          </Button>
          <Button onClick={() => setPreviewDialog(false)}>
            Close
          </Button>
        </DialogActions>
      </Dialog>
    </Container>
  );
};

export default ContentGenerationPage; 