import React, { useState, useEffect, useRef } from 'react';
import {
  Container,
  Typography,
  Paper,
  Box,
  TextField,
  Button,
  Avatar,
  Chip,
  List,
  ListItem,
  ListItemAvatar,
  ListItemText,
  CircularProgress,
  Alert,
  Switch,
  FormControlLabel,
  Tooltip,
} from '@mui/material';
import {
  Send,
  Chat,
  SmartToy,
  Person,
  Error,
  History,
  Add,
  ChatBubble,
  Notifications,
  Cancel,
} from '@mui/icons-material';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import rehypeHighlight from 'rehype-highlight';
import rehypeRaw from 'rehype-raw';
import 'highlight.js/styles/github.css';
import { chatAPI, documentAPI } from '../services/api';
import type { Document } from '../services/api';
import { chatStorage, type ChatMessage } from '../utils/chatStorage';
import ChatHistorySidebar from '../components/ChatHistorySidebar';
import { useBackgroundChat } from '../hooks/useBackgroundChat';
import toast from 'react-hot-toast';

const ChatPage: React.FC = () => {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [inputMessage, setInputMessage] = useState('');
  const [loading, setLoading] = useState(false);
  const [documents, setDocuments] = useState<Document[]>([]);
  const [currentSessionId, setCurrentSessionId] = useState<string>('');
  const [loadingHistory, setLoadingHistory] = useState(true);
  const [historySidebarOpen, setHistorySidebarOpen] = useState(false);
  const [useBackgroundProcessing, setUseBackgroundProcessing] = useState(() => {
    const saved = localStorage.getItem('useBackgroundProcessing');
    return saved ? JSON.parse(saved) : false;
  });
  const messagesEndRef = useRef<HTMLDivElement>(null);

  // Background chat hook
  const backgroundChat = useBackgroundChat(currentSessionId);

  useEffect(() => {
    fetchDocuments();
    loadChatHistory();
  }, []);

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  // Save messages to IndexedDB whenever messages change (but not when loading from history)
  useEffect(() => {
    if (currentSessionId && messages.length > 0 && !loadingHistory) {
      // Add a small delay to avoid race conditions when switching sessions
      const timeoutId = setTimeout(() => {
        saveChatHistory();
      }, 100);
      
      return () => clearTimeout(timeoutId);
    }
  }, [messages, currentSessionId, loadingHistory]);

  // Save background processing preference
  useEffect(() => {
    localStorage.setItem('useBackgroundProcessing', JSON.stringify(useBackgroundProcessing));
  }, [useBackgroundProcessing]);

  // Listen for background chat updates via custom events
  useEffect(() => {
    const handleBackgroundChatUpdate = async (event: Event) => {
      const customEvent = event as CustomEvent;
      const { sessionId, type } = customEvent.detail;
      
      if (sessionId === currentSessionId) {
        try {
          const savedMessages = await chatStorage.loadMessages(currentSessionId);
          setMessages(savedMessages);
          
          // Reset the background chat state
          backgroundChat.reset();
        } catch (error) {
          console.error('Error reloading messages after background update:', error);
        }
      }
    };

    // Add event listener
    window.addEventListener('backgroundChatUpdated', handleBackgroundChatUpdate);

    // Cleanup
    return () => {
      window.removeEventListener('backgroundChatUpdated', handleBackgroundChatUpdate);
    };
  }, [currentSessionId]);

  const fetchDocuments = async () => {
    try {
      const response = await documentAPI.getDocuments();
      if (response.success && response.documents) {
        setDocuments(response.documents);
      }
    } catch (error) {
      console.error('Error fetching documents:', error);
    }
  };

  const loadChatHistory = async () => {
    try {
      setLoadingHistory(true);
      
      // First, try to get the most recent session
      const allSessions = await chatStorage.getAllSessions();
      
      let sessionId: string;
      if (allSessions.length > 0) {
        // Use the most recent session (sessions are sorted by updatedAt desc)
        sessionId = allSessions[0].id;

      } else {
        // No sessions exist, create a new one
        sessionId = chatStorage.createNewSessionId();

      }
      
      setCurrentSessionId(sessionId);
      const savedMessages = await chatStorage.loadMessages(sessionId);
      setMessages(savedMessages);
    } catch (error) {
      console.error('Error loading chat history:', error);
      toast.error('Failed to load chat history');
    } finally {
      setLoadingHistory(false);
    }
  };

  const saveChatHistory = async () => {
    try {
      await chatStorage.saveMessages(currentSessionId, messages);
    } catch (error) {
      console.error('Error saving chat history:', error);
      // Don't show error toast for saving issues to avoid spam
    }
  };

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  const handleSendMessage = async () => {
    if (!inputMessage.trim() || loading || backgroundChat.isProcessing) return;

    if (documents.length === 0) {
      toast.error('Please upload some documents first to chat with them.');
      return;
    }

    const userMessage: ChatMessage = {
      id: `user-${Date.now()}`,
      role: 'user',
      content: inputMessage.trim(),
      timestamp: new Date(),
    };

    setMessages(prev => [...prev, userMessage]);
    setInputMessage('');

    // Convert messages to conversation history format
    const conversationHistory = messages.map(msg => ({
      role: msg.role === 'user' ? 'user' : 'assistant',
      content: msg.content,
    }));

    if (useBackgroundProcessing) {
      // Use background processing
      try {
        await backgroundChat.startBackgroundChat(
          userMessage.content,
          conversationHistory
        );
      } catch (error) {
        console.error('Error starting background chat:', error);
        const errorMessage: ChatMessage = {
          id: `error-${Date.now()}`,
          role: 'assistant',
          content: 'Sorry, I encountered an error starting background processing. Please try again.',
          timestamp: new Date(),
          error: true,
        };
        setMessages(prev => [...prev, errorMessage]);
      }
    } else {
      // Use regular synchronous processing
      setLoading(true);
      try {
        const response = await chatAPI.chatWithDocuments(
          userMessage.content,
          conversationHistory
        );

        if (response.success) {
          const assistantMessage: ChatMessage = {
            id: `assistant-${Date.now()}`,
            role: 'assistant',
            content: response.response,
            timestamp: new Date(),
          };
          setMessages(prev => [...prev, assistantMessage]);
        } else {
          const errorMessage: ChatMessage = {
            id: `error-${Date.now()}`,
            role: 'assistant',
            content: response.message || 'Sorry, I encountered an error processing your request.',
            timestamp: new Date(),
            error: true,
          };
          setMessages(prev => [...prev, errorMessage]);
        }
      } catch (error) {
        console.error('Error sending message:', error);
        const errorMessage: ChatMessage = {
          id: `error-${Date.now()}`,
          role: 'assistant',
          content: 'Sorry, I encountered an error. Please try again.',
          timestamp: new Date(),
          error: true,
        };
        setMessages(prev => [...prev, errorMessage]);
        toast.error('Failed to send message');
      } finally {
        setLoading(false);
      }
    }
  };

  const handleKeyPress = (event: React.KeyboardEvent) => {
    if (event.key === 'Enter' && !event.shiftKey) {
      event.preventDefault();
      handleSendMessage();
    }
  };

  const startNewChat = async () => {
    try {
      setMessages([]);
      // Create a new session for the next conversation
      const newSessionId = chatStorage.createNewSessionId();
      setCurrentSessionId(newSessionId);
      toast.success('Started a new chat session!');
    } catch (error) {
      console.error('Error starting new chat:', error);
      toast.error('Failed to start new chat');
    }
  };

  const handleSessionSelect = async (sessionId: string) => {
    try {
      setLoadingHistory(true);
      
      // Set the session ID first, then load messages
      setCurrentSessionId(sessionId);
      const sessionMessages = await chatStorage.loadMessages(sessionId);
      setMessages(sessionMessages);
      
      // Scroll to bottom after messages are loaded
      setTimeout(() => {
        scrollToBottom();
      }, 100);
    } catch (error) {
      console.error('Error loading session:', error);
      toast.error('Failed to load chat session');
    } finally {
      setLoadingHistory(false);
    }
  };

  const handleNewSession = () => {
    startNewChat();
  };

  const formatTime = (date: Date) => {
    return date.toLocaleTimeString('en-US', {
      hour: '2-digit',
      minute: '2-digit',
    });
  };

  return (
    <Container maxWidth="xl" sx={{ py: 4, height: '100vh', display: 'flex', flexDirection: 'column' }}>
      <Box sx={{ mb: 4 }}>
        {/* Centered Title */}
        <Box sx={{ textAlign: 'center', mb: 2 }}>
          <Typography variant="h1" component="h1">
            Chat with Documents
          </Typography>
        </Box>
        
        {/* Action Buttons - Centered below title */}
        <Box sx={{ 
          display: 'flex', 
          justifyContent: 'center', 
          gap: 2, 
          mb: 2 
        }}>
          <Button
            variant="outlined"
            startIcon={<History />}
            onClick={() => setHistorySidebarOpen(true)}
          >
            History
          </Button>
          {messages.length > 0 && (
            <Button
              variant="outlined"
              startIcon={<ChatBubble />}
              onClick={startNewChat}
            >
              New Chat
            </Button>
          )}
        </Box>
        <Typography variant="h6" color="text.secondary">
          Ask questions and get insights from your document collection
        </Typography>
        
        {documents.length === 0 && (
          <Alert severity="warning" sx={{ mt: 2 }}>
            No documents found. Please upload some documents first to start chatting.
          </Alert>
        )}
      </Box>

      {/* Chat Container */}
      <Paper elevation={2} sx={{ flexGrow: 1, display: 'flex', flexDirection: 'column' }}>
        {/* Chat Messages Area */}
        <Box sx={{ flexGrow: 1, p: 4, overflowY: 'auto', maxHeight: 'calc(100vh - 300px)' }}>
          {loadingHistory ? (
            <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', py: 8 }}>
              <Box sx={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 2 }}>
                <CircularProgress />
                <Typography variant="body2" color="text.secondary">
                  Loading chat history...
                </Typography>
              </Box>
            </Box>
          ) : messages.length === 0 ? (
            <Box sx={{ textAlign: 'center', py: 8 }}>
              <SmartToy sx={{ fontSize: 64, color: 'primary.main', mb: 2 }} />
              <Typography variant="h5" sx={{ mb: 2 }}>
                Ready to Chat
              </Typography>
              <Typography variant="body1" color="text.secondary" sx={{ mb: 3 }}>
                Ask me anything about your documents. I'll help you find information and insights.
              </Typography>
              <Box sx={{ display: 'flex', justifyContent: 'center', gap: 2, flexWrap: 'wrap' }}>
                <Chip 
                  label={`${documents.length} Documents Available`}
                  color={documents.length > 0 ? 'success' : 'default'}
                  clickable={false}
                  onClick={(e) => e.stopPropagation()}
                  sx={{ fontWeight: 600 }}
                />
                <Chip 
                  label="AI Assistant Online" 
                  color="success"
                  clickable={false}
                  onClick={(e) => e.stopPropagation()}
                  sx={{ fontWeight: 600 }}
                />
              </Box>
            </Box>
          ) : (
            <List sx={{ width: '100%' }}>
              {messages.map((message) => (
                <ListItem
                  key={message.id}
                  sx={{
                    display: 'flex',
                    justifyContent: message.role === 'user' ? 'flex-end' : 'flex-start',
                    alignItems: 'flex-start',
                    mb: 2,
                  }}
                >
                  <Box
                    sx={{
                      display: 'flex',
                      flexDirection: message.role === 'user' ? 'row-reverse' : 'row',
                      alignItems: 'flex-start',
                      maxWidth: '70%',
                      gap: 1,
                    }}
                  >
                    <Avatar
                      sx={{
                        bgcolor: message.role === 'user' ? 'primary.main' : 
                                message.error ? 'error.main' : 'secondary.main',
                        width: 40,
                        height: 40,
                      }}
                    >
                      {message.role === 'user' ? <Person /> : 
                       message.error ? <Error /> : <SmartToy />}
                    </Avatar>
                    <Paper
                      elevation={1}
                      sx={{
                        p: 2,
                        backgroundColor: message.role === 'user' ? 'primary.light' : 
                                       message.error ? 'error.light' : 'grey.100',
                        color: message.role === 'user' ? 'primary.contrastText' : 'text.primary',
                        borderRadius: 2,
                      }}
                    >
                      {message.role === 'assistant' && !message.error ? (
                        <Box sx={{ '& p': { margin: '0.5em 0' }, '& ul, & ol': { paddingLeft: '1.5em' } }}>
                          <ReactMarkdown
                            remarkPlugins={[remarkGfm]}
                            rehypePlugins={[rehypeHighlight, rehypeRaw]}
                            components={{
                              p: ({ children }) => (
                                <Typography variant="body1" component="div" sx={{ mb: 1 }}>
                                  {children}
                                </Typography>
                              ),
                              h1: ({ children }) => (
                                <Typography variant="h4" component="h1" sx={{ mb: 1, mt: 1 }}>
                                  {children}
                                </Typography>
                              ),
                              h2: ({ children }) => (
                                <Typography variant="h5" component="h2" sx={{ mb: 1, mt: 1 }}>
                                  {children}
                                </Typography>
                              ),
                              h3: ({ children }) => (
                                <Typography variant="h6" component="h3" sx={{ mb: 1, mt: 1 }}>
                                  {children}
                                </Typography>
                              ),
                              code: ({ inline, children, ...props }: any) => (
                                <Box
                                  component={inline ? 'code' : 'pre'}
                                  sx={{
                                    backgroundColor: 'grey.200',
                                    padding: inline ? '0.2em 0.4em' : '1em',
                                    borderRadius: 1,
                                    fontFamily: 'monospace',
                                    fontSize: '0.9em',
                                    display: inline ? 'inline' : 'block',
                                    overflow: 'auto',
                                    whiteSpace: inline ? 'normal' : 'pre',
                                  }}
                                  {...props}
                                >
                                  {children}
                                </Box>
                              ),
                              ul: ({ children }) => (
                                <Box component="ul" sx={{ paddingLeft: '1.5em', margin: '0.5em 0' }}>
                                  {children}
                                </Box>
                              ),
                              ol: ({ children }) => (
                                <Box component="ol" sx={{ paddingLeft: '1.5em', margin: '0.5em 0' }}>
                                  {children}
                                </Box>
                              ),
                              li: ({ children }) => (
                                <Typography component="li" variant="body1" sx={{ mb: 0.5 }}>
                                  {children}
                                </Typography>
                              ),
                            }}
                          >
                            {message.content}
                          </ReactMarkdown>
                        </Box>
                      ) : (
                        <Typography variant="body1" sx={{ whiteSpace: 'pre-wrap' }}>
                          {message.content}
                        </Typography>
                      )}
                      <Typography 
                        variant="caption" 
                        color={message.role === 'user' ? 'primary.contrastText' : 'text.secondary'}
                        sx={{ mt: 1, display: 'block' }}
                      >
                        {formatTime(message.timestamp)}
                      </Typography>
                    </Paper>
                  </Box>
                </ListItem>
              ))}
            </List>
          )}
          
          {loading && (
            <Box sx={{ display: 'flex', justifyContent: 'center', p: 2 }}>
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
                <CircularProgress size={20} />
                <Typography variant="body2" color="text.secondary">
                  AI is thinking...
                </Typography>
              </Box>
            </Box>
          )}
          
          <div ref={messagesEndRef} />
        </Box>

        {/* Chat Input */}
        <Box sx={{ p: 4, borderTop: 1, borderColor: 'divider' }}>
          {/* Background Processing Status */}
          {backgroundChat.isProcessing && (
            <Box sx={{ mb: 2, p: 2, backgroundColor: 'info.light', borderRadius: 1 }}>
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
                <CircularProgress size={20} />
                <Typography variant="body2" color="info.dark">
                  {backgroundChat.status === 'pending' 
                    ? 'Your request is queued for processing...' 
                    : 'AI is processing your request in the background...'}
                </Typography>
                <Button
                  size="small"
                  startIcon={<Cancel />}
                  onClick={backgroundChat.cancelJob}
                  sx={{ ml: 'auto' }}
                >
                  Cancel
                </Button>
              </Box>
            </Box>
          )}
          
          {/* Background Processing Toggle */}
          <Box sx={{ mb: 2, display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
            <Tooltip title="Enable background processing to send messages and navigate freely while AI processes your request">
              <FormControlLabel
                control={
                  <Switch
                    checked={useBackgroundProcessing}
                    onChange={(e) => setUseBackgroundProcessing(e.target.checked)}
                    color="primary"
                  />
                }
                label={
                  <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                    <Notifications sx={{ fontSize: 20 }} />
                    <Typography variant="body2">Background Processing</Typography>
                  </Box>
                }
              />
            </Tooltip>
            {useBackgroundProcessing && (
              <Typography variant="caption" color="text.secondary">
                You'll receive a notification when complete
              </Typography>
            )}
          </Box>

          <Box sx={{ display: 'flex', gap: 2, alignItems: 'flex-end' }}>
            <TextField
              fullWidth
              multiline
              maxRows={4}
              placeholder="Ask a question about your documents..."
              variant="outlined"
              value={inputMessage}
              onChange={(e) => setInputMessage(e.target.value)}
              onKeyPress={handleKeyPress}
              disabled={loading || backgroundChat.isProcessing || documents.length === 0}
              sx={{
                '& .MuiOutlinedInput-root': {
                  borderRadius: 3,
                },
              }}
            />
            <Button
              variant="contained"
              endIcon={
                loading || backgroundChat.isProcessing ? <CircularProgress size={20} /> : <Send />
              }
              onClick={handleSendMessage}
              disabled={
                loading || 
                backgroundChat.isProcessing || 
                !inputMessage.trim() || 
                documents.length === 0
              }
              sx={{
                height: 56,
                borderRadius: 3,
                px: 3,
              }}
            >
              {useBackgroundProcessing ? 'Queue' : 'Send'}
            </Button>
          </Box>
        </Box>
      </Paper>

      {/* Chat History Sidebar */}
      <ChatHistorySidebar
        open={historySidebarOpen}
        onClose={() => setHistorySidebarOpen(false)}
        currentSessionId={currentSessionId}
        onSessionSelect={handleSessionSelect}
        onNewSession={handleNewSession}
      />
    </Container>
  );
};

export default ChatPage; 