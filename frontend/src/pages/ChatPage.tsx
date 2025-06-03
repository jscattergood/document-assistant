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
} from '@mui/material';
import {
  Send,
  Chat,
  SmartToy,
  Person,
  Error,
} from '@mui/icons-material';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import rehypeHighlight from 'rehype-highlight';
import rehypeRaw from 'rehype-raw';
import 'highlight.js/styles/github.css';
import { chatAPI, documentAPI } from '../services/api';
import type { Document } from '../services/api';
import toast from 'react-hot-toast';

interface ChatMessage {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  timestamp: Date;
  error?: boolean;
}

const ChatPage: React.FC = () => {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [inputMessage, setInputMessage] = useState('');
  const [loading, setLoading] = useState(false);
  const [documents, setDocuments] = useState<Document[]>([]);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    fetchDocuments();
  }, []);

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

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

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  const handleSendMessage = async () => {
    if (!inputMessage.trim() || loading) return;

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
    setLoading(true);

    try {
      // Convert messages to conversation history format
      const conversationHistory = messages.map(msg => ({
        role: msg.role === 'user' ? 'user' : 'assistant',
        content: msg.content,
      }));

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
  };

  const handleKeyPress = (event: React.KeyboardEvent) => {
    if (event.key === 'Enter' && !event.shiftKey) {
      event.preventDefault();
      handleSendMessage();
    }
  };

  const clearChat = () => {
    setMessages([]);
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
        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
          <Typography variant="h1" component="h1">
            Chat with Documents
          </Typography>
          {messages.length > 0 && (
            <Button variant="outlined" onClick={clearChat}>
              Clear Chat
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
        <Box sx={{ flexGrow: 1, p: 3, overflowY: 'auto', maxHeight: 'calc(100vh - 300px)' }}>
          {messages.length === 0 ? (
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
        <Box sx={{ p: 3, borderTop: 1, borderColor: 'divider' }}>
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
              disabled={loading || documents.length === 0}
              sx={{
                '& .MuiOutlinedInput-root': {
                  borderRadius: 3,
                },
              }}
            />
            <Button
              variant="contained"
              endIcon={loading ? <CircularProgress size={20} /> : <Send />}
              onClick={handleSendMessage}
              disabled={loading || !inputMessage.trim() || documents.length === 0}
              sx={{
                height: 56,
                borderRadius: 3,
                px: 3,
              }}
            >
              Send
            </Button>
          </Box>
        </Box>
      </Paper>
    </Container>
  );
};

export default ChatPage; 