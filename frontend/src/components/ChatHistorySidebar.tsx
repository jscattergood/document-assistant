import React, { useState, useEffect } from 'react';
import {
  Box,
  Drawer,
  List,
  ListItem,
  ListItemButton,
  ListItemText,
  ListItemIcon,
  Typography,
  IconButton,
  Divider,
  Tooltip,
  Menu,
  MenuItem,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  TextField,
  Button,
} from '@mui/material';
import {
  History,
  Chat,
  Delete,
  Edit,
  MoreVert,
  Add,
} from '@mui/icons-material';
import { chatStorage, type ChatSession } from '../utils/chatStorage';
import toast from 'react-hot-toast';

interface ChatHistorySidebarProps {
  open: boolean;
  onClose: () => void;
  currentSessionId: string;
  onSessionSelect: (sessionId: string) => void;
  onNewSession: () => void;
}

const ChatHistorySidebar: React.FC<ChatHistorySidebarProps> = ({
  open,
  onClose,
  currentSessionId,
  onSessionSelect,
  onNewSession,
}) => {
  const [sessions, setSessions] = useState<ChatSession[]>([]);
  const [loading, setLoading] = useState(false);
  const [sessionPreviews, setSessionPreviews] = useState<Record<string, { messageCount: number; firstMessage: string }>>({});
  const [menuAnchor, setMenuAnchor] = useState<null | HTMLElement>(null);
  const [selectedSession, setSelectedSession] = useState<ChatSession | null>(null);
  const [renameDialog, setRenameDialog] = useState(false);
  const [newName, setNewName] = useState('');

  useEffect(() => {
    if (open) {
      loadSessions();
    }
  }, [open]);

  const loadSessions = async () => {
    try {
      setLoading(true);
      const allSessions = await chatStorage.getAllSessions();
      setSessions(allSessions);
      
      // Load message previews for each session
      const previews: Record<string, { messageCount: number; firstMessage: string }> = {};
      
      for (const session of allSessions) {
        try {
          const messages = await chatStorage.loadMessages(session.id);
          previews[session.id] = {
            messageCount: messages.length,
            firstMessage: messages.length > 0 
              ? messages[0].content.substring(0, 50) + (messages[0].content.length > 50 ? '...' : '')
              : 'No messages'
          };
        } catch (error) {
          console.error('Error loading messages for session:', session.id, error);
          previews[session.id] = {
            messageCount: 0,
            firstMessage: 'Error loading messages'
          };
        }
      }
      
      setSessionPreviews(previews);
    } catch (error) {
      console.error('Error loading sessions:', error);
      toast.error('Failed to load chat history');
    } finally {
      setLoading(false);
    }
  };

  const handleMenuOpen = (event: React.MouseEvent<HTMLElement>, session: ChatSession) => {
    event.stopPropagation();
    setMenuAnchor(event.currentTarget);
    setSelectedSession(session);
  };

  const handleMenuClose = () => {
    setMenuAnchor(null);
    setSelectedSession(null);
  };

  const handleRename = () => {
    if (selectedSession) {
      setNewName(selectedSession.name);
      setRenameDialog(true);
    }
    handleMenuClose();
  };

  const handleRenameSubmit = async () => {
    if (selectedSession && newName.trim()) {
      try {
        await chatStorage.renameSession(selectedSession.id, newName.trim());
        await loadSessions();
        toast.success('Session renamed successfully');
      } catch (error) {
        console.error('Error renaming session:', error);
        toast.error('Failed to rename session');
      }
    }
    setRenameDialog(false);
    setNewName('');
  };

  const handleDelete = async () => {
    if (selectedSession) {
      try {
        await chatStorage.deleteSession(selectedSession.id);
        await loadSessions();
        
        // If we deleted the current session, create a new one
        if (selectedSession.id === currentSessionId) {
          onNewSession();
        }
        
        toast.success('Session deleted successfully');
      } catch (error) {
        console.error('Error deleting session:', error);
        toast.error('Failed to delete session');
      }
    }
    handleMenuClose();
  };

  const formatSessionDate = (date: Date) => {
    const now = new Date();
    const isToday = date.toDateString() === now.toDateString();
    const yesterday = new Date(now.getTime() - 24 * 60 * 60 * 1000);
    const isYesterday = date.toDateString() === yesterday.toDateString();

    if (isToday) {
      return `Today ${date.toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit' })}`;
    } else if (isYesterday) {
      return `Yesterday ${date.toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit' })}`;
    } else {
      return date.toLocaleDateString('en-US', { 
        month: 'short', 
        day: 'numeric',
        hour: '2-digit',
        minute: '2-digit'
      });
    }
  };

  const drawerWidth = 320;

  return (
    <>
      <Drawer
        anchor="left"
        open={open}
        onClose={onClose}
        sx={{
          width: drawerWidth,
          flexShrink: 0,
          '& .MuiDrawer-paper': {
            width: drawerWidth,
            boxSizing: 'border-box',
          },
        }}
      >
        <Box sx={{ p: 2, display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
            <History color="primary" />
            <Typography variant="h6">Chat History</Typography>
          </Box>
          <Tooltip title="New Chat">
            <IconButton onClick={onNewSession} color="primary">
              <Add />
            </IconButton>
          </Tooltip>
        </Box>
        
        <Divider />
        
        <Box sx={{ flexGrow: 1, overflow: 'auto' }}>
          {loading ? (
            <Box sx={{ p: 2, textAlign: 'center' }}>
              <Typography variant="body2" color="text.secondary">
                Loading sessions...
              </Typography>
            </Box>
          ) : sessions.length === 0 ? (
            <Box sx={{ p: 2, textAlign: 'center' }}>
              <Typography variant="body2" color="text.secondary">
                No chat history yet
              </Typography>
            </Box>
          ) : (
            <List>
              {sessions.map((session) => {
                const isActive = session.id === currentSessionId;
                const preview = sessionPreviews[session.id];
                const messagePreview = preview?.firstMessage || 'Loading...';
                const messageCount = preview?.messageCount || 0;

                return (
                  <ListItem
                    key={session.id}
                    disablePadding
                    sx={{
                      backgroundColor: isActive ? 'action.selected' : 'transparent',
                      '&:hover': {
                        backgroundColor: isActive ? 'action.selected' : 'action.hover',
                      },
                    }}
                  >
                    <ListItemButton
                      onClick={() => {
                        onSessionSelect(session.id);
                        onClose();
                      }}
                      sx={{ pr: 1 }}
                    >
                      <ListItemIcon>
                        <Chat sx={{ color: isActive ? 'primary.main' : 'text.secondary' }} />
                      </ListItemIcon>
                      <ListItemText
                        primary={
                          <Typography
                            variant="body2"
                            sx={{
                              fontWeight: isActive ? 600 : 400,
                              color: isActive ? 'primary.main' : 'text.primary',
                            }}
                          >
                            {session.name}
                          </Typography>
                        }
                        secondary={
                          <Box component="span">
                            <Typography variant="caption" color="text.secondary" sx={{ display: 'block' }}>
                              {messagePreview}
                            </Typography>
                            <Typography variant="caption" color="text.secondary">
                              {messageCount} messages • {formatSessionDate(session.updatedAt)}
                            </Typography>
                          </Box>
                        }
                      />
                      <IconButton
                        size="small"
                        onClick={(e) => handleMenuOpen(e, session)}
                        sx={{ ml: 1 }}
                      >
                        <MoreVert fontSize="small" />
                      </IconButton>
                    </ListItemButton>
                  </ListItem>
                );
              })}
            </List>
          )}
        </Box>
      </Drawer>

      {/* Context Menu */}
      <Menu
        anchorEl={menuAnchor}
        open={Boolean(menuAnchor)}
        onClose={handleMenuClose}
      >
        <MenuItem onClick={handleRename}>
          <ListItemIcon>
            <Edit fontSize="small" />
          </ListItemIcon>
          <ListItemText>Rename</ListItemText>
        </MenuItem>
        <MenuItem onClick={handleDelete}>
          <ListItemIcon>
            <Delete fontSize="small" />
          </ListItemIcon>
          <ListItemText>Delete</ListItemText>
        </MenuItem>
      </Menu>

      {/* Rename Dialog */}
      <Dialog open={renameDialog} onClose={() => setRenameDialog(false)} maxWidth="sm" fullWidth>
        <DialogTitle>Rename Chat Session</DialogTitle>
        <DialogContent>
          <TextField
            autoFocus
            margin="dense"
            label="Session Name"
            fullWidth
            variant="outlined"
            value={newName}
            onChange={(e) => setNewName(e.target.value)}
            onKeyPress={(e) => {
              if (e.key === 'Enter') {
                handleRenameSubmit();
              }
            }}
          />
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setRenameDialog(false)}>Cancel</Button>
          <Button onClick={handleRenameSubmit} variant="contained">
            Save
          </Button>
        </DialogActions>
      </Dialog>
    </>
  );
};

export default ChatHistorySidebar; 