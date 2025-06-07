import React, { type ReactNode } from 'react';
import { Link, useLocation } from 'react-router-dom';
import {
  Box,
  Drawer,
  List,
  ListItem,
  ListItemButton,
  ListItemIcon,
  ListItemText,
  Typography,
  Divider,
  Chip,
} from '@mui/material';
import {
  Home as HomeIcon,
  Description as DocumentTextIcon,
  Chat as ChatBubbleLeftRightIcon,
  Cloud as CloudIcon,
  Settings as CogIcon,
} from '@mui/icons-material';
import logoSvg from '../assets/logo.svg';

interface LayoutProps {
  children: ReactNode;
}

const navigation = [
  { name: 'Home', href: '/', icon: HomeIcon },
  { name: 'Documents', href: '/documents', icon: DocumentTextIcon },
  { name: 'Chat', href: '/chat', icon: ChatBubbleLeftRightIcon },
  { name: 'Confluence', href: '/confluence', icon: CloudIcon },
  { name: 'Settings', href: '/settings', icon: CogIcon },
];

const drawerWidth = 280;

const Layout: React.FC<LayoutProps> = ({ children }) => {
  const location = useLocation();

  return (
    <Box sx={{ display: 'flex' }}>
      {/* Sidebar */}
      <Drawer
        variant="permanent"
        sx={{
          width: drawerWidth,
          flexShrink: 0,
          '& .MuiDrawer-paper': {
            width: drawerWidth,
            boxSizing: 'border-box',
            backgroundColor: 'background.paper',
            borderRight: '1px solid',
            borderColor: 'divider',
          },
        }}
      >
        <Box sx={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
          {/* Logo */}
          <Box
            component={Link}
            to="/"
            sx={{
              p: 4,
              borderBottom: '1px solid',
              borderColor: 'divider',
              display: 'flex',
              alignItems: 'center',
              textDecoration: 'none',
              color: 'inherit',
              transition: 'all 0.2s ease-in-out',
              '&:hover': {
                backgroundColor: 'action.hover',
                '& img': {
                  transform: 'scale(1.05)',
                },
              },
            }}
          >
            <Box
              component="img"
              src={logoSvg}
              alt="Document Assistant Logo"
              sx={{
                width: 64,
                height: 64,
                mr: 3,
                flexShrink: 0,
                transition: 'transform 0.2s ease-in-out',
              }}
            />
            <Typography variant="h5" component="h1" sx={{ fontWeight: 600 }}>
              Document Assistant
            </Typography>
          </Box>

          {/* Navigation */}
          <Box sx={{ flexGrow: 1, p: 2 }}>
            <List disablePadding>
              {navigation.map((item) => {
                const isActive = location.pathname === item.href;
                const IconComponent = item.icon;
                return (
                  <ListItem key={item.name} disablePadding sx={{ mb: 0.5 }}>
                    <ListItemButton
                      component={Link}
                      to={item.href}
                      selected={isActive}
                      sx={{
                        borderRadius: 2,
                        '&.Mui-selected': {
                          backgroundColor: 'primary.light',
                          color: 'primary.contrastText',
                          '&:hover': {
                            backgroundColor: 'primary.main',
                          },
                          '& .MuiListItemIcon-root': {
                            color: 'primary.contrastText',
                          },
                        },
                        '&:hover': {
                          backgroundColor: 'action.hover',
                        },
                      }}
                    >
                      <ListItemIcon sx={{ minWidth: 40 }}>
                        <IconComponent />
                      </ListItemIcon>
                      <ListItemText 
                        primary={item.name}
                        primaryTypographyProps={{
                          fontWeight: isActive ? 600 : 500,
                          fontSize: '0.95rem',
                        }}
                      />
                    </ListItemButton>
                  </ListItem>
                );
              })}
            </List>
          </Box>

          {/* Footer */}
          <Box sx={{ p: 3, borderTop: '1px solid', borderColor: 'divider' }}>
            <Box sx={{ textAlign: 'center', mb: 2 }}>
              <Chip
                label="AI Ready"
                color="success"
                size="small"
                clickable={false}
                onClick={(e) => e.stopPropagation()}
                sx={{ mb: 1 }}
              />
            </Box>
            <Typography
              variant="caption"
              color="text.secondary"
              sx={{
                display: 'block',
                textAlign: 'center',
                lineHeight: 1.4,
              }}
            >
              Document Assistant v1.0.0
              <br />
              Powered by LlamaIndex, GPT4All & Ollama
            </Typography>
          </Box>
        </Box>
      </Drawer>

      {/* Main content */}
      <Box
        component="main"
        sx={{
          flexGrow: 1,
          minHeight: '100vh',
          backgroundColor: 'background.default',
        }}
      >
        {children}
      </Box>
    </Box>
  );
};

export default Layout; 