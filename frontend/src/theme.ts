import { createTheme } from '@mui/material/styles';
import type { CSSObject, Theme, Mixins } from '@mui/material/styles';

const drawerWidth = 240;

const openedMixin = (theme: Theme): CSSObject => ({
  width: drawerWidth,
  transition: theme.transitions.create('width', {
    easing: theme.transitions.easing.sharp,
    duration: theme.transitions.duration.enteringScreen,
  }),
  overflowX: 'hidden',
});

const closedMixin = (theme: Theme): CSSObject => ({
  transition: theme.transitions.create('width', {
    easing: theme.transitions.easing.sharp,
    duration: theme.transitions.duration.leavingScreen,
  }),
  overflowX: 'hidden',
  width: `calc(${theme.spacing(7)} + 1px)`,
  [theme.breakpoints.up('sm')]: {
    width: `calc(${theme.spacing(8)} + 1px)`,
  },
});

declare module '@mui/material/styles' {
  interface Mixins {
    openedMixin: typeof openedMixin;
    closedMixin: typeof closedMixin;
  }
}

const theme = createTheme({
  palette: {
    primary: {
      main: '#1976d2',
      light: '#42a5f5',
      dark: '#1565c0',
      contrastText: '#ffffff',
    },
    secondary: {
      main: '#f57c00', // Complementary orange
      light: '#ffb74d',
      dark: '#e65100',
      contrastText: '#ffffff',
    },
    success: {
      main: '#2e7d32', // Professional green
      light: '#4caf50',
      dark: '#1b5e20',
      contrastText: '#ffffff',
    },
    warning: {
      main: '#ed6c02',
      light: '#ff9800',
      dark: '#e65100',
      contrastText: '#ffffff',
    },
    error: {
      main: '#d32f2f',
      light: '#f44336',
      dark: '#b71c1c',
      contrastText: '#ffffff',
    },
    background: {
      default: '#f8fafc', // Slightly cooler white
      paper: '#ffffff',
    },
    text: {
      primary: '#1e293b', // Richer dark blue-gray
      secondary: '#64748b', // Cool gray
    },
    divider: '#e2e8f0',
  },
  typography: {
    fontFamily: '"Inter", "Roboto", "Helvetica", "Arial", sans-serif',
    h1: {
      fontSize: '2.5rem',
      fontWeight: 600,
      lineHeight: 1.2,
    },
    h2: {
      fontSize: '2rem',
      fontWeight: 600,
      lineHeight: 1.3,
    },
    h3: {
      fontSize: '1.5rem',
      fontWeight: 600,
      lineHeight: 1.4,
    },
    h4: {
      fontSize: '1.25rem',
      fontWeight: 600,
      lineHeight: 1.4,
    },
    body1: {
      fontSize: '1rem',
      lineHeight: 1.6,
    },
    body2: {
      fontSize: '0.875rem',
      lineHeight: 1.6,
    },
  },
  shape: {
    borderRadius: 12,
  },
  components: {
    MuiCard: {
      styleOverrides: {
        root: {
          boxShadow: '0 1px 3px rgba(0,0,0,0.12), 0 1px 2px rgba(0,0,0,0.24)',
          transition: 'all 0.3s cubic-bezier(0.25, 0.8, 0.25, 1)',
          '&:hover': {
            boxShadow: '0 14px 28px rgba(0,0,0,0.25), 0 10px 10px rgba(0,0,0,0.22)',
            transform: 'translateY(-2px)',
          },
        },
      },
    },
    MuiButton: {
      styleOverrides: {
        root: {
          textTransform: 'none',
          fontWeight: 600,
          borderRadius: 8,
          boxShadow: 'none',
          '&:hover': {
            boxShadow: '0 2px 8px rgba(25, 118, 210, 0.3)',
          },
          '& .MuiButton-endIcon, & .MuiButton-startIcon': {
            color: 'inherit',
          },
        },
        contained: {
          background: '#1976d2',
          color: '#ffffff',
          '&:hover': {
            background: '#1565c0',
          },
          '&.Mui-disabled': {
            backgroundColor: 'rgba(0, 0, 0, 0.12)',
            color: 'rgba(0, 0, 0, 0.26)',
          },
        },
        containedPrimary: {
          background: '#1976d2',
          color: '#ffffff',
          '&:hover': {
            background: '#1565c0',
          },
        },
        containedSecondary: {
          background: '#f57c00',
          color: '#ffffff',
          '&:hover': {
            background: '#e65100',
          },
        },
        containedSuccess: {
          background: '#2e7d32',
          color: '#ffffff',
          '&:hover': {
            background: '#1b5e20',
          },
        },
        containedError: {
          background: '#d32f2f',
          color: '#ffffff',
          '&:hover': {
            background: '#b71c1c',
          },
        },
        containedWarning: {
          background: '#ed6c02',
          color: '#ffffff',
          '&:hover': {
            background: '#e65100',
          },
        },
        outlined: {
          color: 'inherit',
          '&:hover': {
            backgroundColor: 'rgba(25, 118, 210, 0.04)',
          },
        },
        outlinedPrimary: {
          color: '#1976d2',
        },
        outlinedSecondary: {
          color: '#f57c00',
        },
        outlinedError: {
          color: '#d32f2f',
        },
        outlinedSuccess: {
          color: '#2e7d32',
        },
        outlinedWarning: {
          color: '#ed6c02',
        },
      },
    },
    MuiPaper: {
      styleOverrides: {
        root: {
          boxShadow: '0 1px 3px rgba(0,0,0,0.12), 0 1px 2px rgba(0,0,0,0.24)',
          backgroundImage: 'none',
        },
      },
    },
    MuiDrawer: {
      styleOverrides: {
        paper: {
          backgroundImage: 'linear-gradient(180deg, #ffffff 0%, #f8fafc 100%)',
        },
      },
    },
    MuiChip: {
      styleOverrides: {
        colorSuccess: {
          background: 'linear-gradient(135deg, #2e7d32 0%, #1b5e20 100%)',
          color: '#ffffff',
        },
      },
    },
  },
  mixins: {
    openedMixin,
    closedMixin,
  } as Mixins,
});

export default theme; 