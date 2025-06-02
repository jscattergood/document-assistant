import React from 'react';
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
  Switch,
  FormControlLabel,
  Alert,
} from '@mui/material';
import {
  Cloud,
  Settings,
  Save,
  Sync,
} from '@mui/icons-material';

const ConfluencePage: React.FC = () => {
  return (
    <Container maxWidth="xl" sx={{ py: 4 }}>
      <Box sx={{ mb: 6 }}>
        <Typography variant="h1" component="h1" sx={{ mb: 2 }}>
          Confluence Integration
        </Typography>
        <Typography variant="h6" color="text.secondary">
          Connect and sync with your Confluence workspace
        </Typography>
      </Box>

      <Grid container spacing={4}>
        {/* Configuration */}
        <Grid size={{ xs: 12, lg: 8 }}>
          <Paper elevation={2} sx={{ p: 4 }}>
            <Typography variant="h4" component="h2" sx={{ mb: 4 }}>
              Connection Settings
            </Typography>
            
            <Alert severity="info" sx={{ mb: 4 }}>
              Configure your Confluence settings to enable automatic synchronization and content generation.
            </Alert>

            <Box component="form" sx={{ '& .MuiTextField-root': { mb: 3 } }}>
              <TextField
                fullWidth
                label="Confluence URL"
                placeholder="https://your-domain.atlassian.net"
                variant="outlined"
              />
              
              <TextField
                fullWidth
                label="Username/Email"
                placeholder="your-email@domain.com"
                variant="outlined"
              />
              
              <TextField
                fullWidth
                label="API Token"
                type="password"
                placeholder="Your Confluence API Token"
                variant="outlined"
                helperText="Generate an API token from your Confluence account settings"
              />
              
              <TextField
                fullWidth
                label="Space Key"
                placeholder="DOCS"
                variant="outlined"
                helperText="The space key where documents will be created"
              />

              <FormControlLabel
                control={<Switch />}
                label="Enable automatic synchronization"
                sx={{ mb: 3 }}
              />

              <Box sx={{ display: 'flex', gap: 2 }}>
                <Button variant="contained" startIcon={<Save />} size="large">
                  Save Configuration
                </Button>
                <Button variant="outlined" startIcon={<Sync />} size="large">
                  Test Connection
                </Button>
              </Box>
            </Box>
          </Paper>
        </Grid>

        {/* Status */}
        <Grid size={{ xs: 12, lg: 4 }}>
          <Card sx={{ mb: 3 }}>
            <CardContent sx={{ textAlign: 'center' }}>
              <Cloud sx={{ fontSize: 48, color: 'text.secondary', mb: 2 }} />
              <Typography variant="h6" sx={{ mb: 1 }}>
                Connection Status
              </Typography>
              <Typography variant="body2" color="text.secondary">
                Not Connected
              </Typography>
            </CardContent>
          </Card>

          <Card>
            <CardContent>
              <Typography variant="h6" sx={{ mb: 2 }}>
                Quick Actions
              </Typography>
              <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
                <Button variant="outlined" fullWidth disabled>
                  Sync Documents
                </Button>
                <Button variant="outlined" fullWidth disabled>
                  Create Page
                </Button>
                <Button variant="outlined" fullWidth disabled>
                  View Spaces
                </Button>
              </Box>
            </CardContent>
          </Card>
        </Grid>
      </Grid>
    </Container>
  );
};

export default ConfluencePage; 