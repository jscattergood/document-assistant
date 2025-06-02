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
  Select,
  MenuItem,
  FormControl,
  InputLabel,
  Divider,
  Slider,
} from '@mui/material';
import {
  Settings,
  Save,
  Refresh,
  Storage,
  Psychology,
  Security,
} from '@mui/icons-material';

const SettingsPage: React.FC = () => {
  return (
    <Container maxWidth="xl" sx={{ py: 4 }}>
      <Box sx={{ mb: 6 }}>
        <Typography variant="h1" component="h1" sx={{ mb: 2 }}>
          Settings
        </Typography>
        <Typography variant="h6" color="text.secondary">
          Configure your Document Assistant preferences
        </Typography>
      </Box>

      <Grid container spacing={4}>
        {/* AI Model Settings */}
        <Grid size={{ xs: 12, lg: 6 }}>
          <Paper elevation={2} sx={{ p: 4, mb: 4 }}>
            <Box sx={{ display: 'flex', alignItems: 'center', mb: 3 }}>
              <Psychology sx={{ mr: 2, color: 'primary.main' }} />
              <Typography variant="h4" component="h2">
                AI Model Settings
              </Typography>
            </Box>
            
            <Box sx={{ '& .MuiFormControl-root': { mb: 3 } }}>
              <FormControl fullWidth>
                <InputLabel>Model Type</InputLabel>
                <Select value="gpt4all" label="Model Type">
                  <MenuItem value="gpt4all">GPT4All (Offline)</MenuItem>
                  <MenuItem value="llama2">Llama2 (Offline)</MenuItem>
                </Select>
              </FormControl>

              <Box sx={{ mb: 3 }}>
                <Typography gutterBottom>Response Temperature</Typography>
                <Slider
                  defaultValue={0.7}
                  min={0}
                  max={1}
                  step={0.1}
                  marks
                  valueLabelDisplay="auto"
                />
              </Box>

              <Box sx={{ mb: 3 }}>
                <Typography gutterBottom>Max Response Length</Typography>
                <Slider
                  defaultValue={512}
                  min={128}
                  max={2048}
                  step={128}
                  marks
                  valueLabelDisplay="auto"
                />
              </Box>

              <FormControlLabel
                control={<Switch defaultChecked />}
                label="Use document context in responses"
              />
            </Box>
          </Paper>

          {/* Storage Settings */}
          <Paper elevation={2} sx={{ p: 4 }}>
            <Box sx={{ display: 'flex', alignItems: 'center', mb: 3 }}>
              <Storage sx={{ mr: 2, color: 'primary.main' }} />
              <Typography variant="h4" component="h2">
                Storage Settings
              </Typography>
            </Box>
            
            <Box sx={{ '& .MuiTextField-root': { mb: 3 } }}>
              <TextField
                fullWidth
                label="Documents Directory"
                value="/path/to/documents"
                variant="outlined"
                helperText="Location where uploaded documents are stored"
              />
              
              <TextField
                fullWidth
                label="Vector Database Path"
                value="/path/to/vectordb"
                variant="outlined"
                helperText="Location of the vector database for document embeddings"
              />

              <FormControlLabel
                control={<Switch defaultChecked />}
                label="Auto-backup document index"
                sx={{ mb: 2 }}
              />

              <Button variant="outlined" startIcon={<Refresh />}>
                Clear Cache
              </Button>
            </Box>
          </Paper>
        </Grid>

        {/* General & Security Settings */}
        <Grid size={{ xs: 12, lg: 6 }}>
          <Paper elevation={2} sx={{ p: 4, mb: 4 }}>
            <Box sx={{ display: 'flex', alignItems: 'center', mb: 3 }}>
              <Settings sx={{ mr: 2, color: 'primary.main' }} />
              <Typography variant="h4" component="h2">
                General Settings
              </Typography>
            </Box>
            
            <Box sx={{ '& .MuiFormControl-root': { mb: 3 } }}>
              <FormControl fullWidth>
                <InputLabel>Language</InputLabel>
                <Select value="en" label="Language">
                  <MenuItem value="en">English</MenuItem>
                  <MenuItem value="es">Spanish</MenuItem>
                  <MenuItem value="fr">French</MenuItem>
                  <MenuItem value="de">German</MenuItem>
                </Select>
              </FormControl>

              <FormControl fullWidth>
                <InputLabel>Theme</InputLabel>
                <Select value="light" label="Theme">
                  <MenuItem value="light">Light</MenuItem>
                  <MenuItem value="dark">Dark</MenuItem>
                  <MenuItem value="auto">Auto</MenuItem>
                </Select>
              </FormControl>

              <FormControlLabel
                control={<Switch defaultChecked />}
                label="Enable notifications"
                sx={{ mb: 2 }}
              />

              <FormControlLabel
                control={<Switch defaultChecked />}
                label="Auto-save chat history"
              />
            </Box>
          </Paper>

          <Paper elevation={2} sx={{ p: 4 }}>
            <Box sx={{ display: 'flex', alignItems: 'center', mb: 3 }}>
              <Security sx={{ mr: 2, color: 'primary.main' }} />
              <Typography variant="h4" component="h2">
                Privacy & Security
              </Typography>
            </Box>
            
            <Box>
              <FormControlLabel
                control={<Switch defaultChecked />}
                label="Keep data local (offline mode)"
                sx={{ mb: 2 }}
              />

              <FormControlLabel
                control={<Switch />}
                label="Enable usage analytics"
                sx={{ mb: 2 }}
              />

              <Divider sx={{ my: 3 }} />

              <Button variant="outlined" color="error" sx={{ mr: 2 }}>
                Delete All Data
              </Button>
              
              <Button variant="outlined">
                Export Data
              </Button>
            </Box>
          </Paper>
        </Grid>
      </Grid>

      {/* Save Button */}
      <Box sx={{ mt: 4, textAlign: 'center' }}>
        <Button variant="contained" size="large" startIcon={<Save />}>
          Save All Settings
        </Button>
      </Box>
    </Container>
  );
};

export default SettingsPage; 