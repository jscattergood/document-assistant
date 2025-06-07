import React from 'react';
import { useNavigate } from 'react-router-dom';
import {
  Container,
  Typography,
  Card,
  CardContent,
  Button,
  Box,
  Avatar,
  Chip,
  Stepper,
  Step,
  StepLabel,
  Paper,
} from '@mui/material';
import {
  CloudUpload as DocumentArrowUpIcon,
  Chat as ChatBubbleLeftRightIcon,
  Create as PencilSquareIcon,
  Description,
  QuestionAnswer,
  AutoAwesome,
  CheckCircle,
} from '@mui/icons-material';

const steps = [
  'Upload your first document to get started',
  'Configure Confluence integration (optional)',
  'Start chatting with your documents',
  'Generate new content based on your knowledge base'
];

const HomePage: React.FC = () => {
  const navigate = useNavigate();

  return (
    <Box>
      {/* Hero Section */}
      <Box
        sx={{
          background: 'linear-gradient(135deg, #1976d2 0%, #1565c0 50%, #0d47a1 100%)',
          color: 'white',
          py: 8,
          mb: 4,
        }}
      >
        <Container maxWidth="xl">
          <Box sx={{ textAlign: 'center' }}>
            <Typography variant="h1" component="h1" sx={{ mb: 2, fontWeight: 700 }}>
              Welcome to Document Assistant
            </Typography>
            <Typography variant="h5" sx={{ mb: 4, opacity: 0.9, maxWidth: 800, mx: 'auto' }}>
              AI-powered document analysis and generation tool with offline capabilities.
              Upload, analyze, chat, and generate content with cutting-edge AI technology.
            </Typography>
            <Box sx={{ display: 'flex', gap: 2, justifyContent: 'center', flexWrap: 'wrap' }}>
              <Button
                variant="contained"
                size="large"
                onClick={() => navigate('/documents')}
                sx={{
                  bgcolor: 'rgba(255,255,255,0.2)',
                  backdropFilter: 'blur(10px)',
                  border: '1px solid rgba(255,255,255,0.3)',
                  '&:hover': {
                    bgcolor: 'rgba(255,255,255,0.3)',
                  },
                }}
              >
                Get Started
              </Button>
              <Button
                variant="outlined"
                size="large"
                onClick={() => navigate('/settings')}
                sx={{
                  borderColor: 'rgba(255,255,255,0.5)',
                  color: 'white',
                  '&:hover': {
                    borderColor: 'rgba(255,255,255,0.8)',
                    bgcolor: 'rgba(255,255,255,0.1)',
                  },
                }}
              >
                Configure AI Models
              </Button>
            </Box>
          </Box>
        </Container>
      </Box>

      <Container maxWidth="xl" sx={{ py: 4 }}>

      {/* Feature cards */}
      <Box sx={{ display: 'grid', gridTemplateColumns: { xs: '1fr', md: 'repeat(2, 1fr)', lg: 'repeat(3, 1fr)' }, gap: 3, mb: 6 }}>
        {/* Upload Documents */}
        <Card sx={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
          <CardContent sx={{ textAlign: 'center', flexGrow: 1, p: 4 }}>
            <Avatar
              sx={{
                bgcolor: 'primary.light',
                width: 64,
                height: 64,
                mx: 'auto',
                mb: 3,
              }}
            >
              <DocumentArrowUpIcon sx={{ fontSize: 32 }} />
            </Avatar>
            <Typography variant="h4" component="h3" sx={{ mb: 2 }}>
              Upload Documents
            </Typography>
            <Typography variant="body1" color="text.secondary" sx={{ mb: 3 }}>
              Upload PDFs, Word docs, text files, and more for AI analysis
            </Typography>
            <Button
              variant="contained"
              color="primary"
              size="large"
              fullWidth
              onClick={() => navigate('/documents')}
              sx={{ mt: 'auto' }}
            >
              Start Uploading
            </Button>
          </CardContent>
        </Card>

        {/* Chat with Documents */}
        <Card sx={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
          <CardContent sx={{ textAlign: 'center', flexGrow: 1, p: 4 }}>
            <Avatar
              sx={{
                bgcolor: 'success.light',
                width: 64,
                height: 64,
                mx: 'auto',
                mb: 3,
              }}
            >
              <ChatBubbleLeftRightIcon sx={{ fontSize: 32 }} />
            </Avatar>
            <Typography variant="h4" component="h3" sx={{ mb: 2 }}>
              Chat with Documents
            </Typography>
            <Typography variant="body1" color="text.secondary" sx={{ mb: 3 }}>
              Ask questions and get insights from your document collection
            </Typography>
            <Button
              variant="contained"
              color="success"
              size="large"
              fullWidth
              onClick={() => navigate('/chat')}
              sx={{ mt: 'auto' }}
            >
              Start Chatting
            </Button>
          </CardContent>
        </Card>

        {/* Generate Content */}
        <Card sx={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
          <CardContent sx={{ textAlign: 'center', flexGrow: 1, p: 4 }}>
            <Avatar
              sx={{
                bgcolor: 'secondary.light',
                width: 64,
                height: 64,
                mx: 'auto',
                mb: 3,
              }}
            >
              <PencilSquareIcon sx={{ fontSize: 32 }} />
            </Avatar>
            <Typography variant="h4" component="h3" sx={{ mb: 2 }}>
              Generate Content
            </Typography>
            <Typography variant="body1" color="text.secondary" sx={{ mb: 3 }}>
              Create new documents and Confluence pages with AI assistance
            </Typography>
            <Button
              variant="contained"
              color="secondary"
              size="large"
              fullWidth
              onClick={() => navigate('/confluence')}
              sx={{ mt: 'auto' }}
            >
              Generate Now
            </Button>
          </CardContent>
        </Card>
      </Box>

      {/* Quick Stats */}
      <Paper elevation={2} sx={{ p: 4, mb: 6 }}>
        <Typography variant="h2" component="h2" sx={{ mb: 4 }}>
          Quick Overview
        </Typography>
        <Box sx={{ display: 'grid', gridTemplateColumns: { xs: '1fr', sm: 'repeat(2, 1fr)', md: 'repeat(4, 1fr)' }, gap: 3 }}>
          <Box sx={{ textAlign: 'center', p: 3, bgcolor: 'primary.light', borderRadius: 2 }}>
            <Typography variant="h3" color="primary.contrastText" sx={{ mb: 1 }}>
              0
            </Typography>
            <Typography variant="body2" color="primary.contrastText">
              Documents Uploaded
            </Typography>
          </Box>
          <Box sx={{ textAlign: 'center', p: 3, bgcolor: 'success.light', borderRadius: 2 }}>
            <Typography variant="h3" color="success.contrastText" sx={{ mb: 1 }}>
              0
            </Typography>
            <Typography variant="body2" color="success.contrastText">
              Chat Sessions
            </Typography>
          </Box>
          <Box sx={{ textAlign: 'center', p: 3, bgcolor: 'secondary.light', borderRadius: 2 }}>
            <Typography variant="h3" color="secondary.contrastText" sx={{ mb: 1 }}>
              0
            </Typography>
            <Typography variant="body2" color="secondary.contrastText">
              Pages Generated
            </Typography>
          </Box>
          <Box sx={{ textAlign: 'center', p: 3, bgcolor: 'success.main', borderRadius: 2 }}>
            <Typography variant="h3" color="success.contrastText" sx={{ mb: 1 }}>
              Ready
            </Typography>
            <Typography variant="body2" color="success.contrastText">
              AI Model Status
            </Typography>
          </Box>
        </Box>
      </Paper>

      {/* Getting Started */}
      <Paper 
        elevation={2} 
        sx={{ 
          p: 4, 
          background: 'linear-gradient(135deg, #e3f2fd 0%, #f3e5f5 100%)',
        }}
      >
        <Typography variant="h2" component="h2" sx={{ mb: 4 }}>
          Getting Started
        </Typography>
        <Stepper orientation="vertical">
          {steps.map((step, index) => (
            <Step key={index} active={true}>
              <StepLabel
                StepIconComponent={() => (
                  <Avatar
                    sx={{
                      bgcolor: 'primary.main',
                      width: 32,
                      height: 32,
                      fontSize: '0.875rem',
                      fontWeight: 600,
                    }}
                  >
                    {index + 1}
                  </Avatar>
                )}
              >
                <Typography variant="body1" sx={{ fontWeight: 500 }}>
                  {step}
                </Typography>
              </StepLabel>
            </Step>
          ))}
        </Stepper>
      </Paper>
      </Container>
    </Box>
  );
};

export default HomePage; 