import axios from 'axios';

// Configure axios base URL
const API_BASE_URL = 'http://localhost:8000/api';

const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Types for API responses
export interface Document {
  id: string;
  title: string;
  type: string;
  content?: string;
  file_path?: string;
  status: 'uploaded' | 'processing' | 'indexed' | 'error';
  created_at: string;
  updated_at: string;
  size_bytes?: number;
}

export interface DocumentResponse {
  success: boolean;
  message: string;
  document?: Document;
  documents?: Document[];
}

export interface ChatResponse {
  success: boolean;
  response: string;
  message?: string;
}

// Document API methods
export const documentAPI = {
  // Upload a file
  async uploadDocument(file: File): Promise<DocumentResponse> {
    const formData = new FormData();
    formData.append('file', file);
    
    const response = await api.post('/documents/upload', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });
    
    return response.data;
  },

  // Get all documents
  async getDocuments(): Promise<DocumentResponse> {
    const response = await api.get('/documents/');
    return response.data;
  },

  // Get a specific document
  async getDocument(documentId: string): Promise<DocumentResponse> {
    const response = await api.get(`/documents/${documentId}`);
    return response.data;
  },

  // Delete a document
  async deleteDocument(documentId: string): Promise<{ success: boolean; message: string }> {
    const response = await api.delete(`/documents/${documentId}`);
    return response.data;
  },
};

// Chat API methods
export const chatAPI = {
  // Query documents
  async queryDocuments(query: string, documentIds?: string[]): Promise<ChatResponse> {
    const response = await api.post('/chat/query', {
      query,
      document_ids: documentIds,
    });
    return response.data;
  },

  // Chat with documents
  async chatWithDocuments(
    message: string, 
    conversationHistory?: Array<{role: string; content: string}>
  ): Promise<ChatResponse> {
    const response = await api.post('/chat/chat', {
      message,
      conversation_history: conversationHistory,
    });
    return response.data;
  },

  // Generate content
  async generateContent(prompt: string, context?: string): Promise<ChatResponse> {
    const response = await api.post('/chat/generate', {
      prompt,
      context,
    });
    return response.data;
  },
};

// Confluence API methods
export const confluenceAPI = {
  // Test connection
  async testConnection(config: {
    url: string;
    username?: string;
    token: string;
    space_key?: string;
    auth_type?: string;
  }): Promise<{ 
    success: boolean; 
    message: string; 
    user?: string;
    space?: string;
  }> {
    try {
      const response = await api.post('/confluence/test', config);
      return response.data;
    } catch (error) {
      if (axios.isAxiosError(error)) {
        if (error.response) {
          // Server responded with error status
          console.error('Confluence test error response:', error.response.data);
          return {
            success: false,
            message: error.response.data?.message || `Server error: ${error.response.status}`
          };
        } else if (error.request) {
          // Request was made but no response received
          return {
            success: false,
            message: 'No response from server. Please check if the backend is running.'
          };
        }
      }
      // Something else happened
      return {
        success: false,
        message: `Request failed: ${error instanceof Error ? error.message : 'Unknown error'}`
      };
    }
  },

  // Import pages
  async importPages(config: {
    url: string;
    username: string;
    token: string;
    space_key: string;
    page_ids?: string[];
  }): Promise<{ success: boolean; message: string; pages?: any[] }> {
    const response = await api.post('/confluence/import', config);
    return response.data;
  },
};

export default api; 