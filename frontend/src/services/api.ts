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
  metadata?: Record<string, any>;
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

export interface JobResponse {
  job_id: string;
  status: string;
  message: string;
}

export interface JobStatusResponse {
  job_id: string;
  status: 'pending' | 'processing' | 'completed' | 'failed';
  message?: string;
  response?: string;
  created_at: string;
  completed_at?: string;
  error?: string;
}

export interface ConfluenceCredentials {
  url: string;
  username?: string;
  api_token: string;
  auth_type: string;
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

  // Background chat processing
  async startBackgroundChat(
    message: string,
    conversationHistory?: Array<{role: string; content: string}>,
    documentIds?: string[]
  ): Promise<JobResponse> {
    const response = await api.post('/chat/chat-background', {
      message,
      conversation_history: conversationHistory,
      document_ids: documentIds,
    });
    return response.data;
  },

  // Check job status
  async getJobStatus(jobId: string): Promise<JobStatusResponse> {
    const response = await api.get(`/chat/job/${jobId}`);
    return response.data;
  },

  // Clean up completed job
  async cleanupJob(jobId: string): Promise<{success: boolean; message: string}> {
    const response = await api.delete(`/chat/job/${jobId}`);
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

  // Add pages to sync list
  async addPagesToSync(
    credentials: {
      url: string;
      username?: string;
      api_token: string;
      auth_type: string;
    },
    webUrls: string[]
  ): Promise<{
    success: boolean;
    message: string;
    synced_pages?: any[];
    errors?: string[];
  }> {
    const response = await api.post('/confluence/sync/add', {
      credentials,
      web_urls: webUrls
    });
    return response.data;
  },

  // List sync pages
  async listSyncPages(): Promise<{
    success: boolean;
    pages: any[];
    count: number;
  }> {
    const response = await api.get('/confluence/sync/list');
    return response.data;
  },

  // Run sync for pages
  async runSync(
    credentials: {
      url: string;
      username?: string;
      api_token: string;
      auth_type: string;
    },
    pageIds?: string[]
  ): Promise<{
    success: boolean;
    message: string;
    synced_count: number;
    errors?: string[];
  }> {
    const response = await api.post('/confluence/sync/run', {
      credentials,
      page_ids: pageIds
    });
    return response.data;
  },

  // Remove page from sync list
  async removeFromSync(pageId: string): Promise<{
    success: boolean;
    message: string;
  }> {
    const response = await api.delete(`/confluence/sync/${pageId}`);
    return response.data;
  },

  // Temporarily ingest a page
  async ingestPageTemporarily(
    credentials: {
      url: string;
      username?: string;
      api_token: string;
      auth_type: string;
    },
    webUrl: string
  ): Promise<{
    success: boolean;
    message: string;
    page_id: string;
    title: string;
    content_preview: string;
    expires_at: string;
  }> {
    const response = await api.post('/confluence/ingest/temporary', {
      credentials,
      web_url: webUrl
    });
    return response.data;
  },

  // List temporary pages
  async listTemporaryPages(): Promise<{
    success: boolean;
    pages: any[];
    count: number;
  }> {
    const response = await api.get('/confluence/ingest/temporary');
    return response.data;
  },

  // Get temporary page content
  async getTemporaryPage(pageId: string): Promise<{
    success: boolean;
    page: any;
  }> {
    const response = await api.get(`/confluence/ingest/temporary/${pageId}`);
    return response.data;
  },

  // Remove temporary page
  async removeTemporaryPage(pageId: string): Promise<{
    success: boolean;
    message: string;
  }> {
    const response = await api.delete(`/confluence/ingest/temporary/${pageId}`);
    return response.data;
  },

  importPageAsDocument: async (credentials: ConfluenceCredentials, webUrl: string) => {
    const response = await api.post('/confluence/import-as-document', {
      credentials,
      web_url: webUrl
    });
    return response.data;
  },
};

// Models API methods
export const modelsAPI = {
  // Get available embedding models
  async getAvailableModels(): Promise<{
    success: boolean;
    models: Record<string, any>;
    count: number;
  }> {
    const response = await api.get('/models/embeddings/available');
    return response.data;
  },

  // Get current embedding model info
  async getCurrentModel(): Promise<{
    success: boolean;
    model_info: any;
  }> {
    const response = await api.get('/models/embeddings/current');
    return response.data;
  },

  // Set embedding model
  async setEmbeddingModel(modelKey: string, useGpu?: boolean): Promise<{
    success: boolean;
    message: string;
    model_info?: any;
  }> {
    const response = await api.post('/models/embeddings/set', {
      model_key: modelKey,
      use_gpu: useGpu,
    });
    return response.data;
  },

  // Test embedding model
  async testEmbeddingModel(text?: string): Promise<{
    success: boolean;
    text: string;
    embedding_length: number;
    embedding_sample: number[];
    model_info: any;
  }> {
    const params = text ? { text } : {};
    const response = await api.post('/models/embeddings/test', {}, { params });
    return response.data;
  },

  // Get system status
  async getSystemStatus(): Promise<{
    success: boolean;
    gpu_info: any;
    embedding_model: any;
    available_models: string[];
    document_count: number;
  }> {
    const response = await api.get('/models/system/status');
    return response.data;
  },

  // Clear vector store
  async clearVectorStore(): Promise<{
    success: boolean;
    message: string;
  }> {
    const response = await api.post('/models/vector-store/clear');
    return response.data;
  },

  // Storage configuration methods
  getStorageConfig: () => api.get('/models/storage/config'),
  getStorageStatus: () => api.get('/models/storage/status'),
  validateStoragePath: (path: string) => api.post('/models/storage/validate-path', { path }),
  clearStorageCache: () => api.post('/models/storage/clear-cache'),

  // GPT4All Model Management
  async getAvailableGPT4AllModels(): Promise<{
    success: boolean;
    models: any[];
    total_count: number;
    downloaded_count: number;
    active_model: string | null;
  }> {
    const response = await api.get('/models/gpt4all/available');
    return response.data;
  },

  async getDownloadedGPT4AllModels(): Promise<{
    success: boolean;
    models: any[];
    count: number;
    active_model: string | null;
  }> {
    const response = await api.get('/models/gpt4all/downloaded');
    return response.data;
  },

  async downloadGPT4AllModel(modelName: string, downloadUrl: string): Promise<{
    success: boolean;
    message: string;
    filename?: string;
    size_human?: string;
  }> {
    const response = await api.post('/models/gpt4all/download', {
      model_name: modelName,
      download_url: downloadUrl,
    });
    return response.data;
  },

  async uploadGPT4AllModel(file: File): Promise<{
    success: boolean;
    message: string;
    filename?: string;
    size_bytes?: number;
    size_human?: string;
  }> {
    const formData = new FormData();
    formData.append('file', file);
    
    const response = await api.post('/models/gpt4all/upload', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });
    return response.data;
  },

  async deleteGPT4AllModel(filename: string): Promise<{
    success: boolean;
    message: string;
  }> {
    const response = await api.delete(`/models/gpt4all/${filename}`);
    return response.data;
  },

  async setActiveGPT4AllModel(filename: string): Promise<{
    success: boolean;
    message: string;
    active_model?: string;
  }> {
    const response = await api.post('/models/gpt4all/set-active', null, {
      params: { filename },
    });
    return response.data;
  },

  async getDownloadStatus(filename: string): Promise<{
    success: boolean;
    exists: boolean;
    size_bytes: number;
    size_human?: string;
    expected_size?: number;
    is_complete?: boolean;
    progress: number;
  }> {
    const response = await api.get(`/models/gpt4all/download-status/${filename}`);
    return response.data;
  },

  // Settings endpoints
  async getSettings(): Promise<{
    success: boolean;
    max_tokens: number;
    temperature: number;
    use_document_context: boolean;
    enable_notifications: boolean;
  }> {
    const response = await api.get('/models/settings');
    return response.data;
  },

  async updateSettings(settings: {
    max_tokens?: number;
    temperature?: number;
    use_document_context?: boolean;
    enable_notifications?: boolean;
  }): Promise<{
    success: boolean;
    message: string;
    settings: any;
  }> {
    const response = await api.post('/models/settings', settings);
    return response.data;
  },

  async resetSettings(): Promise<{
    success: boolean;
    message: string;
    settings: any;
  }> {
    const response = await api.post('/models/settings/reset');
    return response.data;
  },

  // Provider Management
  async getCurrentProvider(): Promise<{
    success: boolean;
    provider: string;
    gpt4all_model?: string;
    ollama_model?: string;
  }> {
    const response = await api.get('/models/providers/current');
    return response.data;
  },

  async setProvider(provider: string, model?: string): Promise<{
    success: boolean;
    message: string;
    provider: string;
    model?: string;
  }> {
    const params = new URLSearchParams({ provider });
    if (model) {
      params.append('model', model);
    }
    const response = await api.post(`/models/providers/set?${params.toString()}`);
    return response.data;
  },

  async getOllamaModels(): Promise<{
    success: boolean;
    models: string[];
    ollama_running: boolean;
    message?: string;
  }> {
    const response = await api.get('/models/ollama/models');
    return response.data;
  },

  // Ollama Process Management
  async getOllamaStatus(): Promise<{
    success: boolean;
    running: boolean;
    responding: boolean;
    process_id?: string;
    version?: string;
    models_count: number;
    models: string[];
    can_control: boolean;
    platform: string;
  }> {
    // Add cache busting parameter
    const response = await api.get(`/models/ollama/status?_t=${Date.now()}`);
    return response.data;
  },

  async startOllama(): Promise<{
    success: boolean;
    message: string;
    already_running?: boolean;
    method?: string;
    install_url?: string;
  }> {
    const response = await api.post('/models/ollama/start');
    return response.data;
  },

  async stopOllama(): Promise<{
    success: boolean;
    message: string;
    already_stopped?: boolean;
    method?: string;
  }> {
    const response = await api.post('/models/ollama/stop');
    return response.data;
  },

  async restartOllama(): Promise<{
    success: boolean;
    message: string;
    stop_method?: string;
    start_method?: string;
  }> {
    const response = await api.post('/models/ollama/restart');
    return response.data;
  },
};

export default api; 