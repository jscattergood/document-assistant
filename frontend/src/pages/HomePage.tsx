import React from 'react';

const HomePage: React.FC = () => {
  return (
    <div className="p-8">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <div className="mb-8">
          <h1 className="text-3xl font-bold text-gray-900">Welcome to Document Assistant</h1>
          <p className="mt-2 text-lg text-gray-600">
            AI-powered document analysis and generation tool with offline capabilities
          </p>
        </div>

        {/* Feature cards */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6 mb-8">
          {/* Upload Documents */}
          <div className="bg-white rounded-lg shadow p-6">
            <div className="text-center">
              <div className="mx-auto h-12 w-12 text-blue-600 mb-4">
                <svg className="w-full h-full" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" />
                </svg>
              </div>
              <h3 className="text-lg font-medium text-gray-900 mb-2">Upload Documents</h3>
              <p className="text-sm text-gray-500 mb-4">
                Upload PDFs, Word docs, text files, and more for AI analysis
              </p>
              <button className="bg-blue-600 text-white px-4 py-2 rounded-md hover:bg-blue-700 transition-colors">
                Start Uploading
              </button>
            </div>
          </div>

          {/* Chat with Documents */}
          <div className="bg-white rounded-lg shadow p-6">
            <div className="text-center">
              <div className="mx-auto h-12 w-12 text-green-600 mb-4">
                <svg className="w-full h-full" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 12h.01M12 12h.01M16 12h.01M21 12c0 4.418-4.03 8-9 8a9.863 9.863 0 01-4.255-.949L3 20l1.395-3.72C3.512 15.042 3 13.574 3 12c0-4.418 4.03-8 9-8s9 3.582 9 8z" />
                </svg>
              </div>
              <h3 className="text-lg font-medium text-gray-900 mb-2">Chat with Documents</h3>
              <p className="text-sm text-gray-500 mb-4">
                Ask questions and get insights from your document collection
              </p>
              <button className="bg-green-600 text-white px-4 py-2 rounded-md hover:bg-green-700 transition-colors">
                Start Chatting
              </button>
            </div>
          </div>

          {/* Generate Content */}
          <div className="bg-white rounded-lg shadow p-6">
            <div className="text-center">
              <div className="mx-auto h-12 w-12 text-purple-600 mb-4">
                <svg className="w-full h-full" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M11 5H6a2 2 0 00-2 2v11a2 2 0 002 2h11a2 2 0 002-2v-5m-1.414-9.414a2 2 0 112.828 2.828L11.828 15H9v-2.828l8.586-8.586z" />
                </svg>
              </div>
              <h3 className="text-lg font-medium text-gray-900 mb-2">Generate Content</h3>
              <p className="text-sm text-gray-500 mb-4">
                Create new documents and Confluence pages with AI assistance
              </p>
              <button className="bg-purple-600 text-white px-4 py-2 rounded-md hover:bg-purple-700 transition-colors">
                Generate Now
              </button>
            </div>
          </div>
        </div>

        {/* Quick Stats */}
        <div className="bg-white rounded-lg shadow p-6">
          <h2 className="text-xl font-bold text-gray-900 mb-4">Quick Overview</h2>
          <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
            <div className="text-center">
              <div className="text-2xl font-bold text-blue-600">0</div>
              <div className="text-sm text-gray-500">Documents Uploaded</div>
            </div>
            <div className="text-center">
              <div className="text-2xl font-bold text-green-600">0</div>
              <div className="text-sm text-gray-500">Chat Sessions</div>
            </div>
            <div className="text-center">
              <div className="text-2xl font-bold text-purple-600">0</div>
              <div className="text-sm text-gray-500">Pages Generated</div>
            </div>
            <div className="text-center">
              <div className="text-2xl font-bold text-orange-600">Offline</div>
              <div className="text-sm text-gray-500">AI Model Status</div>
            </div>
          </div>
        </div>

        {/* Getting Started */}
        <div className="mt-8 bg-blue-50 rounded-lg p-6">
          <h2 className="text-xl font-bold text-gray-900 mb-4">Getting Started</h2>
          <div className="space-y-3">
            <div className="flex items-center">
              <div className="flex-shrink-0 w-6 h-6 bg-blue-600 text-white rounded-full flex items-center justify-center text-sm font-medium">1</div>
              <div className="ml-3 text-gray-700">Upload your first document to get started</div>
            </div>
            <div className="flex items-center">
              <div className="flex-shrink-0 w-6 h-6 bg-blue-600 text-white rounded-full flex items-center justify-center text-sm font-medium">2</div>
              <div className="ml-3 text-gray-700">Configure Confluence integration (optional)</div>
            </div>
            <div className="flex items-center">
              <div className="flex-shrink-0 w-6 h-6 bg-blue-600 text-white rounded-full flex items-center justify-center text-sm font-medium">3</div>
              <div className="ml-3 text-gray-700">Start chatting with your documents</div>
            </div>
            <div className="flex items-center">
              <div className="flex-shrink-0 w-6 h-6 bg-blue-600 text-white rounded-full flex items-center justify-center text-sm font-medium">4</div>
              <div className="ml-3 text-gray-700">Generate new content based on your knowledge base</div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default HomePage; 