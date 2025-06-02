import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import { Toaster } from 'react-hot-toast';
import './App.css';

// Components
import Layout from './components/Layout';
import HomePage from './pages/HomePage';
import DocumentsPage from './pages/DocumentsPage';
import ChatPage from './pages/ChatPage';
import ConfluencePage from './pages/ConfluencePage';
import SettingsPage from './pages/SettingsPage';

function App() {
  return (
    <div className="App">
      <Router>
        <Layout>
          <Routes>
            <Route path="/" element={<HomePage />} />
            <Route path="/documents" element={<DocumentsPage />} />
            <Route path="/chat" element={<ChatPage />} />
            <Route path="/confluence" element={<ConfluencePage />} />
            <Route path="/settings" element={<SettingsPage />} />
          </Routes>
        </Layout>
      </Router>
      <Toaster 
        position="top-right"
        toastOptions={{
          duration: 4000,
          style: {
            background: '#363636',
            color: '#fff',
          },
          success: {
            duration: 3000,
          },
        }}
      />
    </div>
  );
}

export default App;
