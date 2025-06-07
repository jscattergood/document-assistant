import { useState, useCallback, useRef, useEffect } from 'react';
import { chatAPI, type JobStatusResponse } from '../services/api';
import { chatStorage, type ChatMessage } from '../utils/chatStorage';
import toast from 'react-hot-toast';

interface BackgroundChatState {
  jobId: string | null;
  status: 'idle' | 'pending' | 'processing' | 'completed' | 'failed';
  response: string | null;
  error: string | null;
  sessionId?: string;
}

export const useBackgroundChat = (sessionId?: string) => {
  const [state, setState] = useState<BackgroundChatState>(() => {
    // Try to restore state from sessionStorage on initialization
    const saved = sessionStorage.getItem('backgroundChatState');
    if (saved) {
      try {
        const parsed = JSON.parse(saved);
        // Only restore if there's an active job
        if (parsed.jobId && (parsed.status === 'pending' || parsed.status === 'processing')) {
          return parsed;
        }
      } catch (error) {
        console.error('Error parsing saved background chat state:', error);
      }
    }
    return {
      jobId: null,
      status: 'idle',
      response: null,
      error: null,
      sessionId,
    };
  });

  const pollingIntervalRef = useRef<number | null>(null);

  // Save state to sessionStorage whenever it changes
  useEffect(() => {
    sessionStorage.setItem('backgroundChatState', JSON.stringify(state));
  }, [state]);

  const startPolling = useCallback((jobId: string) => {
    if (pollingIntervalRef.current) {
      clearInterval(pollingIntervalRef.current);
    }

    let isMounted = true;

    pollingIntervalRef.current = setInterval(async () => {
      try {
        const jobStatus = await chatAPI.getJobStatus(jobId);
        
        if (!isMounted) return; // Don't update state if component unmounted
        
        setState(prev => ({
          ...prev,
          status: jobStatus.status,
        }));

        if (jobStatus.status === 'completed') {
          // Save response directly to IndexedDB regardless of component mount status
          const currentSessionId = sessionId || sessionStorage.getItem('currentChatSessionId');
          
          if (currentSessionId && jobStatus.response) {
            // Create assistant message
            const assistantMessage: ChatMessage = {
              id: `assistant-${Date.now()}`,
              role: 'assistant',
              content: jobStatus.response,
              timestamp: new Date(),
            };

            // Save the response immediately using async/await for better error handling
            (async () => {
              try {
                const currentMessages = await chatStorage.loadMessages(currentSessionId);
                const updatedMessages = [...currentMessages, assistantMessage];
                await chatStorage.saveMessages(currentSessionId, updatedMessages);
                
                // Fire a custom event to notify components that messages were updated
                window.dispatchEvent(new CustomEvent('backgroundChatUpdated', { 
                  detail: { sessionId: currentSessionId, type: 'completed' } 
                }));
                
                // Update state to trigger UI refresh
                if (isMounted) {
                  setState(prev => ({
                    ...prev,
                    response: jobStatus.response || null,
                  }));
                }
              } catch (error) {
                console.error('Error saving background response to storage:', error);
              }
            })();
          }
          
          if (pollingIntervalRef.current) {
            clearInterval(pollingIntervalRef.current);
            pollingIntervalRef.current = null;
          }

          // Show notification
          if ('Notification' in window && Notification.permission === 'granted') {
            new Notification('Chat Response Ready', {
              body: 'Your AI response is ready!',
              icon: '/favicon.svg',
            });
          } else {
            toast.success('Your AI response is ready!');
          }

          // Clean up job after successful completion
          setTimeout(() => {
            chatAPI.cleanupJob(jobId).catch(console.error);
          }, 1000);

        } else if (jobStatus.status === 'failed') {
          // Save error message directly to IndexedDB regardless of component mount status
          const currentSessionId = sessionId || sessionStorage.getItem('currentChatSessionId');
          
          if (currentSessionId) {
            // Create error message
            const errorMessage: ChatMessage = {
              id: `error-${Date.now()}`,
              role: 'assistant',
              content: `Sorry, I encountered an error: ${jobStatus.error || 'Unknown error'}`,
              timestamp: new Date(),
              error: true,
            };

            // Save the error immediately using async/await for better error handling
            (async () => {
              try {
                const currentMessages = await chatStorage.loadMessages(currentSessionId);
                const updatedMessages = [...currentMessages, errorMessage];
                await chatStorage.saveMessages(currentSessionId, updatedMessages);
                
                // Fire a custom event to notify components that messages were updated
                window.dispatchEvent(new CustomEvent('backgroundChatUpdated', { 
                  detail: { sessionId: currentSessionId, type: 'failed' } 
                }));
                
                // Update state to trigger UI refresh
                if (isMounted) {
                  setState(prev => ({
                    ...prev,
                    error: jobStatus.error || 'Unknown error occurred',
                  }));
                }
              } catch (error) {
                console.error('Error saving background error to storage:', error);
              }
            })();
          }
          
          if (pollingIntervalRef.current) {
            clearInterval(pollingIntervalRef.current);
            pollingIntervalRef.current = null;
          }

          toast.error('Chat processing failed: ' + (jobStatus.error || 'Unknown error'));

          // Clean up failed job
          setTimeout(() => {
            chatAPI.cleanupJob(jobId).catch(console.error);
          }, 1000);
        }
      } catch (error) {
        console.error('Error polling job status:', error);
        if (isMounted) {
          toast.error('Error checking chat status');
        }
        
        if (pollingIntervalRef.current) {
          clearInterval(pollingIntervalRef.current);
          pollingIntervalRef.current = null;
        }
      }
    }, 2000); // Poll every 2 seconds

    // Return cleanup function to mark as unmounted
    return () => {
      isMounted = false;
    };
  }, []);

  // Resume polling if there's an active job on mount and cleanup on unmount
  useEffect(() => {
    const initialJobId = state.jobId;
    const initialStatus = state.status;
    
    let cleanup: (() => void) | undefined;
    
    if (initialJobId && (initialStatus === 'pending' || initialStatus === 'processing')) {
      cleanup = startPolling(initialJobId);
    }
    
    // Cleanup polling on unmount
    return () => {
      if (cleanup) {
        cleanup();
      }
      if (pollingIntervalRef.current) {
        clearInterval(pollingIntervalRef.current);
        pollingIntervalRef.current = null;
      }
    };
  }, []); // Only run on mount

  const startBackgroundChat = useCallback(async (
    message: string,
    conversationHistory?: Array<{role: string; content: string}>,
    documentIds?: string[]
  ) => {
    try {
      // Request notification permission if not already granted
      if ('Notification' in window && Notification.permission === 'default') {
        await Notification.requestPermission();
      }

      const jobResponse = await chatAPI.startBackgroundChat(
        message,
        conversationHistory,
        documentIds
      );

      // Store current session ID for background processing
      if (sessionId) {
        sessionStorage.setItem('currentChatSessionId', sessionId);
      }

      setState({
        jobId: jobResponse.job_id,
        status: 'pending',
        response: null,
        error: null,
        sessionId,
      });

      toast.success('Chat request sent! Processing in background...');
      const cleanup = startPolling(jobResponse.job_id);

      return jobResponse.job_id;
    } catch (error) {
      console.error('Error starting background chat:', error);
      toast.error('Failed to start background chat');
      throw error;
    }
  }, [startPolling]);

  const cancelJob = useCallback(() => {
    if (pollingIntervalRef.current) {
      clearInterval(pollingIntervalRef.current);
      pollingIntervalRef.current = null;
    }

    if (state.jobId) {
      chatAPI.cleanupJob(state.jobId).catch(console.error);
    }

    setState({
      jobId: null,
      status: 'idle',
      response: null,
      error: null,
    });
  }, [state.jobId]);

  const reset = useCallback(() => {
    if (pollingIntervalRef.current) {
      clearInterval(pollingIntervalRef.current);
      pollingIntervalRef.current = null;
    }

    setState({
      jobId: null,
      status: 'idle',
      response: null,
      error: null,
    });
  }, []);

  return {
    ...state,
    startBackgroundChat,
    cancelJob,
    reset,
    isProcessing: state.status === 'pending' || state.status === 'processing',
  };
}; 