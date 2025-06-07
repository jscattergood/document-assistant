interface ChatMessage {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  timestamp: Date;
  error?: boolean;
}

interface ChatSession {
  id: string;
  name: string;
  createdAt: Date;
  updatedAt: Date;
  messages: ChatMessage[];
}

class ChatStorageManager {
  private dbName = 'DocumentAssistantDB';
  private dbVersion = 1;
  private db: IDBDatabase | null = null;

  async init(): Promise<void> {
    return new Promise((resolve, reject) => {
      const request = indexedDB.open(this.dbName, this.dbVersion);

      request.onerror = () => {
        console.error('Failed to open IndexedDB');
        reject(new Error('Failed to open IndexedDB'));
      };

      request.onsuccess = (event) => {
        this.db = (event.target as IDBOpenDBRequest).result;
        resolve();
      };

      request.onupgradeneeded = (event) => {
        const db = (event.target as IDBOpenDBRequest).result;

        // Create chat sessions store
        if (!db.objectStoreNames.contains('chatSessions')) {
          const sessionStore = db.createObjectStore('chatSessions', { keyPath: 'id' });
          sessionStore.createIndex('createdAt', 'createdAt', { unique: false });
          sessionStore.createIndex('updatedAt', 'updatedAt', { unique: false });
        }

        // Create chat messages store
        if (!db.objectStoreNames.contains('chatMessages')) {
          const messageStore = db.createObjectStore('chatMessages', { keyPath: 'id' });
          messageStore.createIndex('sessionId', 'sessionId', { unique: false });
          messageStore.createIndex('timestamp', 'timestamp', { unique: false });
        }
      };
    });
  }

  private async ensureDB(): Promise<IDBDatabase> {
    if (!this.db) {
      await this.init();
    }
    if (!this.db) {
      throw new Error('Database not initialized');
    }
    return this.db;
  }

  // Generate a default session name based on timestamp
  private generateSessionName(timestamp: Date): string {
    const today = new Date();
    const isToday = timestamp.toDateString() === today.toDateString();
    const yesterday = new Date(today.getTime() - 24 * 60 * 60 * 1000);
    const isYesterday = timestamp.toDateString() === yesterday.toDateString();

    if (isToday) {
      return `Today's Chat - ${timestamp.toLocaleTimeString('en-US', { 
        hour: '2-digit', 
        minute: '2-digit' 
      })}`;
    } else if (isYesterday) {
      return `Yesterday's Chat - ${timestamp.toLocaleTimeString('en-US', { 
        hour: '2-digit', 
        minute: '2-digit' 
      })}`;
    } else {
      return `Chat - ${timestamp.toLocaleDateString('en-US', { 
        month: 'short', 
        day: 'numeric' 
      })} ${timestamp.toLocaleTimeString('en-US', { 
        hour: '2-digit', 
        minute: '2-digit' 
      })}`;
    }
  }

  async saveMessages(sessionId: string, messages: ChatMessage[]): Promise<void> {
    const db = await this.ensureDB();
    const transaction = db.transaction(['chatSessions', 'chatMessages'], 'readwrite');
    
    try {
      const sessionStore = transaction.objectStore('chatSessions');
      const messageStore = transaction.objectStore('chatMessages');

      // Get or create session
      let session: ChatSession;
      const existingSession = await new Promise<ChatSession | undefined>((resolve) => {
        const request = sessionStore.get(sessionId);
        request.onsuccess = () => resolve(request.result);
        request.onerror = () => resolve(undefined);
      });

      const now = new Date();

      if (existingSession) {
        session = {
          ...existingSession,
          updatedAt: now,
          messages: messages,
        };
      } else {
        session = {
          id: sessionId,
          name: this.generateSessionName(now),
          createdAt: now,
          updatedAt: now,
          messages: messages,
        };
      }

      // Save session
      sessionStore.put(session);

      // Clear existing messages for this session
      await new Promise<void>((resolve) => {
        const messageIndex = messageStore.index('sessionId');
        const messageCursor = messageIndex.openCursor(IDBKeyRange.only(sessionId));
        let deletionCount = 0;
        
        messageCursor.onsuccess = (event) => {
          const cursor = (event.target as IDBRequest).result;
          if (cursor) {
            cursor.delete();
            deletionCount++;
            cursor.continue();
          } else {
            // All existing messages have been processed

            resolve();
          }
        };
        
        messageCursor.onerror = () => {
          console.error('Error clearing existing messages');
          resolve(); // Still resolve to continue with saving new messages
        };
      });

      // Save new messages
      messages.forEach((message) => {
        messageStore.put({
          ...message,
          sessionId,
          timestamp: new Date(message.timestamp), // Ensure it's a Date object
        });
      });
      


      await new Promise<void>((resolve, reject) => {
        transaction.oncomplete = () => resolve();
        transaction.onerror = () => reject(transaction.error);
      });
    } catch (error) {
      console.error('Error saving messages:', error);
      throw error;
    }
  }

  async loadMessages(sessionId: string): Promise<ChatMessage[]> {
    const db = await this.ensureDB();
    const transaction = db.transaction('chatMessages', 'readonly');
    const messageStore = transaction.objectStore('chatMessages');
    const index = messageStore.index('sessionId');

    return new Promise((resolve, reject) => {
      const messages: ChatMessage[] = [];
      const request = index.openCursor(IDBKeyRange.only(sessionId));

      request.onsuccess = (event) => {
        const cursor = (event.target as IDBRequest).result;
        if (cursor) {
          const message = cursor.value;
          messages.push({
            id: message.id,
            role: message.role,
            content: message.content,
            timestamp: new Date(message.timestamp), // Convert back to Date
            error: message.error,
          });
          cursor.continue();
        } else {
          // Sort messages by timestamp
          messages.sort((a, b) => a.timestamp.getTime() - b.timestamp.getTime());

          resolve(messages);
        }
      };

      request.onerror = () => {
        console.error(`Error loading messages for session ${sessionId}:`, request.error);
        reject(request.error);
      };
    });
  }

  async getAllSessions(): Promise<ChatSession[]> {
    const db = await this.ensureDB();
    const transaction = db.transaction('chatSessions', 'readonly');
    const sessionStore = transaction.objectStore('chatSessions');

    return new Promise((resolve, reject) => {
      const sessions: ChatSession[] = [];
      const request = sessionStore.openCursor();

      request.onsuccess = (event) => {
        const cursor = (event.target as IDBRequest).result;
        if (cursor) {
          const session = cursor.value;
          sessions.push({
            ...session,
            createdAt: new Date(session.createdAt),
            updatedAt: new Date(session.updatedAt),
          });
          cursor.continue();
        } else {
          // Sort sessions by updatedAt (most recent first)
          sessions.sort((a, b) => b.updatedAt.getTime() - a.updatedAt.getTime());
          resolve(sessions);
        }
      };

      request.onerror = () => reject(request.error);
    });
  }

  async deleteSession(sessionId: string): Promise<void> {
    const db = await this.ensureDB();
    const transaction = db.transaction(['chatSessions', 'chatMessages'], 'readwrite');
    
    try {
      const sessionStore = transaction.objectStore('chatSessions');
      const messageStore = transaction.objectStore('chatMessages');

      // Delete session
      sessionStore.delete(sessionId);

      // Delete all messages for this session
      const messageIndex = messageStore.index('sessionId');
      const messageCursor = messageIndex.openCursor(IDBKeyRange.only(sessionId));
      messageCursor.onsuccess = (event) => {
        const cursor = (event.target as IDBRequest).result;
        if (cursor) {
          cursor.delete();
          cursor.continue();
        }
      };

      await new Promise<void>((resolve, reject) => {
        transaction.oncomplete = () => resolve();
        transaction.onerror = () => reject(transaction.error);
      });
    } catch (error) {
      console.error('Error deleting session:', error);
      throw error;
    }
  }

  async renameSession(sessionId: string, newName: string): Promise<void> {
    const db = await this.ensureDB();
    const transaction = db.transaction('chatSessions', 'readwrite');
    const sessionStore = transaction.objectStore('chatSessions');

    return new Promise((resolve, reject) => {
      const request = sessionStore.get(sessionId);
      
      request.onsuccess = () => {
        const session = request.result;
        if (session) {
          session.name = newName;
          session.updatedAt = new Date();
          
          const updateRequest = sessionStore.put(session);
          updateRequest.onsuccess = () => resolve();
          updateRequest.onerror = () => reject(updateRequest.error);
        } else {
          reject(new Error('Session not found'));
        }
      };

      request.onerror = () => reject(request.error);
    });
  }

  // Get the current session ID (today's session)
  getCurrentSessionId(): string {
    const today = new Date();
    return `session-${today.getFullYear()}-${today.getMonth() + 1}-${today.getDate()}`;
  }

  // Create a new session ID
  createNewSessionId(): string {
    return `session-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
  }
}

// Create a singleton instance
export const chatStorage = new ChatStorageManager();

// Initialize on import
chatStorage.init().catch(console.error);

export type { ChatMessage, ChatSession }; 