import React, { ReactNode } from 'react';
import { Link, useLocation } from 'react-router-dom';
import {
  HomeIcon,
  DocumentTextIcon,
  ChatBubbleLeftRightIcon,
  CloudIcon,
  CogIcon,
} from '@heroicons/react/24/outline';
import classNames from 'classnames';

interface LayoutProps {
  children: ReactNode;
}

const navigation = [
  { name: 'Home', href: '/', icon: HomeIcon },
  { name: 'Documents', href: '/documents', icon: DocumentTextIcon },
  { name: 'Chat', href: '/chat', icon: ChatBubbleLeftRightIcon },
  { name: 'Confluence', href: '/confluence', icon: CloudIcon },
  { name: 'Settings', href: '/settings', icon: CogIcon },
];

const Layout: React.FC<LayoutProps> = ({ children }) => {
  const location = useLocation();

  return (
    <div className="min-h-screen bg-gray-50">
      <div className="flex">
        {/* Sidebar */}
        <div className="fixed inset-y-0 left-0 z-50 w-64 bg-white shadow-lg">
          <div className="flex h-full flex-col">
            {/* Logo */}
            <div className="flex items-center px-6 py-4">
              <DocumentTextIcon className="h-8 w-8 text-blue-600" />
              <span className="ml-2 text-xl font-bold text-gray-900">
                Document Assistant
              </span>
            </div>

            {/* Navigation */}
            <nav className="flex-1 space-y-1 px-3 py-4">
              {navigation.map((item) => {
                const isActive = location.pathname === item.href;
                return (
                  <Link
                    key={item.name}
                    to={item.href}
                    className={classNames(
                      'group flex items-center px-3 py-2 text-sm font-medium rounded-md transition-colors',
                      {
                        'bg-blue-50 text-blue-700 border-r-2 border-blue-700': isActive,
                        'text-gray-600 hover:bg-gray-50 hover:text-gray-900': !isActive,
                      }
                    )}
                  >
                    <item.icon
                      className={classNames(
                        'mr-3 h-5 w-5 flex-shrink-0',
                        {
                          'text-blue-500': isActive,
                          'text-gray-400 group-hover:text-gray-500': !isActive,
                        }
                      )}
                    />
                    {item.name}
                  </Link>
                );
              })}
            </nav>

            {/* Footer */}
            <div className="p-4 border-t border-gray-200">
              <div className="text-xs text-gray-500 text-center">
                Document Assistant v1.0.0
                <br />
                Powered by LlamaIndex & GPT4All
              </div>
            </div>
          </div>
        </div>

        {/* Main content */}
        <div className="ml-64 flex-1">
          <main className="min-h-screen">
            {children}
          </main>
        </div>
      </div>
    </div>
  );
};

export default Layout; 